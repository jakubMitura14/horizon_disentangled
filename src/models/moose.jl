using Lux
using Random
using NNlib
using ComponentArrays
using SafeTensors
using Adapt

# --- Architecture Definition ---

"""
    ConvBlock(in_chs, out_chs, kernel, stride; activation=leakyrelu)

Basic building block: Conv -> InstanceNorm -> Activation
"""
struct ConvBlock{C, N, A} <: Lux.AbstractLuxLayer
    conv::C
    norm::N
    activation::A
end

function ConvBlock(in_chs::Int, out_chs::Int, kernel::Tuple, stride::Tuple=size(kernel) .÷ 2 .+ 1; activation=leakyrelu)
    pad = kernel .÷ 2
    return ConvBlock(
        Conv(kernel, in_chs => out_chs, stride=stride, pad=pad, use_bias=true), 
        InstanceNorm(out_chs; affine=true),
        activation
    )
end

Lux.initialparameters(rng::AbstractRNG, l::ConvBlock) = (conv=Lux.initialparameters(rng, l.conv), norm=Lux.initialparameters(rng, l.norm))
Lux.initialstates(rng::AbstractRNG, l::ConvBlock) = (conv=Lux.initialstates(rng, l.conv), norm=Lux.initialstates(rng, l.norm))

function (l::ConvBlock)(x, ps, st)
    x, st_conv = l.conv(x, ps.conv, st.conv)
    x, st_norm = l.norm(x, ps.norm, st.norm)
    return l.activation.(x), (conv=st_conv, norm=st_norm)
end


"""
    StackedConvLayers(in_chs, out_chs, kernel, num_convs; first_stride=1)
"""
struct StackedConvLayers{L} <: Lux.AbstractLuxLayer
    layers::L
end

function StackedConvLayers(in_chs::Int, out_chs::Int, kernel::Tuple, num_convs::Int; first_stride=1)
    layers = []
    # First layer does the stride (if any) and channel change
    push!(layers, ConvBlock(in_chs, out_chs, kernel, first_stride))
    
    # Subsequent layers keep dimensions
    for _ in 2:num_convs
        push!(layers, ConvBlock(out_chs, out_chs, kernel, (1,1,1))) # Stride 1
    end
    return StackedConvLayers(Chain(layers...))
end

Lux.initialparameters(rng::AbstractRNG, l::StackedConvLayers) = Lux.initialparameters(rng, l.layers)
Lux.initialstates(rng::AbstractRNG, l::StackedConvLayers) = Lux.initialstates(rng, l.layers)
(l::StackedConvLayers)(x, ps, st) = l.layers(x, ps, st)


# Custom "Manual" definition for better control over skips
# MOOSE has 6 Stages
struct MooseEncoder{S1, S2, S3, S4, S5, S6} <: Lux.AbstractLuxLayer
    stage1::S1
    stage2::S2
    stage3::S3
    stage4::S4
    stage5::S5
    stage6::S6
end

function MooseEncoder()
    # Based on plans.json for clin_pt_fdg_brain_v1 (3d_fullres)
    # Strides: [1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,1,2]
    # Kernels: [3,3,3]...
    # Features: Base 32, Max 320.
    # Stage 1: 1 -> 32
    # Stage 2: 32 -> 64
    # Stage 3: 64 -> 128
    # Stage 4: 128 -> 256
    # Stage 5: 256 -> 320 (capped at 320?)
    # Stage 6: 320 -> 320
    
    # NOTE: Input channels 1 (CT/PET single channel usually? Or checking input)
    # Setup said dummy input (1, 1, 64...), so 1 channel.
    
    return MooseEncoder(
        StackedConvLayers(1, 32, (3,3,3), 2; first_stride=(1,1,1)),
        StackedConvLayers(32, 64, (3,3,3), 2; first_stride=(2,2,2)),
        StackedConvLayers(64, 128, (3,3,3), 2; first_stride=(2,2,2)),
        StackedConvLayers(128, 256, (3,3,3), 2; first_stride=(2,2,2)),
        StackedConvLayers(256, 320, (3,3,3), 2; first_stride=(2,2,2)),
        StackedConvLayers(320, 320, (3,3,3), 2; first_stride=(2,1,2))
    )
end

Lux.initialparameters(rng::AbstractRNG, l::MooseEncoder) = (
    stage1=Lux.initialparameters(rng, l.stage1),
    stage2=Lux.initialparameters(rng, l.stage2),
    stage3=Lux.initialparameters(rng, l.stage3),
    stage4=Lux.initialparameters(rng, l.stage4),
    stage5=Lux.initialparameters(rng, l.stage5),
    stage6=Lux.initialparameters(rng, l.stage6)
)

Lux.initialstates(rng::AbstractRNG, l::MooseEncoder) = (
    stage1=Lux.initialstates(rng, l.stage1),
    stage2=Lux.initialstates(rng, l.stage2),
    stage3=Lux.initialstates(rng, l.stage3),
    stage4=Lux.initialstates(rng, l.stage4),
    stage5=Lux.initialstates(rng, l.stage5),
    stage6=Lux.initialstates(rng, l.stage6)
)

function (l::MooseEncoder)(x, ps, st)
    s1, st1 = l.stage1(x, ps.stage1, st.stage1)
    s2, st2 = l.stage2(s1, ps.stage2, st.stage2)
    s3, st3 = l.stage3(s2, ps.stage3, st.stage3)
    s4, st4 = l.stage4(s3, ps.stage4, st.stage4)
    s5, st5 = l.stage5(s4, ps.stage5, st.stage5)
    s6, st6 = l.stage6(s5, ps.stage6, st.stage6)
    
    # Skips: s1, s2, s3, s4, s5.
    # Bottleneck: s6
    return (s1, s2, s3, s4, s5, s6), (stage1=st1, stage2=st2, stage3=st3, stage4=st4, stage5=st5, stage6=st6)
end


"""
    DecoderBlock
    
    Upsample -> Concat(Skip) -> ConvBlock -> ConvBlock
"""
struct DecoderBlock{U, C} <: Lux.AbstractLuxLayer
    upsample::U
    convs::C
end

function DecoderBlock(in_chs, skip_chs, out_chs, stride=(2,2,2))
    upsample = ConvTranspose((2,2,2), in_chs => out_chs, stride=stride, use_bias=true) 
    
    convs = StackedConvLayers(out_chs + skip_chs, out_chs, (3,3,3), 2; first_stride=(1,1,1))
    
    return DecoderBlock(upsample, convs)
end

Lux.initialparameters(rng::AbstractRNG, l::DecoderBlock) = (upsample=Lux.initialparameters(rng, l.upsample), convs=Lux.initialparameters(rng, l.convs))
Lux.initialstates(rng::AbstractRNG, l::DecoderBlock) = (upsample=Lux.initialstates(rng, l.upsample), convs=Lux.initialstates(rng, l.convs))

function (l::DecoderBlock)((x, skip), ps, st)
    up, st_up = l.upsample(x, ps.upsample, st.upsample)
    
    # Handle padding/cropping if shapes differ? 
    # nnU-Net typically relies on correct shapes, but if input is odd, upsample duplicates.
    # For now assume shapes match or Lux/NNlib handles broadcasting (it won't).
    # We might need to crop/pad in a robust implementation.
    
    concat = cat(up, skip; dims=4)
    out, st_convs = l.convs(concat, ps.convs, st.convs)
    
    return out, (upsample=st_up, convs=st_convs)
end


struct MooseDecoder{D1, D2, D3, D4, D5} <: Lux.AbstractLuxLayer
    block1::D1 # Bottleneck (S6) -> Stage 5
    block2::D2 # -> Stage 4
    block3::D3 # -> Stage 3
    block4::D4 # -> Stage 2
    block5::D5 # -> Stage 1
end

function MooseDecoder()
    # Encoder Output features: 32(S1), 64(S2), 128(S3), 256(S4), 320(S5), 320(S6)
    # Bottleneck is Stage 6 (320 features)
    
    # Decoder 1: Input 320 (S6), Skip 320 (S5). Out: 320. Stride (2,1,2) (Inverse of S6)
    # Wait, Transposed Conv stride must match Encoder Stride?
    # Encoder S6 stride was (2,1,2).
    d1 = DecoderBlock(320, 320, 320, (2,1,2))
    
    # Decoder 2: Input 320, Skip 256 (S4). Out: 256. Stride (2,2,2)
    d2 = DecoderBlock(320, 256, 256, (2,2,2))
    
    # Decoder 3: Input 256, Skip 128 (S3). Out: 128. Stride (2,2,2)
    d3 = DecoderBlock(256, 128, 128, (2,2,2))
    
    # Decoder 4: Input 128, Skip 64 (S2). Out: 64. Stride (2,2,2)
    d4 = DecoderBlock(128, 64, 64, (2,2,2))
    
    # Decoder 5: Input 64, Skip 32 (S1). Out: 32. Stride (2,2,2)
    d5 = DecoderBlock(64, 32, 32, (2,2,2))
    
    return MooseDecoder(d1, d2, d3, d4, d5)
end

Lux.initialparameters(rng::AbstractRNG, l::MooseDecoder) = (
    block1=Lux.initialparameters(rng, l.block1),
    block2=Lux.initialparameters(rng, l.block2),
    block3=Lux.initialparameters(rng, l.block3),
    block4=Lux.initialparameters(rng, l.block4),
    block5=Lux.initialparameters(rng, l.block5)
)

Lux.initialstates(rng::AbstractRNG, l::MooseDecoder) = (
    block1=Lux.initialstates(rng, l.block1),
    block2=Lux.initialstates(rng, l.block2),
    block3=Lux.initialstates(rng, l.block3),
    block4=Lux.initialstates(rng, l.block4),
    block5=Lux.initialstates(rng, l.block5)
)

function (l::MooseDecoder)((x, skips), ps, st)
    s1, s2, s3, s4, s5, s6 = skips
    
    # x is s6 (bottleneck)
    
    d1, st1 = l.block1((x, s5), ps.block1, st.block1)
    d2, st2 = l.block2((d1, s4), ps.block2, st.block2)
    d3, st3 = l.block3((d2, s3), ps.block3, st.block3)
    d4, st4 = l.block4((d3, s2), ps.block4, st.block4)
    d5, st5 = l.block5((d4, s1), ps.block5, st.block5)
    
    return d5, (block1=st1, block2=st2, block3=st3, block4=st4, block5=st5)
end


struct MooseModel <: Lux.AbstractLuxLayer
    encoder::MooseEncoder
    decoder::MooseDecoder
    final_conv::Conv
end

function MooseModel(num_classes=83)
    return MooseModel(
        MooseEncoder(),
        MooseDecoder(),
        Conv((1,1,1), 32 => num_classes)
    )
end

Lux.initialparameters(rng::AbstractRNG, l::MooseModel) = (
    encoder=Lux.initialparameters(rng, l.encoder),
    decoder=Lux.initialparameters(rng, l.decoder),
    final_conv=Lux.initialparameters(rng, l.final_conv)
)

Lux.initialstates(rng::AbstractRNG, l::MooseModel) = (
    encoder=Lux.initialstates(rng, l.encoder),
    decoder=Lux.initialstates(rng, l.decoder),
    final_conv=Lux.initialstates(rng, l.final_conv)
)

function (l::MooseModel)(x, ps, st)
    # Encoder
    skips, st_enc = l.encoder(x, ps.encoder, st.encoder)
    
    s1, s2, s3, s4, s5, s6 = skips
    bottleneck = s6
    
    # Decoder
    y, st_dec = l.decoder((bottleneck, skips), ps.decoder, st.decoder)
    
    # Final Conv
    out, st_final = l.final_conv(y, ps.final_conv, st.final_conv)
    
    return out, (encoder=st_enc, decoder=st_dec, final_conv=st_final)
end
