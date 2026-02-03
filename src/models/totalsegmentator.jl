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
    # Standard nnU-Net uses Instance Normalization
    return ConvBlock(
        Conv(kernel, in_chs => out_chs, stride=stride, pad=pad, use_bias=true), 

        # Actually standard nnU-Net: Conv (bias=True) -> InstNorm (affine=True) -> LeakyReLU.
        # But wait, usually bias is redundant with affine norm.
        # Let's check state_dict keys later. For now assume bias=True.
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

A sequence of ConvBlocks. The first one handles the stride (downsampling).
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


"""
    UNetEncoder(input_dim, base_features, stage_strides, stage_kernels)

"""
struct UNetEncoder{L} <: Lux.AbstractLuxLayer
    stages::L
end

function UNetEncoder(input_dim::Int, base_features::Int, stage_strides, stage_kernels, num_convs_per_stage)
    stages = []
    current_features = input_dim
    output_features = base_features
    
    for i in 1:length(stage_strides)
        stride = stage_strides[i]
        kernel = stage_kernels[i]
        n_convs = num_convs_per_stage[i]
        
        # Calculate features (doubling every stage usually, maxing at 320)
        # But we need to follow the exact plan.
        # Plan says: 32 -> 64 -> 128 -> 256 -> 320 -> 320
        
        push!(stages, StackedConvLayers(current_features, output_features, kernel, n_convs; first_stride=stride))
        
        current_features = output_features
        # Update for next stage
        output_features = min(output_features * 2, 320)
    end
    
    return UNetEncoder(Chain(stages...))
end

# Need to return intermediate skip connections!
function (l::UNetEncoder)(x, ps, st)
    skips = []
    current_input = x
    
    # Iterate manually through Chain to collect skips
    # Chain doesn't expose list iteration easily if it is a NamedTuple.
    # But Lux.Chain acts like a function.
    # We need to construct it carefully or iterate internal layers.
    
    # Better: define Encoder as a struct with a vector of layers needed or use logic here.
    # For now, let's assume `l.stages` is a Chain.
    
    st_stages = st.stages
    ps_stages = ps.stages
    
    # We can iterate keys if names are predictable (layer_1, etc.)
    # Or just store them in a Tuple in the struct.
    
    # Actually, for the Encoder, we want the output of EACH stage to be a skip connection,
    # EXCEPT usually the last one goes to the bridge/bottleneck.
    # But usually the bottleneck is just the last encoder stage.
    
    # Let's inspect how Lux Chain works.
    # We can unwrap the layers.
    
    keys_list = keys(l.stages)
    for (i, key) in enumerate(keys_list)
        layer = l.stages[key]
        p = ps_stages[key]
        s = st_stages[key]
        
        current_input, s_new = layer(current_input, p, s)
        
        # Update state
        # st_stages = merge(st_stages, (key => s_new,)) # Immutable update - inefficient in loop?
        # Ideally we collect states and reconstruct.
        # But actually, specific implementation of applying chain is better.
        
        # Store skip
        push!(skips, current_input)
    end
    
    # We need to reconstruct the full state to return it validly
    # This loop above is pseudo-code for state.
    # Real implementation needs proper state handling.
    
    # Simplification: Use `Lux.applychain` logic or just hardcode the forward pass if number of stages is fixed (5).
    # Since we are building a specific network, hardcoding stages is safer for explicit state management.
    
    return skips, st
end


# Custom "Manual" definition for better control over skips
struct TotalSegmentatorEncoder{S1, S2, S3, S4, S5} <: Lux.AbstractLuxLayer
    stage1::S1
    stage2::S2
    stage3::S3
    stage4::S4
    stage5::S5
end

function TotalSegmentatorEncoder()
    # Based on plans.json
    # Stages strides: [1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]
    # Kernels: [3,3,3]...
    
    return TotalSegmentatorEncoder(
        StackedConvLayers(1, 32, (3,3,3), 2; first_stride=(1,1,1)),
        StackedConvLayers(32, 64, (3,3,3), 2; first_stride=(2,2,2)),
        StackedConvLayers(64, 128, (3,3,3), 2; first_stride=(2,2,2)),
        StackedConvLayers(128, 256, (3,3,3), 2; first_stride=(2,2,2)),
        StackedConvLayers(256, 320, (3,3,3), 2; first_stride=(2,2,2))
    )
end

Lux.initialparameters(rng::AbstractRNG, l::TotalSegmentatorEncoder) = (
    stage1=Lux.initialparameters(rng, l.stage1),
    stage2=Lux.initialparameters(rng, l.stage2),
    stage3=Lux.initialparameters(rng, l.stage3),
    stage4=Lux.initialparameters(rng, l.stage4),
    stage5=Lux.initialparameters(rng, l.stage5)
)

Lux.initialstates(rng::AbstractRNG, l::TotalSegmentatorEncoder) = (
    stage1=Lux.initialstates(rng, l.stage1),
    stage2=Lux.initialstates(rng, l.stage2),
    stage3=Lux.initialstates(rng, l.stage3),
    stage4=Lux.initialstates(rng, l.stage4),
    stage5=Lux.initialstates(rng, l.stage5)
)

function (l::TotalSegmentatorEncoder)(x, ps, st)
    s1, st1 = l.stage1(x, ps.stage1, st.stage1)
    s2, st2 = l.stage2(s1, ps.stage2, st.stage2)
    s3, st3 = l.stage3(s2, ps.stage3, st.stage3)
    s4, st4 = l.stage4(s3, ps.stage4, st.stage4)
    s5, st5 = l.stage5(s4, ps.stage5, st.stage5)
    
    return (s1, s2, s3, s4, s5), (stage1=st1, stage2=st2, stage3=st3, stage4=st4, stage5=st5)
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
    # Transposed Conv for upsampling
    # Usually: ConvTranspose3d(in, out, kernel, stride, padding)
    # nnU-Net uses Transposed Conv with kernel=2, stride=2 usually.
    upsample = ConvTranspose((2,2,2), in_chs => out_chs, stride=stride, use_bias=true) 
    
    # Combined channels = out_chs (from upsample) + skip_chs
    # Then convs to reduce to out_chs
    convs = StackedConvLayers(out_chs + skip_chs, out_chs, (3,3,3), 2; first_stride=(1,1,1))
    
    return DecoderBlock(upsample, convs)
end

Lux.initialparameters(rng::AbstractRNG, l::DecoderBlock) = (upsample=Lux.initialparameters(rng, l.upsample), convs=Lux.initialparameters(rng, l.convs))
Lux.initialstates(rng::AbstractRNG, l::DecoderBlock) = (upsample=Lux.initialstates(rng, l.upsample), convs=Lux.initialstates(rng, l.convs))

function (l::DecoderBlock)((x, skip), ps, st)
    # x is from deeper layer (lower res)
    # skip is from encoder (higher res)
    
    up, st_up = l.upsample(x, ps.upsample, st.upsample)
    
    # Center crop or pad might be needed if shapes don't match exactly?
    # nnU-Net usually ensures padding keeps shapes consistent.
    # Lux/NNlib should handle concatenation if dimensions match.
    
    # Concatenate along channel dim (dim 4 for WHDCN)
    
    concat = cat(up, skip; dims=4)
    
    out, st_convs = l.convs(concat, ps.convs, st.convs)
    
    return out, (upsample=st_up, convs=st_convs)
end


struct TotalSegmentatorDecoder{D1, D2, D3, D4} <: Lux.AbstractLuxLayer
    block1::D1 # Bottleneck -> Stage 4
    block2::D2 # -> Stage 3
    block3::D3 # -> Stage 2
    block4::D4 # -> Stage 1
end

function TotalSegmentatorDecoder()
    # Encoder Output features: 32, 64, 128, 256, 320
    # Bottleneck is Stage 5 (320 features)
    
    # Decoder 1: Input 320, Skip 256 (Stage 4). Out: 256
    d1 = DecoderBlock(320, 256, 256)
    
    # Decoder 2: Input 256, Skip 128 (Stage 3). Out: 128
    d2 = DecoderBlock(256, 128, 128)
    
    # Decoder 3: Input 128, Skip 64 (Stage 2). Out: 64
    d3 = DecoderBlock(128, 64, 64)
    
    # Decoder 4: Input 64, Skip 32 (Stage 1). Out: 32
    d4 = DecoderBlock(64, 32, 32)
    
    return TotalSegmentatorDecoder(d1, d2, d3, d4)
end

Lux.initialparameters(rng::AbstractRNG, l::TotalSegmentatorDecoder) = (
    block1=Lux.initialparameters(rng, l.block1),
    block2=Lux.initialparameters(rng, l.block2),
    block3=Lux.initialparameters(rng, l.block3),
    block4=Lux.initialparameters(rng, l.block4)
)

Lux.initialstates(rng::AbstractRNG, l::TotalSegmentatorDecoder) = (
    block1=Lux.initialstates(rng, l.block1),
    block2=Lux.initialstates(rng, l.block2),
    block3=Lux.initialstates(rng, l.block3),
    block4=Lux.initialstates(rng, l.block4)
)

function (l::TotalSegmentatorDecoder)((x, skips), ps, st)
    # x is output of Stage 5
    # skips is (s1, s2, s3, s4, s5)
    # We use s4, s3, s2, s1 for skips
    s1, s2, s3, s4, s5 = skips
    
    d1, st1 = l.block1((x, s4), ps.block1, st.block1)
    d2, st2 = l.block2((d1, s3), ps.block2, st.block2)
    d3, st3 = l.block3((d2, s2), ps.block3, st.block3)
    d4, st4 = l.block4((d3, s1), ps.block4, st.block4)
    
    return d4, (block1=st1, block2=st2, block3=st3, block4=st4)
end


struct TotalSegmentator <: Lux.AbstractLuxLayer
    encoder::TotalSegmentatorEncoder
    decoder::TotalSegmentatorDecoder
    final_conv::Conv # 1x1x1 conv to num_classes
end

function TotalSegmentator(num_classes=118)
    return TotalSegmentator(
        TotalSegmentatorEncoder(),
        TotalSegmentatorDecoder(),
        Conv((1,1,1), 32 => num_classes) # Final 1x1 conv
    )
end

Lux.initialparameters(rng::AbstractRNG, l::TotalSegmentator) = (
    encoder=Lux.initialparameters(rng, l.encoder),
    decoder=Lux.initialparameters(rng, l.decoder),
    final_conv=Lux.initialparameters(rng, l.final_conv)
)

Lux.initialstates(rng::AbstractRNG, l::TotalSegmentator) = (
    encoder=Lux.initialstates(rng, l.encoder),
    decoder=Lux.initialstates(rng, l.decoder),
    final_conv=Lux.initialstates(rng, l.final_conv)
)

function (l::TotalSegmentator)(x, ps, st)
    # Encoder
    skips, st_enc = l.encoder(x, ps.encoder, st.encoder)
    
    # Decomposition
    s1, s2, s3, s4, s5 = skips
    bottleneck = s5
    
    # Decoder
    y, st_dec = l.decoder((bottleneck, skips), ps.decoder, st.decoder)
    
    # Final Conv
    out, st_final = l.final_conv(y, ps.final_conv, st.final_conv)
    
    return out, (encoder=st_enc, decoder=st_dec, final_conv=st_final)
end
