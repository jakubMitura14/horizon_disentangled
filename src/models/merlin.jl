using Lux
using Random
using NNlib

# --- Building Blocks ---

"""
    InflatedConv3D
    
    Wrapper for Conv3D to match I3ResNet specifications.
    PyTorch: Conv3d(in, out, kernel, stride, padding, dilation)
    Lux: Conv(kernel, in=>out, stride, pad, dilation)
"""
function InflatedConv3D(in_chs, out_chs, kernel, stride, padding; has_bias=true)
    # PyTorch defaults cross-correlation. Lux defaults convolution.
    # We will handle weight flipping during LOADING.
    # Structurally, it is just a Conv layer.
    return Conv(kernel, in_chs => out_chs; stride=stride, pad=padding, use_bias=has_bias)
end

"""
    Bottleneck3D
    
    Standard ResNet Bottleneck but 3D.
    Structure:
    1. Conv1 1x1x1 (reduce)
    2. Conv2 kxkxk (spatial/temporal)
    3. Conv3 1x1x1 (expand)
    4. Shortcut (optional downsample)
"""
struct Bottleneck3D{C1, BN1, C2, BN2, C3, BN3, DS, A} <: Lux.AbstractLuxLayer
    conv1::C1
    bn1::BN1
    conv2::C2
    bn2::BN2
    conv3::C3
    bn3::BN3
    downsample::DS
    activation::A
end

function Bottleneck3D(in_chs, mid_chs, out_chs; stride=1, downsample=nothing)
    # ResNet Bottleneck:
    # Conv1: 1x1x1, in->mid
    # Conv2: 3x3x3 (usually), mid->mid. Stride applies here?
    # In ResNet, stride usually applies at Conv2 (3x3).
    # In I3ResNet `i3res.py`:
    # conv1: time_dim=1. (1x1x1).
    # conv2: time_dim=3. (3,3,3) or similar.
    #        time_stride=spatial_stride (which is `stride`).
    #        kernel=(3,3,3) if center=True inflation of 3x3.
    #        padding=(1,1,1).
    # conv3: time_dim=1. (1x1x1).
    
    # Conv1
    c1 = InflatedConv3D(in_chs, mid_chs, (1,1,1), 1, 0; has_bias=false)
    bn1 = BatchNorm(mid_chs)
    
    # Conv2
    # stride applies to (D, H, W)?
    # `inflate.py`: stride = (time_stride, stride[0], stride[0]).
    # If stride=2 passed to Bottleneck3D, it means (2,2,2) stride?
    # `Bottleneck3d`: spatial_stride = bottleneck2d.conv2.stride[0].
    # `conv2` time_stride = spatial_stride.
    # So stride is symmetric (s, s, s).
    # kernel (3,3,3), pad (1,1,1).
    c2 = InflatedConv3D(mid_chs, mid_chs, (3,3,3), stride, 1; has_bias=false)
    bn2 = BatchNorm(mid_chs)
    
    # Conv3
    c3 = InflatedConv3D(mid_chs, out_chs, (1,1,1), 1, 0; has_bias=false)
    bn3 = BatchNorm(out_chs)
    
    return Bottleneck3D(c1, bn1, c2, bn2, c3, bn3, downsample, relu)
end

Lux.initialparameters(rng::AbstractRNG, l::Bottleneck3D) = (
    conv1=Lux.initialparameters(rng, l.conv1),
    bn1=Lux.initialparameters(rng, l.bn1),
    conv2=Lux.initialparameters(rng, l.conv2),
    bn2=Lux.initialparameters(rng, l.bn2),
    conv3=Lux.initialparameters(rng, l.conv3),
    bn3=Lux.initialparameters(rng, l.bn3),
    downsample=(l.downsample === nothing ? NamedTuple() : Lux.initialparameters(rng, l.downsample))
)

Lux.initialstates(rng::AbstractRNG, l::Bottleneck3D) = (
    conv1=Lux.initialstates(rng, l.conv1),
    bn1=Lux.initialstates(rng, l.bn1),
    conv2=Lux.initialstates(rng, l.conv2),
    bn2=Lux.initialstates(rng, l.bn2),
    conv3=Lux.initialstates(rng, l.conv3),
    bn3=Lux.initialstates(rng, l.bn3),
    downsample=(l.downsample === nothing ? NamedTuple() : Lux.initialstates(rng, l.downsample)),
    activation=()
)

function (l::Bottleneck3D)(x, ps, st)
    y, st_c1 = l.conv1(x, ps.conv1, st.conv1)
    y, st_bn1 = l.bn1(y, ps.bn1, st.bn1)
    y = l.activation.(y)
    
    y, st_c2 = l.conv2(y, ps.conv2, st.conv2)
    y, st_bn2 = l.bn2(y, ps.bn2, st.bn2)
    y = l.activation.(y)
    
    y, st_c3 = l.conv3(y, ps.conv3, st.conv3)
    y, st_bn3 = l.bn3(y, ps.bn3, st.bn3)
    
    if l.downsample !== nothing
        residual, st_ds = l.downsample(x, ps.downsample, st.downsample)
        st = merge(st, (downsample=st_ds,))
    else
        residual = x
    end
    
    y = l.activation.(y .+ residual)
    
    return y, (conv1=st_c1, bn1=st_bn1, conv2=st_c2, bn2=st_bn2, conv3=st_c3, bn3=st_bn3, downsample=(l.downsample === nothing ? nothing : st.downsample))
end


"""
    I3ResNet
    
    ResNet152-3D Backbone.
"""
struct I3ResNet{C1, BN1, MP, L1, L2, L3, L4, AP, FC} <: Lux.AbstractLuxLayer
    conv1::C1
    bn1::BN1
    maxpool::MP
    layer1::L1
    layer2::L2
    layer3::L3
    layer4::L4
    avgpool::AP
    fc::FC # Optional classifier/embedding support
    
    # Config
    num_triplicate::Int # Number of times to repeat input channel (1->3)
end

function make_layer(in_chs, mid_chs, out_chs, num_blocks; stride=1)
    layers = []
    
    # Downsample layer if stride != 1 or in_chs != out_chs
    downsample = nothing
    if stride != 1 || in_chs != out_chs
        # inflate_downsample: Conv 1x1x1 with stride
        ds_conv = InflatedConv3D(in_chs, out_chs, (1,1,1), stride, 0; has_bias=false)
        ds_bn = BatchNorm(out_chs)
        downsample = Chain(ds_conv, ds_bn)
        # Note: Lux Chain works, but we need to verify parameter naming mapping
    end
    
    push!(layers, Bottleneck3D(in_chs, mid_chs, out_chs; stride=stride, downsample=downsample))
    
    for _ in 2:num_blocks
        push!(layers, Bottleneck3D(out_chs, mid_chs, out_chs; stride=1))
    end
    
    return Chain(layers...)
end

function Merlin(return_embeddings=true)
    # ResNet152 Config
    # Blocks: [3, 8, 36, 3]
    # Channels: 64 -> 256 -> 512 -> 1024 -> 2048
    
    # Input Conv: 7x7 stride 2 (2D) -> Inflated (3, 7, 7) stride (1, 2, 2) pad (1, 3, 3)
    # PyTorch (D, H, W): Kernel (3, 7, 7), Stride (1, 2, 2), Pad (1, 3, 3)
    # Lux (W, H, D): Kernel (7, 7, 3), Stride (2, 2, 1), Pad (3, 3, 1)
    
    conv1 = InflatedConv3D(3, 64, (7,7,3), (2,2,1), (3,3,1); has_bias=false)
    bn1 = BatchNorm(64)
    # MaxPool: 3x3 s2 -> Inflated (3,3,3) s(2,2,2) p(1,1,1)
    # PyTorch (D, H, W): (3,3,3), (2,2,2), (1,1,1)
    # Lux (W, H, D): (3,3,3), (2,2,2), (1,1,1) (Symmetric)
    maxpool = MaxPool((3,3,3); stride=(2,2,2), pad=(1,1,1))
    
    # Layers
    # Layer 1: 3 blocks. Stride 1.
    layer1 = make_layer(64, 64, 256, 3; stride=1)
    
    # Layer 2: 8 blocks. Stride 2.
    # PyTorch Stride 2 (Spatial). usually (1, 2, 2).
    # Lux Stride (2, 2, 1).
    layer2 = make_layer(256, 128, 512, 8; stride=(2,2,1))
    
    # Layer 3: 36 blocks. Stride 2.
    layer3 = make_layer(512, 256, 1024, 36; stride=(2,2,1))
    
    # Layer 4: 3 blocks. Stride 2.
    layer4 = make_layer(1024, 512, 2048, 3; stride=(2,2,1))
    
    # AvgPool
    # conv_class=True in build.py.
    # avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1).
    # resnet2d.avgpool created by torchvision is usually AdaptiveAvgPool2d((1,1)).
    # inflate.inflate_pool -> AdaptiveAvgPool3d((1,1,1)).
    avgpool = GlobalMeanPool() # Or AdaptiveMeanPool((1,1,1))
    
    # Classifier / Heads
    # We will just return the features after avgpool.
    fc = NoOpLayer()
    
    return I3ResNet(conv1, bn1, maxpool, layer1, layer2, layer3, layer4, avgpool, fc, 3)
end

# Lux Boilerplate for I3ResNet
Lux.initialparameters(rng::AbstractRNG, l::I3ResNet) = (
    conv1=Lux.initialparameters(rng, l.conv1),
    bn1=Lux.initialparameters(rng, l.bn1),
    maxpool=Lux.initialparameters(rng, l.maxpool),
    layer1=Lux.initialparameters(rng, l.layer1),
    layer2=Lux.initialparameters(rng, l.layer2),
    layer3=Lux.initialparameters(rng, l.layer3),
    layer4=Lux.initialparameters(rng, l.layer4),
    avgpool=Lux.initialparameters(rng, l.avgpool),
    fc=Lux.initialparameters(rng, l.fc)
)

Lux.initialstates(rng::AbstractRNG, l::I3ResNet) = (
    conv1=Lux.initialstates(rng, l.conv1),
    bn1=Lux.initialstates(rng, l.bn1),
    maxpool=Lux.initialstates(rng, l.maxpool),
    layer1=Lux.initialstates(rng, l.layer1),
    layer2=Lux.initialstates(rng, l.layer2),
    layer3=Lux.initialstates(rng, l.layer3),
    layer4=Lux.initialstates(rng, l.layer4),
    avgpool=Lux.initialstates(rng, l.avgpool),
    fc=Lux.initialstates(rng, l.fc)
)

function (l::I3ResNet)(x, ps, st)
    # x: (W, H, D, C, N) ideally (1, H, W, D, N)? NO.
    # Spect: (W, H, D, 1, N).
    # Merlin triplicates channel.
    
    # Lux Cat: dims=4 (Channel).
    x_tri = cat(x, x, x; dims=4)
    
    y, st_c1 = l.conv1(x_tri, ps.conv1, st.conv1)
    y, st_bn1 = l.bn1(y, ps.bn1, st.bn1)
    y = relu.(y)
    y, st_mp = l.maxpool(y, ps.maxpool, st.maxpool)
    
    y, st_l1 = l.layer1(y, ps.layer1, st.layer1)
    y, st_l2 = l.layer2(y, ps.layer2, st.layer2)
    y, st_l3 = l.layer3(y, ps.layer3, st.layer3)
    y, st_l4 = l.layer4(y, ps.layer4, st.layer4)
    
    y, st_ap = l.avgpool(y, ps.avgpool, st.avgpool)
    
    return y, (conv1=st_c1, bn1=st_bn1, maxpool=st_mp, layer1=st_l1, layer2=st_l2, layer3=st_l3, layer4=st_l4, avgpool=st_ap, fc=(;))
end
