using ArgParse
using Lux
using MPI
using Random
using Optimisers
using Zygote
using Statistics
using Printf
using ComponentArrays
using CUDA
using LuxCUDA

# Lux Distributed Training Script (Hybrid CPU/GPU)
# ================================================
# This script implements a distributed training loop for the CausalPCa model.
# It automatically detects if CUDA GPUs are available and moves data/models accordingly.
# It uses MPI for gradient synchronization (Data Parallelism).

# --- Heavier Architecture: 3D ResNet-50 Block ---
struct ResNetBlock3D <: Lux.AbstractLuxLayer
    conv1::Conv
    conv2::Conv
    conv3::Conv
    norm1::BatchNorm
    norm2::BatchNorm
    norm3::BatchNorm
    shortcut::Union{Chain, NoOpLayer}
end

# Bottleneck Block for ResNet-152+
function ResNetBottleneck(in_channels::Int, out_channels::Int; stride::Int=1, expansion::Int=4)
    mid_channels = out_channels
    final_channels = out_channels * expansion

    conv1 = Conv((1,1,1), in_channels => mid_channels, stride=1, use_bias=false)
    norm1 = BatchNorm(mid_channels)

    conv2 = Conv((3,3,3), mid_channels => mid_channels, stride=stride, pad=1, use_bias=false)
    norm2 = BatchNorm(mid_channels)

    conv3 = Conv((1,1,1), mid_channels => final_channels, stride=1, use_bias=false)
    norm3 = BatchNorm(final_channels)

    shortcut = if stride > 1 || in_channels != final_channels
        Chain(
            Conv((1,1,1), in_channels => final_channels, stride=stride, use_bias=false),
            BatchNorm(final_channels)
        )
    else
        NoOpLayer()
    end

    return ResNetBlock3D(conv1, conv2, conv3, norm1, norm2, norm3, shortcut)
end

function Lux.initialparameters(rng::AbstractRNG, l::ResNetBlock3D)
    return (
        conv1 = Lux.initialparameters(rng, l.conv1),
        conv2 = Lux.initialparameters(rng, l.conv2),
        conv3 = Lux.initialparameters(rng, l.conv3),
        norm1 = Lux.initialparameters(rng, l.norm1),
        norm2 = Lux.initialparameters(rng, l.norm2),
        norm3 = Lux.initialparameters(rng, l.norm3),
        shortcut = Lux.initialparameters(rng, l.shortcut)
    )
end

function Lux.initialstates(rng::AbstractRNG, l::ResNetBlock3D)
    return (
        conv1 = Lux.initialstates(rng, l.conv1),
        conv2 = Lux.initialstates(rng, l.conv2),
        conv3 = Lux.initialstates(rng, l.conv3),
        norm1 = Lux.initialstates(rng, l.norm1),
        norm2 = Lux.initialstates(rng, l.norm2),
        norm3 = Lux.initialstates(rng, l.norm3),
        shortcut = Lux.initialstates(rng, l.shortcut)
    )
end

function (l::ResNetBlock3D)(x, ps, st)
    y, st_c1 = l.conv1(x, ps.conv1, st.conv1)
    y, st_n1 = l.norm1(y, ps.norm1, st.norm1)
    y = relu.(y)

    y, st_c2 = l.conv2(y, ps.conv2, st.conv2)
    y, st_n2 = l.norm2(y, ps.norm2, st.norm2)
    y = relu.(y)

    y, st_c3 = l.conv3(y, ps.conv3, st.conv3)
    y, st_n3 = l.norm3(y, ps.norm3, st.norm3)

    sc, st_sc = l.shortcut(x, ps.shortcut, st.shortcut)

    return relu.(y .+ sc), (conv1=st_c1, conv2=st_c2, conv3=st_c3, norm1=st_n1, norm2=st_n2, norm3=st_n3, shortcut=st_sc)
end

function SuperHeavyResNet152_3D()
    # ResNet-152 3D Style (Wide)
    # Layers: [3, 8, 36, 3] blocks for ResNet-152

    layers = []

    # Stem
    push!(layers, Conv((7,7,7), 1 => 64, stride=2, pad=3, use_bias=false))
    push!(layers, BatchNorm(64))
    push!(layers, x -> relu.(x))
    push!(layers, MaxPool((3,3,3), stride=2, pad=1))

    in_ch = 64

    # Layer 1 (3 blocks)
    for i in 1:3
        push!(layers, ResNetBottleneck(in_ch, 64))
        in_ch = 64 * 4
    end

    # Layer 2 (8 blocks)
    push!(layers, ResNetBottleneck(in_ch, 128, stride=2))
    in_ch = 128 * 4
    for i in 2:8
        push!(layers, ResNetBottleneck(in_ch, 128))
    end

    # Layer 3 (36 blocks) - The heavy part
    push!(layers, ResNetBottleneck(in_ch, 256, stride=2))
    in_ch = 256 * 4
    for i in 2:36
        push!(layers, ResNetBottleneck(in_ch, 256))
    end

    # Layer 4 (3 blocks)
    push!(layers, ResNetBottleneck(in_ch, 512, stride=2))
    in_ch = 512 * 4
    for i in 2:3
        push!(layers, ResNetBottleneck(in_ch, 512))
    end

    push!(layers, GlobalMeanPool())
    push!(layers, FlattenLayer())
    push!(layers, Dense(in_ch => 10))

    return Chain(layers...)
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--epochs"
            help = "Number of epochs"
            arg_type = Int
            default = 1000
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    epochs = args["epochs"]

    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    use_cuda = CUDA.functional()
    device = use_cuda ? gpu_device() : cpu_device()

    if rank == 0
        println("--- Lux Distributed Training (MPI) ---")
        println("World Size: $size")
        println("Device: $(use_cuda ? "GPU (CUDA)" : "CPU")")
        println("Model: Super Heavy ResNet-152 3D (Wide)")
        println("Epochs: $epochs")
    end

    model = SuperHeavyResNet152_3D()

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    ps = ps |> device
    st = st |> device

    # Use Optimisers.destructure for efficient flattening (idiomatic for Lux + GPU)
    ps_flat, relabel = destructure(ps)

    # Synchronize initial parameters
    if use_cuda
        # If using CUDA, we assume CUDA-aware MPI or fallback to CPU copy
        # We'll use a CPU buffer for Bcast to be safe if MPI is not CUDA-aware
        ps_cpu = ps_flat |> cpu_device()
        MPI.Bcast!(ps_cpu, 0, comm)
        ps_flat = ps_cpu |> device
    else
        MPI.Bcast!(ps_flat, 0, comm)
    end
    ps = relabel(ps_flat)

    opt = Optimisers.Adam(1e-3)
    st_opt = Optimisers.setup(opt, ps)

    local_batch_size = 4
    x = rand(Float32, 128, 128, 64, 1, local_batch_size) |> device
    y = rand(Float32, 10, local_batch_size) |> device

    function loss_fn(p, x, y, st)
        y_pred, st_new = model(x, p, st)
        return mean(abs2, y_pred .- y), st_new
    end

    MPI.Barrier(comm)

    for epoch in 1:epochs
        t_start = time()

        (loss, st), back = Zygote.pullback(p -> loss_fn(p, x, y, st), ps)
        grads = back((Float32(1.0) |> device, nothing))[1]

        # Flatten gradients for MPI synchronization
        gs_flat, _ = destructure(grads)

        if use_cuda
             grad_data = getdata(grads)
             MPI.Allreduce!(grad_data, MPI.SUM, comm)
             grad_data ./= size
        else
             grad_data = getdata(grads)
             MPI.Allreduce!(grad_data, MPI.SUM, comm)
             grad_data ./= size
        end

        st_opt, ps_ca = Optimisers.update(st_opt, ps_ca, grads)

        t_end = time()
        epoch_time = t_end - t_start

        # Checksum every 10 epochs or first few
        if rank == 0 && (epoch <= 5 || epoch % 10 == 0)
            @printf("Epoch %d: Loss %.4f | Time %.4fs | Device: %s\n",
                    epoch, loss, epoch_time, use_cuda ? "GPU" : "CPU")
        end
    end

    MPI.Finalize()
end

main()
