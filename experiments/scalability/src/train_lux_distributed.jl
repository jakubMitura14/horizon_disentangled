using Lux
using MPI
using Random
using Optimisers
using Zygote
using Statistics
using Printf
using ComponentArrays
using CUDA

# Lux Distributed Training Script (Hybrid CPU/GPU)
# ================================================
# This script implements a distributed training loop for the CausalPCa model.
# It automatically detects if CUDA GPUs are available and moves data/models accordingly.
# It uses MPI for gradient synchronization (Data Parallelism).

# --- Heavier Architecture: 3D ResNet Block ---
struct ResNetBlock3D <: Lux.AbstractLuxLayer
    conv1::Conv
    conv2::Conv
    norm1::BatchNorm
    norm2::BatchNorm
    shortcut::Union{Chain, NoOpLayer}
end

function ResNetBlock3D(in_channels::Int, out_channels::Int; stride::Int=1)
    conv1 = Conv((3,3,3), in_channels => out_channels, stride=stride, pad=1, use_bias=false)
    norm1 = BatchNorm(out_channels)
    conv2 = Conv((3,3,3), out_channels => out_channels, stride=1, pad=1, use_bias=false)
    norm2 = BatchNorm(out_channels)

    shortcut = if stride > 1 || in_channels != out_channels
        Chain(
            Conv((1,1,1), in_channels => out_channels, stride=stride, use_bias=false),
            BatchNorm(out_channels)
        )
    else
        NoOpLayer()
    end

    return ResNetBlock3D(conv1, conv2, norm1, norm2, shortcut)
end

function Lux.initialparameters(rng::AbstractRNG, l::ResNetBlock3D)
    return (
        conv1 = Lux.initialparameters(rng, l.conv1),
        conv2 = Lux.initialparameters(rng, l.conv2),
        norm1 = Lux.initialparameters(rng, l.norm1),
        norm2 = Lux.initialparameters(rng, l.norm2),
        shortcut = Lux.initialparameters(rng, l.shortcut)
    )
end

function Lux.initialstates(rng::AbstractRNG, l::ResNetBlock3D)
    return (
        conv1 = Lux.initialstates(rng, l.conv1),
        conv2 = Lux.initialstates(rng, l.conv2),
        norm1 = Lux.initialstates(rng, l.norm1),
        norm2 = Lux.initialstates(rng, l.norm2),
        shortcut = Lux.initialstates(rng, l.shortcut)
    )
end

function (l::ResNetBlock3D)(x, ps, st)
    y, st_c1 = l.conv1(x, ps.conv1, st.conv1)
    y, st_n1 = l.norm1(y, ps.norm1, st.norm1)
    y = relu.(y)

    y, st_c2 = l.conv2(y, ps.conv2, st.conv2)
    y, st_n2 = l.norm2(y, ps.norm2, st.norm2)

    sc, st_sc = l.shortcut(x, ps.shortcut, st.shortcut)

    return relu.(y .+ sc), (conv1=st_c1, conv2=st_c2, norm1=st_n1, norm2=st_n2, shortcut=st_sc)
end

function HeavyResNet3D()
    # Simple ResNet-18 style backbone 3D
    return Chain(
        Conv((7,7,7), 1 => 64, stride=2, pad=3, use_bias=false),
        BatchNorm(64),
        x -> relu.(x),
        MaxPool((3,3,3), stride=2, pad=1),

        # Layer 1
        ResNetBlock3D(64, 64),
        ResNetBlock3D(64, 64),

        # Layer 2
        ResNetBlock3D(64, 128, stride=2),
        ResNetBlock3D(128, 128),

        # Layer 3
        ResNetBlock3D(128, 256, stride=2),
        ResNetBlock3D(256, 256),

        # Layer 4
        ResNetBlock3D(256, 512, stride=2),
        ResNetBlock3D(512, 512),

        GlobalMeanPool(),
        FlattenLayer(),
        Dense(512 => 10)
    )
end

function main()
    # 1. Initialize MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # 2. Device Selection (CPU or GPU)
    use_cuda = CUDA.functional()
    device = use_cuda ? gpu_device() : cpu_device()

    if rank == 0
        println("--- Lux Distributed Training (MPI) ---")
        println("World Size: $size")
        println("Device: $(use_cuda ? "GPU (CUDA)" : "CPU")")
        println("Model: Heavy ResNet-18 3D")
    end

    # 3. Model Definition
    model = HeavyResNet3D()

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    # Move parameters to device
    ps = ps |> device
    st = st |> device

    ps_ca = ComponentArray(ps)

    # 4. Parameter Synchronization
    ps_cpu = ps_ca |> cpu_device()
    MPI.Bcast!(getdata(ps_cpu), 0, comm)

    if use_cuda
        ps_ca = ComponentArray(ps_cpu, getaxes(ps_ca)) |> gpu_device()
    else
        ps_ca = ps_cpu
    end

    opt = Optimisers.Adam(1e-3)
    st_opt = Optimisers.setup(opt, ps_ca)

    # 5. Distributed Data Loading Strategy
    # Increased volume size to 96x96x64 to saturate GPU
    local_batch_size = 4
    x = rand(Float32, 96, 96, 64, 1, local_batch_size) |> device
    y = rand(Float32, 10, local_batch_size) |> device

    function loss_fn(p, x, y, st)
        y_pred, st_new = model(x, p, st)
        return mean(abs2, y_pred .- y), st_new
    end

    # 6. Training Loop
    epochs = 5
    MPI.Barrier(comm)

    for epoch in 1:epochs
        t_start = time()

        # A. Local Gradient Calculation
        (loss, st), back = Zygote.pullback(p -> loss_fn(p, x, y, st), ps_ca)
        grads = back((Float32(1.0) |> device, nothing))[1]

        # B. Distributed Synchronization
        if use_cuda
             grad_data = Array(grads)
             MPI.Allreduce!(grad_data, MPI.SUM, comm)
             grad_data ./= size
             grads = ComponentArray(grad_data, getaxes(ps_ca)) |> gpu_device()
        else
             grad_data = getdata(grads)
             MPI.Allreduce!(grad_data, MPI.SUM, comm)
             grad_data ./= size
        end

        # C. Optimizer Step
        st_opt, ps_ca = Optimisers.update(st_opt, ps_ca, grads)

        t_end = time()
        epoch_time = t_end - t_start

        # Simplified Checksum (first param)
        check_sum = sum(ps_ca[1:1])

        if rank == 0
            @printf("Epoch %d: Loss %.4f | Time %.4fs | Check Sum: %.4f | Device: %s\n",
                    epoch, loss, epoch_time, check_sum, use_cuda ? "GPU" : "CPU")
        end
    end

    MPI.Finalize()
end

main()
