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

function main()
    # 1. Initialize MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # 2. Device Selection (CPU or GPU)
    # On a Slurm cluster, CUDA_VISIBLE_DEVICES is usually set per rank.
    use_cuda = CUDA.functional()
    device = use_cuda ? gpu_device() : cpu_device()

    if rank == 0
        println("--- Lux Distributed Training (MPI) ---")
        println("World Size: $size")
        println("Device: $(use_cuda ? "GPU (CUDA)" : "CPU")")
    end

    # 3. Model Definition (Mock CausalVAE structure)
    # We use a simplified 3D CNN structure to represent the encoder/decoder complexity.
    model = Chain(
        Conv((3,3,3), 1 => 16, pad=1, relu),
        Conv((3,3,3), 16 => 32, pad=1, relu),
        FlattenLayer(),
        Dense(32*16*48*48 => 10)
    )

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    # Move parameters to device (GPU if available)
    ps = ps |> device
    st = st |> device

    # Wrap parameters in ComponentArray for easy flattening/reduction
    # ComponentArray works seamlessly with Lux on both CPU and GPU.
    ps_ca = ComponentArray(ps)

    # 4. Parameter Synchronization
    # Broadcast initial parameters from rank 0 to all workers to ensure deterministic start.
    # Note: We move data to CPU for MPI broadcast if MPI implementation doesn't support CUDA-aware MPI directly,
    # or rely on CUDA-aware MPI if configured. For safety in this script, we assume CPU-side sync for initialization.
    ps_cpu = ps_ca |> cpu_device()
    # For broadcasting ComponentArray, broadcast the underlying data
    MPI.Bcast!(getdata(ps_cpu), 0, comm)

    if use_cuda
        # Re-transfer to GPU if we were on GPU
        ps_ca = ComponentArray(ps_cpu, getaxes(ps_ca)) |> gpu_device()
    else
        ps_ca = ps_cpu
    end

    opt = Optimisers.Adam(1e-3)
    st_opt = Optimisers.setup(opt, ps_ca)

    # 5. Distributed Data Loading Strategy
    # In a real scenario, we would partition the dataset indices based on (rank, size).
    # Dataset Index: i_start = rank * (N / size) + 1
    # Here, we simulate this by generating a unique random batch per rank.

    # Input: 3D Volume (48x48x16), Batch Size 2
    local_batch_size = 2
    x = rand(Float32, 48, 48, 16, 1, local_batch_size) |> device
    y = rand(Float32, 10, local_batch_size) |> device

    function loss_fn(p, x, y, st)
        y_pred, st_new = model(x, p, st)
        # MSE Loss
        return mean(abs2, y_pred .- y), st_new
    end

    # 6. Training Loop
    epochs = 5

    # Sync start
    MPI.Barrier(comm)

    for epoch in 1:epochs
        t_start = time()

        # A. Local Gradient Calculation (Autodiff)
        (loss, st), back = Zygote.pullback(p -> loss_fn(p, x, y, st), ps_ca)

        # Calculate Gradients
        # For the first pass, we use a dummy gradient of 1.0
        grads = back((Float32(1.0) |> device, nothing))[1]

        # B. Distributed Synchronization (Allreduce Gradients)
        # We assume CUDA-aware MPI if on GPU. If not, one would need to copy to CPU.
        # Ideally: grad_cpu = grads |> cpu_device() -> Allreduce -> grad_gpu = grad_cpu |> gpu_device()
        # For simplicity here, we assume the environment supports the transfer or we are on CPU.

        if use_cuda
             # Fallback for non-CUDA-aware MPI: Copy to CPU
             grad_data = Array(grads)
             MPI.Allreduce!(grad_data, MPI.SUM, comm)
             grad_data ./= size
             # Copy back to GPU and reconstruct structure (if needed) or just update ps_ca
             # Optimisers.update needs the structure matching ps_ca.
             # We can update the ps_ca directly if grads matches.
             grads = ComponentArray(grad_data, getaxes(ps_ca)) |> gpu_device()
        else
             # Use underlying array for MPI operations to avoid ComponentArray type issues with MPI.Buffer
             grad_data = getdata(grads)
             MPI.Allreduce!(grad_data, MPI.SUM, comm)
             grad_data ./= size
        end

        # C. Optimizer Step
        st_opt, ps_ca = Optimisers.update(st_opt, ps_ca, grads)

        t_end = time()
        epoch_time = t_end - t_start

        # Checksum for validation (sum of first 5 params)
        check_sum = sum(ps_ca[1:5])

        if rank == 0
            @printf("Epoch %d: Loss %.4f | Time %.4fs | Check Sum: %.4f | Device: %s\n",
                    epoch, loss, epoch_time, check_sum, use_cuda ? "GPU" : "CPU")
        end
    end

    MPI.Finalize()
end

main()
