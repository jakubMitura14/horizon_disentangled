using Lux
using MPI
using Random
using Optimisers
using Zygote
using Statistics
using Printf
using ComponentArrays

# Lux Distributed Training Simulation
# ===================================
# Simulates data parallelism by running model forward/backward passes
# and synchronizing gradients using MPI.Allreduce.

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if rank == 0
        println("--- Lux Distributed Training (MPI) ---")
        println("World Size: $size")
    end

    # 1. Initialize Model (Mock CausalVAE structure)
    model = Chain(
        Conv((3,3,3), 1 => 16, pad=1, relu),
        Conv((3,3,3), 16 => 32, pad=1, relu),
        FlattenLayer(),
        Dense(32*16*48*48 => 10)
    )

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    # Wrap parameters in ComponentArray for easy flattening/reduction
    ps_ca = ComponentArray(ps)

    # Broadcast initial parameters from rank 0 to all workers
    # MPI.Bcast!(ps_ca, 0, comm)
    # ComponentVector might not be directly compatible with MPI.Buffer without extra steps.
    # We can broadcast the underlying array.
    MPI.Bcast!(getdata(ps_ca), 0, comm)

    opt = Optimisers.Adam(1e-3)
    st_opt = Optimisers.setup(opt, ps_ca)

    # Mock Data
    x = rand(Float32, 48, 48, 16, 1, 2)
    y = rand(Float32, 10, 2)

    function loss_fn(p, x, y, st)
        y_pred, st_new = model(x, p, st)
        return mean(abs2, y_pred .- y), st_new
    end

    # Training Loop
    epochs = 2

    MPI.Barrier(comm)

    for epoch in 1:epochs
        t_start = time()

        # 1. Local Gradient Calculation
        (loss, st), back = Zygote.pullback(p -> loss_fn(p, x, y, st), ps_ca)
        grads = back((1.0f0, nothing))[1]

        # 2. Distributed Synchronization (Allreduce Gradients)
        # Use underlying array for MPI operations to avoid ComponentArray type issues
        grad_data = getdata(grads)
        MPI.Allreduce!(grad_data, MPI.SUM, comm)

        # Average gradients
        grad_data ./= size

        # 3. Update Parameters
        st_opt, ps_ca = Optimisers.update(st_opt, ps_ca, grads)

        t_end = time()
        epoch_time = t_end - t_start

        check_sum = sum(ps_ca[1:5])

        if rank == 0
            @printf("Epoch %d: Loss %.4f | Time %.4fs | Check Sum: %.4f\n", epoch, loss, epoch_time, check_sum)
        end
    end

    MPI.Finalize()
end

main()
