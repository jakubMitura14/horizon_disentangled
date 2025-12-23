using Lux
using Random
using Optimisers
using Zygote
using Statistics
using DifferentialEquations
using SciMLSensitivity
using ComponentArrays
using TensorBoardLogger
using Logging

include("models/njde.jl")

function train()
    # Mock Data: 1 patient, 3 timepoints
    # In real pipeline, load from DATA_DIR
    z_true = randn(Float32, 16, 1) # (D, B) - Initial
    z_target = randn(Float32, 16, 1)

    t_span = (0.0f0, 1.0f0)
    jump_times = [0.5f0]
    jump_coords = [rand(Float32, 3)]

    # Logger
    logger = TBLogger("logs/njde", min_level=Logging.Info)

    model_dyn = NJDEDynamics(16)
    model_jump = JumpNet(16)

    rng = Random.default_rng()
    ps_dyn, st_dyn = Lux.setup(rng, model_dyn)
    ps_jump, st_jump = Lux.setup(rng, model_jump)

    ps = ComponentArray(dynamics=ps_dyn, jump=ps_jump)
    st = (dynamics=st_dyn, jump=st_jump)

    opt = Optimisers.Adam(1e-2)
    st_opt = Optimisers.setup(opt, ps)

    function loss_fn(p)
        sol = solve_njde(model_dyn, model_jump, p, st, z_true, t_span, jump_times, jump_coords)
        if sol.retcode != :Success
            return Inf
        end
        z_end = sol.u[end]
        l = mean(abs2, z_end .- z_target)
        return l
    end

    println("--- Training Neural Jump ODE (SciML) ---")
    with_logger(logger) do
        # Epoch 1
        grads = Zygote.gradient(loss_fn, ps)
        l_val = loss_fn(ps)
        @info "train" loss=l_val epoch=1
        println("Epoch 1 Loss: $l_val")

        # Update
        st_opt, ps = Optimisers.update(st_opt, ps, grads[1])

        # Epoch 2
        l_val_2 = loss_fn(ps)
        @info "train" loss=l_val_2 epoch=2
        println("Epoch 2 Loss: $l_val_2")
    end

    println("NJDE Trained.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
