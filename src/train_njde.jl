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
using CUDA
using LuxCUDA

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

    # Device
    dev = gpu_device()
    println("Using device: $dev")

    # Move data
    z_true = z_true |> dev
    z_target = z_target |> dev
    # jump_coords might need to be on CPU for callback or GPU? 
    # Current solve_njde ignores jumps, so it's fine.
    
    model_dyn = NJDEDynamics(16)
    model_jump = JumpNet(16)

    rng = Random.default_rng()
    ps_dyn, st_dyn = Lux.setup(rng, model_dyn)
    ps_jump, st_jump = Lux.setup(rng, model_jump)

    ps = ComponentArray(dynamics=ps_dyn, jump=ps_jump)
    st = (dynamics=st_dyn, jump=st_jump)
    
    ps = ps |> dev
    st = st |> dev

    opt = Optimisers.Adam(1e-2)
    st_opt = Optimisers.setup(opt, ps)

    function loss_fn(p)
        # Note: solve_njde needs to handle GPU arrays if passed
        sol = solve_njde(model_dyn, model_jump, p, st, z_true, t_span, jump_times, jump_coords)
        if sol.retcode != :Success
            return Inf32
        end
        z_end = sol.u[end]
        l = mean(abs2, z_end .- z_target)
        return l
    end

    println("--- Training Neural Jump ODE (SciML) ---")
    
    max_epochs = 50
    patience = 3
    best_loss = Inf
    patience_counter = 0

    with_logger(logger) do
        for i in 1:max_epochs
            grads = Zygote.gradient(loss_fn, ps)
            l_val = loss_fn(ps)
            
            # Update
            st_opt, ps = Optimisers.update(st_opt, ps, grads[1])

            @info "train" loss=l_val epoch=i
            println("Epoch $i Loss: $l_val")
            
             # Early Stopping
            if l_val < best_loss
                best_loss = l_val
                patience_counter = 0
            else
                patience_counter += 1
            end
            
            if patience_counter >= patience
                println("Early stopping at epoch $i (Best: $best_loss)")
                break
            end
        end
    end

    println("NJDE Trained.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
