using Lux
using Random
using Optimisers
using Zygote
using Statistics
using DifferentialEquations
using SciMLSensitivity
using ComponentArrays

include("models/njde.jl")

function train()
    # Mock Data: 1 patient, 3 timepoints
    z_true = randn(Float32, 16, 1) # (D, B) - Initial
    # Target at t=1.0
    z_target = randn(Float32, 16, 1)

    t_span = (0.0f0, 1.0f0)
    jump_times = [0.5f0]
    jump_coords = [rand(Float32, 3)]

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
        # Loss: distance to z_target at end
        z_end = sol.u[end]
        l = mean(abs2, z_end .- z_target)
        return l
    end

    println("--- Training Neural Jump ODE (SciML) ---")
    # One step
    grads = Zygote.gradient(loss_fn, ps)

    # Check bounds or NaNs
    l_val = loss_fn(ps)
    println("Epoch 1 Loss: $l_val")

    # Update
    st_opt, ps = Optimisers.update(st_opt, ps, grads[1])

    l_val_2 = loss_fn(ps)
    println("Epoch 2 Loss: $l_val_2")

    println("NJDE Trained.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
