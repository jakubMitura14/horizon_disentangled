using Lux
using DifferentialEquations
using SciMLSensitivity
using Random
using ComponentArrays

# Dynamics ODE Function: dz/dt = f(z, t)
function NJDEDynamics(dim)
    return Chain(
        Dense(dim => 32, tanh),
        Dense(32 => dim)
    )
end

# Jump Function: z+ = z- + g(z)
function JumpNet(dim, coord_dim=3)
    return Chain(
        Dense(dim + coord_dim => 32, relu),
        Dense(32 => dim)
    )
end

# Wrapper for solving
function solve_njde(model_dyn, model_jump, ps, st, z0, t_span, jump_times, jump_coords)
    # p_dyn = ps.dynamics
    # p_jump = ps.jump

    function ode_func(u, p, t)
        # Lux model call
        out, _ = model_dyn(u, p.dynamics, st.dynamics)
        return out
    end

    # Callback for jumps
    # Simplified: Assuming fixed jump times for pilot
    # In Julia, ContinuousCallback or DiscreteCallback

    cb = nothing
    if !isempty(jump_times)
        # Create a callback for the first jump
        jt = jump_times[1]
        jc = jump_coords[1] # (coord_dim,)

        function affect!(integrator)
            u = integrator.u
            # Input to jump net: [u; coords]
            # Need to handle batch dimension if present or assume single trajectory
            # Lux expects (dim, batch)

            # Prepare input
            # If u is (D, B), repeat jc to (C, B)
            B = size(u, 2)
            coords_batch = repeat(reshape(jc, :, 1), 1, B)
            input = vcat(u, coords_batch)

            jump_val, _ = model_jump(input, integrator.p.jump, st.jump)
            integrator.u += jump_val
        end

        cb = PresetTimeCallback(jt, affect!)
    end

    prob = ODEProblem(ode_func, z0, t_span, ps)
    sol = solve(prob, Tsit5(), callback=cb, saveat=0.1, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return sol
end
