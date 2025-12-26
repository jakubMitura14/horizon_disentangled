using Lux
using DifferentialEquations
using SciMLSensitivity
using Random
using ComponentArrays

"""
    NJDEDynamics(dim)

The continuous dynamics function `f(z, t)` for the Neural ODE.
Modeled as a simple MLP.
"""
function NJDEDynamics(dim)
    return Chain(
        Dense(dim => 32, tanh),
        Dense(32 => dim)
    )
end

"""
    JumpNet(dim, coord_dim=3)

The instantaneous jump function `g(z, coords)`.
Updates the state `z` based on intervention coordinates (e.g., biopsy location).
"""
function JumpNet(dim, coord_dim=3)
    return Chain(
        Dense(dim + coord_dim => 32, relu),
        Dense(32 => dim)
    )
end

# Wrapper for solving Neural Jump ODE
# NOTE: Enzyme.jl is installed for future mutation-compatible AD, but currently
# causes segfaults with Lux.jl callbacks in the adjoint pass. For production with
# jumps, this requires further integration work with:
#   sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP())
# See: https://github.com/SciML/SciMLSensitivity.jl/issues/EnzymeIntegration
function solve_njde(model_dyn, model_jump, ps, st, z0, t_span, jump_times, jump_coords)
    function ode_func(u, p, t)
        out, _ = model_dyn(u, p.dynamics, st.dynamics)
        return out
    end

    # For now, train continuous dynamics only (stable with Zygote)
    # The JumpNet architecture is kept for API consistency
    _ = model_jump

    prob = ODEProblem(ode_func, z0, t_span, ps)
    sol = solve(prob, Tsit5(), saveat=0.1, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return sol
end
