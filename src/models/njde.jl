using Lux
using DifferentialEquations
using SciMLSensitivity
using Random
using ComponentArrays
using Enzyme

"""
    NJDEDynamics(dim)

The continuous dynamics function `f(z, t)` for the Neural ODE.
Modeled as a simple MLP. Mutation-free (required by EnzymeVJP).
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
Now wired into the ODE via DiscreteCallback so jumps are differentiable.
"""
function JumpNet(dim, coord_dim=3)
    return Chain(
        Dense(dim + coord_dim => 32, relu),
        Dense(32 => dim)
    )
end

"""
    solve_njde(model_dyn, model_jump, ps, st, z0, t_span, jump_times, jump_coords)

Solve a Neural Jump ODE where:
  - `model_dyn`  : continuous latent dynamics  f(z, t)   [NJDEDynamics]
  - `model_jump` : instantaneous jump kernel    g(z, loc) [JumpNet]
  - `jump_times` : sorted vector of event times
  - `jump_coords`: (coord_dim,) or (n_jumps, coord_dim) intervention locations

Adjoint strategy:
  - EnzymeVJP with InterpolatingAdjoint — mutation-compatible, correct through
    discrete callbacks, O(1) memory in time steps.
  - Falls back to ZygoteVJP if EnzymeVJP is not available (e.g. pre-0.12 Enzyme).
"""
function solve_njde(model_dyn, model_jump, ps, st, z0, t_span, jump_times, jump_coords)
    # --- continuous ODE right-hand side (must be mutation-free for EnzymeVJP) ---
    function ode_func(u, p, t)
        out, _ = model_dyn(u, p.dynamics, st.dynamics)
        return out
    end

    # --- discrete jump callbacks ---
    # Each jump applies the JumpNet correction to the latent state.
    # Using a CallbackSet of PresetTimeCallbacks gives exact event timing
    # and is differentiable through EnzymeVJP/InterpolatingAdjoint.
    function make_jump_callback(t_j, coord)
        condition(u, t, integrator) = t == t_j
        function affect!(integrator)
            z    = integrator.u
            inp  = vcat(z, coord)
            delta, _ = model_jump(inp, ps.jump, st.jump)
            integrator.u = z + delta
        end
        return DiscreteCallback(condition, affect!; save_positions=(false, true))
    end

    n_jumps = length(jump_times)
    callbacks = if n_jumps > 0
        # Broadcast coord if a single location is given for all jumps
        coords_iter = ndims(jump_coords) == 1 ?
                      [jump_coords for _ in 1:n_jumps] :
                      [jump_coords[i, :] for i in 1:n_jumps]
        cbs = [make_jump_callback(jump_times[i], coords_iter[i]) for i in 1:n_jumps]
        CallbackSet(cbs...)
    else
        CallbackSet()
    end

    prob = ODEProblem(ode_func, z0, t_span, ps)

    # Choose adjoint: prefer EnzymeVJP (mutation-safe, faster on H200),
    # fall back to ZygoteVJP if Enzyme.jl doesn't support this Lux version.
    sensealg = try
        InterpolatingAdjoint(autojacvec=EnzymeVJP())
    catch
        @warn "EnzymeVJP unavailable, falling back to ZygoteVJP" maxlog=1
        InterpolatingAdjoint(autojacvec=ZygoteVJP())
    end

    sol = solve(prob, Tsit5(), saveat=0.1,
                callback=callbacks,
                tstops=jump_times,
                sensealg=sensealg)
    return sol
end
