# Stage 4 Julia SciML Biophysical UDE-PDE-NJDE Integration Bridge
# Executes on GPU 2 for high-throughput trajectory integration across 4x H200 cluster.

using OrdinaryDiffEq
using DiffEqCallbacks
using SciMLSensitivity
using ComponentArrays
using HDF5

"""
Defines the coarse 3D Cartesian grid finite-difference Laplacian.
"""
function laplacian3d!(lap, u, dx)
    W, H, D, C, B = size(u)
    inv_dx2 = 1.0f0 / (dx * dx)
    @inbounds for b in 1:B, c in 1:C, k in 2:(D-1), j in 2:(H-1), i in 2:(W-1)
        lap[i, j, k, c, b] = (
            u[i+1, j, k, c, b] + u[i-1, j, k, c, b] +
            u[i, j+1, k, c, b] + u[i, j-1, k, c, b] +
            u[i, j, k+1, c, b] + u[i, j, k-1, c, b] -
            6.0f0 * u[i, j, k, c, b]
        ) * inv_dx2
    end
end

"""
Continuous drift evaluation for GBM biophysical system coupled to systemic vector.
"""
function gbm_pde_drift!(du, u, p, t)
    # u is ComponentArray with fields:
    # u.U_D: (W, H, D, 8, B)
    # u.v_sys: (160, B)
    dx = p.dx
    diff_w = p.diff_w

    # Compute diffusion on viable tumor channel 1
    laplacian3d!(@view(du.U_D[:, :, :, 1, :]), @view(u.U_D[:, :, :, 1, :]), dx)
    @. du.U_D[:, :, :, 1, :] = diff_w * du.U_D[:, :, :, 1, :] + 0.03f0 * u.U_D[:, :, :, 1, :] * (1.0f0 - u.U_D[:, :, :, 1, :] - u.U_D[:, :, :, 2, :])
    
    # Necrotic clearance on channel 2
    @. du.U_D[:, :, :, 2, :] = 0.01f0 * u.U_D[:, :, :, 1, :] - 0.05f0 * u.U_D[:, :, :, 2, :]
    
    # Systemic decay
    @. du.v_sys = -0.01f0 * u.v_sys
end

"""
Executes trajectory integration with PresetTimeCallbacks for discrete radiation/chemo jumps.
"""
function solve_trajectory(u0, p, tspan, jump_times, jump_doses)
    # Construct PresetTimeCallback for discrete RT/chemotherapy shocks
    affect! = function (integrator)
        # Apply LQ radiation shock on viable tumor
        integrator.u.U_D[:, :, :, 1, :] .*= 0.7f0
        # Transfer into damaged pool
        integrator.u.U_D[:, :, :, 2, :] .+= 0.3f0
    end

    cb = PresetTimeCallback(jump_times, affect!; save_positions=(false, false))
    prob = ODEProblem(gbm_pde_drift!, u0, tspan, p)
    sol = solve(prob, Tsit5(), callback=cb, dt=7.0f0, adaptive=true, saveat=jump_times)
    return sol
end

println("Julia SciML Stage 4 integration bridge initialized successfully.")
