using Lux
using Random
using Optimisers
using Zygote
using Statistics

include("models/vae.jl")

function train()
    # Mock Data (48, 48, 16, 2, 2)
    img = rand(Float32, 48, 48, 16, 2, 2)
    mask = rand(Float32, 48, 48, 16, 1, 2)

    model = CausalVAE(16, 4)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    opt = Optimisers.Adam(1e-3)
    st_opt = Optimisers.setup(opt, ps)

    function loss_fn(p, x, m, st)
        # Forward
        (out, ), st_new = model((x, m), p, st)
        # out is a Tuple: (recon, mu_p, log_p, mu_s, log_s)
        # However, Lux returns (y, st).
        # We destructured (out, ) above, so out is the tuple.

        recon = out[1]
        mu_p = out[2]
        log_p = out[3]

        # Recon loss
        l_rec = mean(abs, recon .- x)
        # KL
        kld = -0.5f0 * mean(1 .+ log_p .- mu_p.^2 .- exp.(log_p))

        loss = l_rec + 0.1f0 * kld
        return loss, st_new
    end

    println("--- Training Causal VAE (Lux) ---")
    for i in 1:2
        (l, st_new), back = Zygote.pullback(p -> loss_fn(p, img, mask, st), ps)
        grads = back((1.0f0, nothing))[1]

        st_opt, ps = Optimisers.update(st_opt, ps, grads)
        st = st_new
        println("Epoch $i Loss: $l")
    end
    println("VAE Trained.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
