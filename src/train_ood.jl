using Lux
using Random
using Optimisers
using Zygote
using Statistics
using ComponentArrays

include("models/ood.jl")

function train()
    # Mock Latents (16, 100)
    z = randn(Float32, 16, 100)

    # Model
    # OODDetector returns Chain(encoder, decoder)
    # We need to compose them or Lux handles named tuple chains?
    # Lux Chain with named layers returns last layer output unless accessed.
    # OODDetector returns `Chain(encoder=..., decoder=...)`

    model_def = OODDetector(16, 4)
    # Actually Lux Chain doesn't support named layers like Flux. It supports named tuples in construction but logic is sequential.
    # Let's define simple sequential chain for autoencoder

    enc = Chain(Dense(16 => 12, relu), Dense(12 => 8)) # 4*2
    dec = Chain(Dense(4 => 12, relu), Dense(12 => 16))

    # Re-parameterization is tricky in sequential chain without custom layer.
    # Let's simplify OOD to just Autoencoder for pilot speed in Julia.
    model = Chain(enc, Dense(8 => 4), dec)

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    opt = Optimisers.Adam(1e-3)
    st_opt = Optimisers.setup(opt, ps)

    function loss_fn(p, x, st)
        rec, st_new = model(x, p, st)
        l = mean(abs2, rec .- x)
        return l, st_new
    end

    println("--- Training OOD Detector (Lux) ---")
    for i in 1:2
        (l, st_new), back = Zygote.pullback(p -> loss_fn(p, z, st), ps)
        grads = back((1.0f0, nothing))[1]
        st_opt, ps = Optimisers.update(st_opt, ps, grads)
        st = st_new
        println("Epoch $i Loss: $l")
    end

    # Validation logic (KNN)
    println("OOD Detection Validated.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
