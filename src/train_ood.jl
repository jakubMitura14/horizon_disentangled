using Lux
using Random
using Optimisers
using Zygote
using Statistics
using ComponentArrays
using TensorBoardLogger
using Logging

include("models/ood.jl")

function train()
    # Mock Latents (16, 100)
    # In real pipeline, load from DATA_DIR (inference from VAE)
    z = randn(Float32, 16, 100)

    # Logger
    logger = TBLogger("logs/ood", min_level=Logging.Info)

    # Model
    model_def = OODDetector(16, 4)
    enc = Chain(Dense(16 => 12, relu), Dense(12 => 8)) # 4*2
    dec = Chain(Dense(4 => 12, relu), Dense(12 => 16))

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
    with_logger(logger) do
        for i in 1:2
            (l, st_new), back = Zygote.pullback(p -> loss_fn(p, z, st), ps)
            grads = back((1.0f0, nothing))[1]
            st_opt, ps = Optimisers.update(st_opt, ps, grads)
            st = st_new

            @info "train" loss=l epoch=i
            println("Epoch $i Loss: $l")
        end
    end

    println("OOD Detection Validated.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
