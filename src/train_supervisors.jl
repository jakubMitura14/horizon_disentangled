using Lux
using Random
using Optimisers
using Zygote
using CSV
using DataFrames
using NIfTI
using Statistics
using Printf

include("models/supervisors.jl")

function load_data(data_dir)
    df = CSV.read(joinpath(data_dir, "clinical_data.csv"), DataFrame)
    # Simple loader: yield one batch of 2 samples
    row1 = df[1, :]
    row2 = df[2, :]

    # Load NIfTIs
    function load_vol(p)
        nii = niread(p)
        return Float32.(nii.raw)
    end

    t2w1 = load_vol(row1.t2w_path)
    adc1 = load_vol(row1.adc_path)
    seg1 = load_vol(row1.seg_path)

    t2w2 = load_vol(row2.t2w_path)
    adc2 = load_vol(row2.adc_path)
    seg2 = load_vol(row2.seg_path)

    # Concat channels: (W, H, D, C, B)
    # W=48, H=48, D=16
    # Input: 2 channels (T2W, ADC)

    x1 = cat(t2w1, adc1, dims=4) # (48,48,16,2)
    x2 = cat(t2w2, adc2, dims=4)

    batch_x = cat(x1, x2, dims=5) # (48,48,16,2,2)
    batch_seg = cat(seg1, seg2, dims=5)

    return batch_x, batch_seg, df
end

function train()
    data_dir = "src/mock_data"
    x, seg, df = load_data(data_dir)

    # Model
    model = UnetSupervisor(2, 3) # 3 classes
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    opt = Optimisers.Adam(1e-3)
    st_opt = Optimisers.setup(opt, ps)

    # Loss
    function loss_function(p, x, y, st)
        pred, st_new = model(x, p, st)
        # Simple MSE proxy for Dice for pilot compilation speed
        l = mean(abs2, pred .- y)
        return l, st_new
    end

    println("--- Training Segmentation Supervisor (Lux) ---")
    for i in 1:2
        # Zygote expects loss to return scalar
        # We need to extract state out or ignore it for gradient calc if state is non-differentiable (usually is)
        # Lux recommends using Zygote.pullback or similar for stateful.
        # Simplified:
        (l, st_new), back = Zygote.pullback(p -> loss_function(p, x, seg, st), ps)
        grads = back((1.0f0, nothing))[1]

        st_opt, ps = Optimisers.update(st_opt, ps, grads)
        st = st_new
        println("Epoch $i Loss: $l")
    end

    # Save checkpoint (mock)
    println("Supervisor trained.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
