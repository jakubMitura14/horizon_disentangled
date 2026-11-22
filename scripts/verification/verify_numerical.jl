using Lux
using ComponentArrays
using SafeTensors
using Statistics
import LuxCore

include("../src/models/totalsegmentator.jl")
include("../src/utils/weight_loading.jl")

function verify_numerical()
    println("Loading verification vectors...")
    vectors = SafeTensors.load_safetensors("verification_data/vectors.safetensors")
    x_pt = vectors["input"] # (1, 1, 64, 64, 64) -> N, C, D, H, W
    y_pt = vectors["output"] # (1, 118, 64, 64, 64)
    
    # Permute to Lux format: (W, H, D, C, N)
    x_lux = permutedims(x_pt, (5, 4, 3, 2, 1))
    y_lux_ref = permutedims(y_pt, (5, 4, 3, 2, 1))
    
    println("Input shape (Lux): ", size(x_lux))
    println("Ref Output shape (Lux): ", size(y_lux_ref))
    
    # Initialize Model
    model = TotalSegmentator(118)
    # Using `check_loading` logic
    ps_loaded = load_totalseg_weights(model, "external_sources/weights/Task297/model_final.safetensors")
    
    # Setup dummy state
    rng = Random.default_rng()
    _, st = Lux.setup(rng, model)
    st = LuxCore.testmode(st) # Important for normalization
    
    println("Running Lux inference...")
    y_lux, _ = model(x_lux, ps_loaded, st)
    
    println("Lux Output shape: ", size(y_lux))
    
    # Compare
    diff = abs.(y_lux .- y_lux_ref)
    max_diff = maximum(diff)
    mean_diff = mean(diff)
    
    println("Max Difference: ", max_diff)
    println("Mean Difference: ", mean_diff)
    
    if max_diff < 1e-4
        println("SUCCESS: Numerical match!")
    else
        println("FAILURE: Mismatch detected.")
        # Analyze mismatch location?
    end
end

verify_numerical()
