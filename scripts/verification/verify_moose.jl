using Lux
using SafeTensors
using ComponentArrays
using Random
using Test
using Statistics

# Include model and utils
include("../src/models/moose.jl")
include("../src/utils/weight_loading.jl")

function verify_moose()
    println("Loading MOOSE verification vectors...")
    vectors_path = "verification_data/vectors_moose.safetensors"
    vectors = SafeTensors.load_safetensors(vectors_path)
    
    x_pt_raw = vectors["input"] # (N, C, D, H, W) -> (1, 1, 64, 64, 64)
    y_pt_raw = vectors["output"] # (N, C, D, H, W)
    
    # Permute Input to Lux: (W, H, D, C, N)
    x_lux = permutedims(x_pt_raw, (5, 4, 3, 2, 1))
    
    println("Lux Input Shape: ", size(x_lux))
    
    # Infer Num Classes
    num_classes = size(y_pt_raw, 2)
    println("Lux Output Target Shape: ", (size(x_lux, 1), size(x_lux, 2), size(x_lux, 3), num_classes, 1))
    println("Inferred Num Classes: ", num_classes)
    
    # Initialize Model
    model = MooseModel(num_classes)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    # Load Weights
    println("Loading MOOSE weights...")
    weights_path = "verification_data/moose_brain_weights.safetensors"
    ps_loaded = load_moose_weights(model, weights_path)
    
    # Run Inference
    println("Running Inference...")
    st_run = Lux.testmode(st)
    
    # Ensure dimensions are correct
    y_lux, _ = model(x_lux, ps_loaded, st_run)
    println("Lux Output Shape: ", size(y_lux))
    
    # Compare
    # Permute PyTorch Output to Match Lux: (W, H, D, C, N)
    y_pt = permutedims(y_pt_raw, (5, 4, 3, 2, 1))
    
    diff = abs.(y_lux .- y_pt)
    max_diff = maximum(diff)
    mean_diff = mean(diff)
    
    println("Max Diff: ", max_diff)
    println("Mean Diff: ", mean_diff)
    
    println("Mean Abs PyTorch: ", mean(abs.(y_pt)))
    println("Mean Abs Lux: ", mean(abs.(y_lux)))
    
    if max_diff < 1e-3
        println("SUCCESS: MOOSE Verified (Low Error).")
    else
        println("WARNING: MOOSE Verification Error High.")
    end
end

verify_moose()
