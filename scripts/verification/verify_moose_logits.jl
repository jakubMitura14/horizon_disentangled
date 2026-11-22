using Lux
using SafeTensors
using ComponentArrays
using Random
using Statistics

include("../src/models/moose.jl")
include("../src/utils/weight_loading.jl")

function verify_moose_logits()
    println("=== Verifying MOOSE Logits ===")
    
    path = "verification_data/logits_moose_pet.safetensors"
    if !isfile(path)
        println("File not found: $path")
        return
    end
    
    tensors = SafeTensors.load_safetensors(path)
    # Python input: (C, D, H, W)
    x_pt = tensors["input"]
    # Python logits: (Classes, D, H, W)
    y_pt = tensors["logits"]
    
    println("Python Input Shape: ", size(x_pt))
    println("Python Logits Shape: ", size(y_pt))
    
    # Infer num classes
    num_classes = size(y_pt, 1) # Dim 1 is classes in (C, D, H, W)
    println("Num Classes: ", num_classes)
    
    # Permute Input to Lux: (W, H, D, C, N)
    # Python (C, D, H, W) -> Lux (W, H, D, C, 1)
    x_lux = permutedims(x_pt, (4, 3, 2, 1))
    x_lux = reshape(x_lux, size(x_lux)..., 1)
    
    println("Lux Input Shape: ", size(x_lux))
    
    # Permute Reference: (W, H, D, C, N)
    y_ref = permutedims(y_pt, (4, 3, 2, 1))
    y_ref = reshape(y_ref, size(y_ref)..., 1)
    
    # Setup Model
    model = MooseModel(num_classes)
    weights_path = "verification_data/moose_brain_weights.safetensors"
    
    println("Loading weights from $weights_path")
    # Uses load_moose_weights with FLIPPING logic
    ps = load_moose_weights(model, weights_path)
    
    rng = Random.default_rng()
    _, st = Lux.setup(rng, model)
    st = Lux.testmode(st)
    
    println("Running Lux Inference (MOOSE)...")
    y_lux, _ = model(x_lux, ps, st)
    
    println("Lux Output Shape: ", size(y_lux))
    
    # Compare
    diff = abs.(y_lux .- y_ref)
    max_diff = maximum(diff)
    mean_diff = mean(diff)
    
    println("Max Difference: ", max_diff)
    println("Mean Difference: ", mean_diff)
    
    if max_diff > 1.0
        println("\nFAILURE: Large mismatch.")
    else
        println("\nSUCCESS: Match.")
    end
    
    # Check Segmentation (Argmax matching)
    println("\nChecking Segmentation Matching...")
    
    function get_labels(logits)
        # dims=4 is Channels
        # argmax returns CartesianIndex(w, h, d, c, n)
        # We want c
        am = dropdims(argmax(logits, dims=4), dims=4)
        return map(x -> x[4], am)
    end
    
    labels_lux = get_labels(y_lux)
    labels_ref = get_labels(y_ref)
    
    mismatch_count = count(labels_lux .!= labels_ref)
    total_voxels = length(labels_lux)
    prob_mismatch = mismatch_count / total_voxels
    
    println("Use labels (1-based Julia index): ", labels_lux[1:5])
    println("Ref labels (1-based Ref index): ", labels_ref[1:5]) 
    
    println("Segmentation Mismatch Fraction: ", prob_mismatch)
    
    if prob_mismatch < 0.01
        println("SUCCESS: Segmentation matches (>99%)")
    else
        println("WARNING: High mismatch in segmentation.")
    end
end

verify_moose_logits()
