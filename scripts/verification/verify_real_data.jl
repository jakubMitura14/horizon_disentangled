using Lux
using SafeTensors
using ComponentArrays
using Random
using Test
using Statistics

# Include model and utils
include("../src/models/totalsegmentator.jl")
include("../src/models/moose.jl")
include("../src/utils/weight_loading.jl")

function verify_case(name, model_type, weights_path, vectors_path, num_classes_expected)
    println("\n=== Verifying $name ===")
    
    # Load Vectors
    if !isfile(vectors_path)
        println("Vectors file not found: $vectors_path")
        return
    end
    vectors = SafeTensors.load_safetensors(vectors_path)
    
    x_pt = vectors["input"] # (C, D, H, W)
    y_pt = vectors["output"] # (Classes, D, H, W)
    
    println("PyTorch Input Shape: ", size(x_pt))
    println("PyTorch Output Shape: ", size(y_pt))
    
    # Preprocess Input for Lux
    # Lux expects (W, H, D, C, N)
    # PyTorch is (C, D, H, W)
    # Permute (C, D, H, W) -> (W, H, D, C)
    x_lux = permutedims(x_pt, (4, 3, 2, 1))
    
    # Add Batch Dimension
    x_lux = reshape(x_lux, size(x_lux)..., 1)
    
    println("Lux Input Shape: ", size(x_lux))
    
    # Initialize Model
    rng = Random.default_rng()
    
    if model_type == "TotalSegmentator"
        model = TotalSegmentator(num_classes_expected)
        # Use totalseg loader
        loader_func = load_totalseg_weights
    elseif model_type == "MOOSE"
        model = MooseModel(num_classes_expected)
        # Use moose loader
        loader_func = load_moose_weights
    else
        error("Unknown model type")
    end
    
    ps, st = Lux.setup(rng, model)
    
    # Load Weights
    println("Loading Weights from $weights_path...")
    ps_loaded = loader_func(model, weights_path)
    
    # Run Inference
    println("Running Lux Inference...")
    st_run = Lux.testmode(st)
    y_lux, _ = model(x_lux, ps_loaded, st_run)
    
    println("Lux Output Shape: ", size(y_lux))
    
    # Compare
    # Lux Output: (W, H, D, Classes, N)
    # PyTorch Output: (Classes, D, H, W)
    
    # Permute PyTorch to match Lux
    # (Classes, D, H, W) -> (W, H, D, Classes)
    y_pt_perm = permutedims(y_pt, (4, 3, 2, 1))
    
    # Squeeze Lux N dim
    y_lux_sq = dropdims(y_lux, dims=5)
    
    diff = abs.(y_lux_sq .- y_pt_perm)
    max_diff = maximum(diff)
    mean_diff = mean(diff)
    
    println("Max Diff: ", max_diff)
    println("Mean Diff: ", mean_diff)
    println("Mean Abs PT: ", mean(abs.(y_pt_perm)))
    
    if max_diff < 1e-3
        println("SUCCESS: $name Verified.")
    else
        println("WARNING: $name Verification Failed or High Error.")
    end
end

function main()
    # verify_case("TotalSegmentator CT", "TotalSegmentator", 
    #     "/media/jm/hddData/projects_new/horizon_disentangled/verification_data/totalsegmentator_weights.safetensors",
    #     "verification_data/real_ts_ct.safetensors", 117) 
    # NOTE: NumClasses for Task 297 is 117? No, it's 104 + others?
    # Verify totalseg script used 118 (117 classes + background?) or similar.
    # Actually setup_totalsegmentator.py downloaded Task 297.
    # verify_totalsegmentator.jl used inferred num_classes.
    
    # Check num_classes from vectors dynamically?
    # Or strict hardcoding.
    # We can infer from vectors["output"] size(x, 1) if loaded.
    
    # Paths 
    ts_ct_weights = "verification_data/totalseg_weights.safetensors"
    ts_mri_weights = "verification_data/totalseg_mri_weights.safetensors"
    moose_weights = "verification_data/moose_brain_weights.safetensors"
    
    # Infer Num Classes helper
    function get_classes(path)
        if !isfile(path) return 0 end
        t = SafeTensors.load_safetensors(path)
        return size(t["output"], 1)
    end
    
    # 1. CT
    name = "TotalSegmentator CT"
    vec_path = "verification_data/real_ts_ct.safetensors"
    classes = get_classes(vec_path)
    if classes > 0
        verify_case(name, "TotalSegmentator", ts_ct_weights, vec_path, classes)
    end

    # 2. MRI
    name = "TotalSegmentator MRI"
    vec_path = "verification_data/real_ts_mri.safetensors"
    classes = get_classes(vec_path)
    if classes > 0
        verify_case(name, "TotalSegmentator", ts_mri_weights, vec_path, classes)
    end
    
    # 3. MOOSE
    name = "MOOSE PET"
    vec_path = "verification_data/real_moose_pet.safetensors"
    classes = get_classes(vec_path)
    if classes > 0
        # Check if vectors exist
        verify_case(name, "MOOSE", moose_weights, vec_path, classes)
    end
end

main()
