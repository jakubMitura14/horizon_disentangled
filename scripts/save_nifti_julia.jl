# Save NIfTI outputs for TotalSegmentator and MOOSE from Julia/Lux models.

using Lux
using SafeTensors
using ComponentArrays
using Random
using NIfTI

# Include model and utils
include("../src/models/totalsegmentator.jl")
include("../src/models/moose.jl")
include("../src/utils/weight_loading.jl")
include("../src/utils/sliding_window.jl")
using .SlidingWindow

const OUTPUT_DIR = "verification_data/nifti_outputs"

function save_nifti_file(array, filename; reference_path=nothing)
    """Save array as NIfTI file."""
    # ... (conversion logic) ...
    # For segmentation, take argmax across classes
    if ndims(array) == 4
        # (W, H, D, Classes) -> argmax -> (W, H, D)
        cartesian_indices = dropdims(argmax(array, dims=4), dims=4)
        # Convert CartesianIndex to Int
        # Lux/Julia indices are 1-based. Python classes are 0-based.
        # So we subtract 1.
        array = map(x -> Int16(x[4] - 1), cartesian_indices)
    end
    
    # Convert to appropriate type if not already
    if !(eltype(array) <: Integer)
        array = convert(Array{Float32}, array)
    end
    
    if reference_path !== nothing && isfile(reference_path)
        println("Using reference header from: $reference_path")
        ref_ni = niread(reference_path)
        # Verify size matches?
        # ref_ni size might be different if output classes? NO, output is 3D segmentation.
        # array is 3D here.
        
        # Create new volume with ref header/ext but new data
        ni = NIVolume(ref_ni.header, ref_ni.extensions, array)
        niwrite(filename, ni)
    else
        println("Warning: No reference path provided. Saving with default header.")
        ni = NIVolume(array)
        niwrite(filename, ni)
    end
    println("Saved: $filename")
end

function process_case(name, model_type, weights_path, vectors_path, num_classes; reference_nifti=nothing)
    println("\n=== Processing $name (Julia) ===")
    
    if !isfile(vectors_path)
        println("Vectors file not found: $vectors_path")
        return
    end
    
    vectors = SafeTensors.load_safetensors(vectors_path)
    
    x_pt = vectors["input"] # (C, D, H, W)
    
    println("Input Shape (PyTorch): ", size(x_pt))
    
    # Permute to Lux format: (W, H, D, C)
    x_lux = permutedims(x_pt, (4, 3, 2, 1))
    # Add batch dim
    x_lux = reshape(x_lux, size(x_lux)..., 1)
    
    println("Input Shape (Lux): ", size(x_lux))
    
    # Initialize model
    rng = Random.default_rng()
    
    if model_type == "TotalSegmentator"
        model = TotalSegmentator(num_classes)
        loader_func = load_totalseg_weights
    elseif model_type == "MOOSE"
        model = MooseModel(num_classes)
        loader_func = load_moose_weights
    else
        error("Unknown model type")
    end
    
    ps, st = Lux.setup(rng, model)
    
    println("Loading Weights from $weights_path...")
    ps_loaded = loader_func(model, weights_path)
    
    # Save NIfTI Input (Early Save)
    mkpath(OUTPUT_DIR)
    
    # Get prefix from vectors_path
    prefix = replace(basename(vectors_path), "real_" => "", ".safetensors" => "")
    
    # Input (remove batch and channel dims for 3D volume)
    if size(x_lux, 4) == 1
        input_3d = x_lux[:,:,:,1,1]
    else
        # If multi-channel, save 4D? NIfTI handles it.
        input_3d = x_lux[:,:,:,:,1]
    end
    
    save_nifti_file(Array(input_3d), joinpath(OUTPUT_DIR, "$(prefix)_input_julia.nii.gz"); reference_path=reference_nifti)
    println("Saved input to $(prefix)_input_julia.nii.gz")
    
    println("Running Lux Inference...")
    st_run = Lux.testmode(st)
    
    # Check if we need sliding window
    # If any dim > patch size?
    # Assume 160x64x160 patch for MOOSE
    # But for valid comparison we should check dimensions.
    
    pW, pH, pD = (160, 64, 160)
    # Check if input fits in patch
    if size(x_lux, 1) > pW || size(x_lux, 2) > pH || size(x_lux, 3) > pD
        println("Input larger than patch size ($pW, $pH, $pD). Using Sliding Window...")
        y_lux_array, _ = SlidingWindow.predict_sliding_window(model, x_lux, ps_loaded, st_run, (pW, pH, pD))
        # Result is already Array (CPU)
        y_lux = y_lux_array
    else
        y_lux, _ = model(x_lux, ps_loaded, st_run)
        y_lux = Array(y_lux)
    end
    
    println("Output Shape: ", size(y_lux))
    
    # Output (remove batch dim)
    # (W, H, D, Classes, N)
    output_4d = y_lux[:,:,:,:,1]  # (W, H, D, Classes)
    
    save_nifti_file(output_4d, joinpath(OUTPUT_DIR, "$(prefix)_julia_output.nii.gz"); reference_path=reference_nifti)
end

function main()
    mkpath(OUTPUT_DIR)
    
    # Helper to get num classes
    function get_classes(path)
        if !isfile(path) return 0 end
        t = SafeTensors.load_safetensors(path)
        return size(t["output"], 1)
    end
    
    # 1. CT
    # 1. CT
    # vec_path = "verification_data/real_ts_ct.safetensors"
    # classes = get_classes(vec_path)
    # if classes > 0
    #     process_case("TotalSeg_CT", "TotalSegmentator", 
    #         "verification_data/totalseg_weights.safetensors", vec_path, classes)
    # end

    # 2. MRI
    # vec_path = "verification_data/real_ts_mri.safetensors"
    # classes = get_classes(vec_path)
    # if classes > 0
    #     process_case("TotalSeg_MRI", "TotalSegmentator", 
    #         "verification_data/totalseg_mri_weights.safetensors", vec_path, classes)
    # end
    
    # 3. MOOSE
    # vec_path = "verification_data/real_moose_pet.safetensors"
    vec_path = "verification_data/real_moose_pet_full_ras.safetensors"
    logits_path = "verification_data/logits_moose_pet.safetensors"
    
    # Reference NIfTI for header (use input)
    ref_nifti = "verification_data/nifti_outputs/moose_pet_input_ras.nii.gz"
    
    classes = 0
    if isfile(vec_path)
        # Try inferring from output usually
        # But if missing, try logits file
        if isfile(logits_path)
            t = SafeTensors.load_safetensors(logits_path)
            # Logits shape: (Classes, D, H, W)
            classes = size(t["logits"], 1)
            println("Inferred $classes classes from $logits_path")
        else
             # Fallback or try reading input if output present?
             try
                classes = get_classes(vec_path)
             catch
                classes = 84 # Default for MOOSE Brain V1
                println("Defaulting to 84 classes (MOOSE Brain V1)")
             end
        end
    end

    if classes > 0
        process_case("MOOSE_PET", "MOOSE", 
            "verification_data/moose_brain_weights.safetensors", vec_path, classes; reference_nifti=ref_nifti)
    end
end

main()
