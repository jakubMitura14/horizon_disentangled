# Generate text output for Merlin (Julia/Lux) for comparison.

using Lux
using SafeTensors
using ComponentArrays
using Random
using Statistics

# Include model and utils
include("../src/models/merlin.jl")
include("../src/utils/merlin_loading.jl")

const OUTPUT_DIR = "verification_data/text_outputs"

function main()
    mkpath(OUTPUT_DIR)
    
    # Load Merlin verification vectors
    vectors_path = "verification_data/merlin_vectors.safetensors"
    if !isfile(vectors_path)
        println("Merlin vectors not found: $vectors_path")
        return
    end
    
    vectors = SafeTensors.load_safetensors(vectors_path)
    
    x_pt = vectors["input"]  # (C, D, H, W)
    y_pt_expected = vectors["output"]
    
    println("Input Shape (PyTorch): ", size(x_pt))
    println("Expected Output Shape: ", size(y_pt_expected))
    
    # Merlin input is 5D: (N, C, D, H, W) in PyTorch
    # Permute to Lux format: (W, H, D, C, N)
    x_lux = permutedims(x_pt, (5, 4, 3, 2, 1))
    # Already 5D, no reshape needed
    
    println("Input Shape (Lux): ", size(x_lux))
    
    # Initialize model
    rng = Random.default_rng()
    model = Merlin()
    ps, st = Lux.setup(rng, model)
    
    # Load weights
    weights_path = "verification_data/merlin_weights.safetensors"
    if !isfile(weights_path)
        println("Merlin weights not found: $weights_path")
        return
    end
    
    println("Loading Weights...")
    ps_loaded, st_loaded = load_merlin_weights(model, weights_path)
    
    println("Running Lux Inference...")
    st_run = Lux.testmode(st_loaded)
    y_lux, _ = model(x_lux, ps_loaded, st_run)
    
    println("Output Shape (Lux): ", size(y_lux))
    
    # Save text summary
    open(joinpath(OUTPUT_DIR, "merlin_julia_output.txt"), "w") do f
        write(f, "=== Merlin Julia (Lux) Output ===\n\n")
        write(f, "Input Shape: $(size(x_lux))\n")
        write(f, "Output Shape: $(size(y_lux))\n\n")
        write(f, "Input Stats:\n")
        write(f, "  Min: $(minimum(x_lux))\n")
        write(f, "  Max: $(maximum(x_lux))\n")
        write(f, "  Mean: $(mean(x_lux))\n")
        write(f, "  Std: $(std(x_lux))\n\n")
        write(f, "Output Stats:\n")
        write(f, "  Min: $(minimum(y_lux))\n")
        write(f, "  Max: $(maximum(y_lux))\n")
        write(f, "  Mean: $(mean(y_lux))\n")
        write(f, "  Std: $(std(y_lux))\n\n")
        
        # Sample values (first 20 elements of flattened output)
        flat = vec(y_lux)[1:20]
        write(f, "Sample Output Values (first 20):\n")
        for (i, v) in enumerate(flat)
            write(f, "  [$(i-1)]: $(v)\n")
        end
        
        # Comparison
        write(f, "\n\n=== Comparison with Python ===\n")
        # Merlin output is (Features, C, N) in Lux, (N, C, Features) in PyTorch
        # Permute PyTorch to match: (Features, C, N)
        y_pt_perm = permutedims(y_pt_expected, (3, 2, 1))
        diff = abs.(y_lux .- y_pt_perm)
        write(f, "Max Diff: $(maximum(diff))\n")
        write(f, "Mean Diff: $(mean(diff))\n")
    end
    
    println("Saved: $(joinpath(OUTPUT_DIR, "merlin_julia_output.txt"))")
end

main()
