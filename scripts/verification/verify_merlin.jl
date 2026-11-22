using Lux
using SafeTensors
using ComponentArrays
using Random
using NNlib
using Statistics

include("../src/models/merlin.jl")
include("../src/utils/merlin_loading.jl")

function verify_merlin()
    println("Loading verification vectors...")
    vectors = SafeTensors.load_safetensors("verification_data/merlin_vectors.safetensors")
    x_pt = vectors["input"] # (1, 1, 16, 64, 64) -> (N, C, D, H, W)
    y_pt = vectors["output"] # (1, 1, 2048)
    
    # Transformation Logic
    # Merlin Python Architecture rotates input: (N, C, D, H, W) -> (N, C, W, D, H)
    # Lux/NNlib Conv3D expects spatial dims reversed compared to PyTorch Conv3D input.
    # PyTorch Spatial Input: (W, D, H)
    # Lux Spatial Input: (H, D, W)
    # Full Lux Input: (H, D, W, C, N)
    # Indices from Original (N, C, D, H, W): (4, 3, 5, 2, 1)
    
    x_lux = permutedims(x_pt, (4, 3, 5, 2, 1))
    
    println("Lux Input Shape: ", size(x_lux))
    
    # Initialize Model
    # We must ensure params match.
    # merlin.jl Merlin() defaults.
    model = Merlin()
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    # Load Weights
    println("Loading weights...")
    ps_loaded, st_loaded = load_merlin_weights(model, "external_sources/weights/Merlin/merlin_image_encoder.safetensors")
    
    println("Loaded PS keys: ", propertynames(ps_loaded))
    
    # Debug State
    if hasproperty(st_loaded, :layer1)
        l1 = st_loaded.layer1
        if hasproperty(l1, :layer_1)
            l1_1 = l1.layer_1
            println("layer1.layer_1 keys: ", keys(l1_1))
            if haskey(l1_1, :downsample)
                println("layer1.layer_1.downsample type: ", typeof(l1_1.downsample))
                println("layer1.layer_1.downsample: ", l1_1.downsample)
            else
                println("layer1.layer_1.downsample MISING")
            end
        end
    end
    
    # Set to testmode? BatchNorm running stats used.
    st_run = Lux.testmode(st_loaded) 
    
    println("st_run generated.")
    if hasproperty(st_run, :layer1)
         l1 = st_run.layer1
         if hasproperty(l1, :layer_1)
             l1_1 = l1.layer_1
             if haskey(l1_1, :downsample)
                 println("st_run.layer1.layer_1.downsample type: ", typeof(l1_1.downsample))
                 println("st_run.layer1.layer_1.downsample: ", l1_1.downsample)
             else
                 println("st_run.layer1.layer_1.downsample MISSING")
             end
         end
    end

    println("Running Inference...")
    y_lux = nothing
    try
        y_lux, _ = model(x_lux, ps_loaded, st_run)
        println("Lux Output Shape: ", size(y_lux))
    catch e
        showerror(stdout, e, catch_backtrace())
        println()
        return
    end
    # Expected (1, 1, 1, 2048, 1)
    
    # Reshape PyTorch output to compare
    # PyTorch (1, 1, 2048).
    # Lux (1, 1, 1, 2048, 1).
    # Flatten both
    y_lux_flat = vec(y_lux)
    y_pt_flat = vec(y_pt)
    
    diff = abs.(y_lux_flat .- y_pt_flat)
    println("Max Diff: ", maximum(diff))
    println("Mean Diff: ", mean(diff))
    println("Mean Abs PyTorch: ", mean(abs.(y_pt_flat)))
    println("Mean Abs Lux: ", mean(abs.(y_lux_flat)))
    
    if maximum(diff) < 1e-3 # Looser tolerance for FP32/Deep Network
        println("SUCCESS: Merlin Verified!")
    else
        println("FAILURE: Mismatch.")
    end
end

verify_merlin()
