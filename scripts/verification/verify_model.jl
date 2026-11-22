using Lux
using Random
using ComponentArrays
using SafeTensors
using NNlib

include("../src/models/totalsegmentator.jl")
include("../src/utils/weight_loading.jl")

function verify_model()
    rng = Random.default_rng()
    model = TotalSegmentator(118)
    
    ps, st = Lux.setup(rng, model)
    println("Model initialized. Parameter count: $(length(ComponentArray(ps)))")
    
    # Load weights
    weights_path = "external_sources/weights/Task297/model_final.safetensors"
    if !isfile(weights_path)
        println("Weights file not found at $weights_path")
        return
    end
    
    println("Loading weights...")
    ps_loaded = load_totalseg_weights(model, weights_path)
    
    # Verify structure
    # Check if ps_loaded has same structure as ps
    # converting to ComponentArray is a good way to check structure matching (if named tuples match)
    
    try
        ca_loaded = ComponentArray(ps_loaded)
        ca_init = ComponentArray(ps)
        
        println("Loaded parameters flattened size: $(length(ca_loaded))")
        println("Initial parameters flattened size: $(length(ca_init))")
        
        if length(ca_loaded) == length(ca_init)
             println("Parameter counts match!")
        else
             println("WARNING: Parameter counts do not match.")
             # We should debug diffs
        end
        
    catch e
        println("Error creating ComponentArray: $e")
        println("This likely means the NamedTuple structure does not match.")
        
        # Check specific path
        println("Init encoder keys: ", keys(ps.encoder))
        if haskey(ps.encoder, :stage1)
             println("Init stage1 keys: ", keys(ps.encoder.stage1))
             if haskey(ps.encoder.stage1, :layers)
                  println("Init stage1.layers keys: ", keys(ps.encoder.stage1.layers))
             else
                  println("Init stage1 has no :layers key! It has: ", keys(ps.encoder.stage1))
             end
        end
        
        println("Loaded encoder keys: ", keys(ps_loaded.encoder))
        if haskey(ps_loaded.encoder, :stage1)
             # ps_loaded might use Dict or NT
             println("Loaded stage1 keys: ", keys(ps_loaded.encoder.stage1))
             if haskey(ps_loaded.encoder.stage1, :layers)
                  println("Loaded stage1.layers keys: ", keys(ps_loaded.encoder.stage1.layers))
             end
        end

        rethrow(e)
    end
    
    # Dummy Inference
    println("Running dummy inference...")
    x = randn(Float32, 114, 114, 120, 1, 1) # W, H, D, C, N
    # We use a smaller size for speed if we can, but model expects specific strides.
    # U-Net usually handles any size divisible by 2^5 = 32.
    # 114 is not divisible by 32... 114/32 = 3.56.
    # nnU-Net handles padding internally usually.
    # Let's try 128x128x128 for simplicity.
    x_test = randn(Float32, 128, 128, 128, 1, 1)
    
    try
        y, _ = model(x_test, ps_loaded, st)
        println("Inference successful. Output shape: $(size(y))")
        
        if size(y, 4) == 118
            println("Output channels correct (118).")
        else
             println("Output channels mismatch. Expected 118, got $(size(y, 4))")
        end
        
    catch e
        println("Inference failed.")
        showerror(stdout, e)
        println()
        # rethrow(e)
    end
end

verify_model()
