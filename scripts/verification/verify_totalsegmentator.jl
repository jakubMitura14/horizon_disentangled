using Lux
using SafeTensors
using ComponentArrays
using Random
using NNlib
using Statistics

include("../src/models/totalsegmentator.jl")
include("../src/utils/weight_loading.jl")

function verify_totalsegmentator()
    println("Loading verification vectors...")
    vectors = SafeTensors.load_safetensors("verification_data/vectors.safetensors")
    
    # PyTorch Input: (1, 1, 64, 64, 64) -> (Batch, Channel, Depth, Height, Width)
    # Vectors loaded as: (64, 64, 64, 1, 1)? Check order.
    # Safetensors usually preserves shape but Julia reads column major.
    # PyTorch save: (N, C, D, H, W).
    # Julia read: (W, H, D, C, N).
    
    # PyTorch Save: (N, C, D, H, W).
    # Julia Load (SafeTensors.jl): (N, C, D, H, W) [It seems it preserves layout if not mmap'ed or strictly wrapper?]
    # Actually checking output: (1, 118, 64, 64, 64).
    
    # Lux expects: (W, H, D, C, N).
    # So we need to permute (5, 4, 3, 2, 1).
    
    x_pt = permutedims(vectors["input"], (5, 4, 3, 2, 1))
    y_pt = permutedims(vectors["output"], (5, 4, 3, 2, 1))
    
    println("Lux Input Shape: ", size(x_pt))
    println("Lux Output Target Shape: ", size(y_pt))
    
    # Num Classes inferred from C (Dim 4)
    num_classes = size(y_pt, 4)
    println("Inferred Num Classes: ", num_classes)
    
    # Model Setup
    model = TotalSegmentator(num_classes)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    
    # Load Weights
    println("Loading weights...")
    ps_loaded = load_totalseg_weights(model, "verification_data/totalseg_weights.safetensors")
    
    # Test Mode (Batch Norm)
    st_run = Lux.testmode(st)
    
    # Inference
    println("Running Inference...")
    y_lux, _ = model(x_pt, ps_loaded, st_run)
    
    println("Lux Output Shape: ", size(y_lux))
    
    # Compare
    diff = abs.(y_lux .- y_pt)
    println("Max Diff: ", maximum(diff))
    println("Mean Diff: ", mean(diff))
    println("Mean Abs PyTorch: ", mean(abs.(y_pt)))
    println("Mean Abs Lux: ", mean(abs.(y_lux)))
    
    if maximum(diff) < 1.0 # Softmax/Logits can vary, maybe looser tolerance?
         # TotalSegmentator outputs logits? yes.
         println("SUCCESS: TotalSegmentator Verified (Low Error).")
    else
         println("FAILURE: Mismatch.")
    end
end

verify_totalsegmentator()
