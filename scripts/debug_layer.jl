using Lux
using SafeTensors
using ComponentArrays
using Random
using NNlib

function debug_layer()
    tensors = SafeTensors.load_safetensors("verification_data/layer_debug.safetensors")
    
    x_pt = tensors["input"] # (1, 1, 32, 32, 32) -> N, C, D, H, W
    y_pt = tensors["output"] # (1, 32, 32, 32, 32)
    w_pt = tensors["weight"] # (32, 1, 3, 3, 3) -> Out, In, D, H, W
    b_pt = tensors["bias"] # (32,)
    
    # --- Transformation ---
    # Input: (W, H, D, C, N) logic for Lux
    # PyTorch: N, C, D, H, W
    # We want W to be dim 1. W is last in PyTorch shape.
    # reshape(x, reverse(size)) ?
    # Let's test reshape vs permutedims on input.
    # If x_pt was created in Python, it's row-major.
    # x_pt[n, c, d, h, w]
    # Julia reads it as linear.
    # If we reshape to (W, H, D, C, N), we are essentially correctly mapping the linear order 
    # IF the linear order in file iterates W fastest.
    # PyTorch memory: x[0,0,0,0,0], x[0,0,0,0,1] ...
    # So yes, W is fastest.
    
    x_lux = reshape(x_pt, reverse(size(x_pt)))
    y_lux_ref = reshape(y_pt, reverse(size(y_pt)))
    b_lux = b_pt # 1D
    
    # Weights
    # PyTorch: (Out, In, D, H, W). W fastest.
    # Lux Conv: Input (W, H, D, C_in, N). Weight (W, H, D, C_in, C_out).
    # We want W to be dim 1 (fastest).
    # PyTorch W is dim 5 (fastest).
    # So `reshape(w_pt, reverse(size(w_pt)))` -> (W, H, D, In, Out).
    # Let's try this.
    
    w_lux = reshape(w_pt, reverse(size(w_pt)))
    
    println("Lux Input: ", size(x_lux))
    println("Lux Weight: ", size(w_lux))
    
    # Layer
    # Conv(kernel, in=>out)
    layer = Conv((3,3,3), 1=>32, stride=1, pad=1, use_bias=true)
    
    ps = (weight=w_lux, bias=b_lux)
    st = (;)
    
    y_lux, _ = layer(x_lux, ps, st)
    
    diff = abs.(y_lux .- y_lux_ref)
    println("Max Diff: ", maximum(diff))
    println("Mean Diff: ", sum(diff)/length(diff))
    
    if maximum(diff) < 1e-4
        println("SUCCESS: Reshape works!")
    else
        println("FAILURE: Reshape incorrect.")
        
        # Try Permutedims
        println("\nTrying permutedims...")
        # PyTorch: (Out, In, D, H, W) -> (W, H, D, In, Out)
        # Permutation: 5, 4, 3, 2, 1
        w_lux_p = permutedims(w_pt, (5, 4, 3, 2, 1))
        
        # We also need to permute Input and Output if we assume loading was wrong?
        # NO, SafeTensors.jl loading of arrays is just reshaping byte stream usually?
        # If I use `permutedims` on the loaded array, I am shuffling data in memory.
        # If `reshape` failed, it means the data layout in memory (after load) was correct for (W..Out) access 
        # OR `Conv` expects something else.
        
        ps_p = (weight=w_lux_p, bias=b_lux)
        y_lux_p, _ = layer(x_lux, ps_p, st)
        
        diff_p = abs.(y_lux_p .- y_lux_ref)
        println("Max Diff (Permuted): ", maximum(diff_p))
        
        if maximum(diff_p) < 1e-4
             println("SUCCESS: Permutedims works!")
        else
             # Try Transpose logic?
             println("FAILURE: Both failed.")
        end
    end
end

debug_layer()
