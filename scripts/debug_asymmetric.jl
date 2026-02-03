using Lux
using SafeTensors
using NNlib

function debug_asymmetric()
    tensors = SafeTensors.load_safetensors("verification_data/asymmetric_debug.safetensors")
    
    # Try 2: Permutedims + Flip Spatial (to match Cross Correlation)
    x_p = permutedims(tensors["input"], (5, 4, 3, 2, 1))
    y_ref_p = permutedims(tensors["output"], (5, 4, 3, 2, 1))
    w_p = permutedims(tensors["weight"], (5, 4, 3, 2, 1)) 
    w_p = reverse(w_p, dims=(1, 2, 3))
    
    layer = Conv((3,3,3), 2=>2, stride=1, pad=1, use_bias=false)
    
    println("--- Test: Permutedims ---")
    
    # Trace Input
    inds_x = findall(x->abs(x)>0.5, x_p)
    println("Active Input indices: ", inds_x)
    
    # Trace Weight
    inds_w = findall(x->abs(x)>0.5, w_p)
    println("Active Weight indices: ", inds_w)
    
    ps = (weight=w_p,)
    y, _ = layer(x_p, ps, (;))
    
    diff = abs.(y .- y_ref_p)
    println("Max Diff: ", maximum(diff))
    
    if maximum(diff) > 1e-4
        println("Locations of mismatches:")
        
        y_c1 = y[:, :, :, 1, 1]
        inds = findall(x->x>0.5, y_c1)
        println("Lux Out Ch 1 active: ", inds)
        
        y_ref_c1 = y_ref_p[:, :, :, 1, 1]
        inds_ref = findall(x->x>0.5, y_ref_c1)
        println("Ref Out Ch 1 active: ", inds_ref)
    end
end

debug_asymmetric()
