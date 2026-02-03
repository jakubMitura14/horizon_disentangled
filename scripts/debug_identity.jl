using Lux
using SafeTensors
using NNlib

function debug_identity()
    tensors = SafeTensors.load_safetensors("verification_data/identity_debug.safetensors")
    x = reshape(tensors["input"], reverse(size(tensors["input"])))
    y_ref = reshape(tensors["output"], reverse(size(tensors["output"])))
    w = reshape(tensors["weight"], reverse(size(tensors["weight"]))) # (3,3,3,1,1)
    
    # Print where the 1.0 is in weights
    cart_inds = findall(x -> x > 0.5, w)
    println("Weight 1.0 index in Lux (Reshape): ", cart_inds)
    
    # Expected: Center of 3x3x3 is (2,2,2)
    
    layer = Conv((3,3,3), 1=>1, stride=1, pad=1, use_bias=false)
    ps = (weight=w,)
    y, _ = layer(x, ps, (;))
    
    diff = abs.(y .- y_ref)
    println("Max Diff: ", maximum(diff))
    
    # Find active output pixel
    out_inds = findall(x -> x > 0.5, y)
    println("Output active pixel: ", out_inds)
    
    ref_inds = findall(x -> x > 0.5, y_ref)
    println("Ref active pixel: ", ref_inds)
end

debug_identity()
