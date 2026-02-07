using SafeTensors

function debug_strides_jl()
    tensors = SafeTensors.load_safetensors("verification_data/strides.safetensors")
    x = tensors["input"]
    println("Size: ", size(x))
    
    # Linear indices
    inds = findall(v -> abs(v) > 0.5, x)
    linear_inds = [LinearIndices(x)[i] for i in inds]
    println("Linear Indices (1-based): ", linear_inds)
    println("Linear Indices (0-based): ", linear_inds .- 1)
    
    # Check expected 16912, 49680
    expected = [16912, 49680]
    println("Match expected (0-based)? ", sort(linear_inds .- 1) == sort(expected))
end

debug_strides_jl()
