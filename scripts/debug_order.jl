using SafeTensors

function debug_order()
    tensors = SafeTensors.load_safetensors("verification_data/order.safetensors")
    t = tensors["tensor"]
    println("Type: ", typeof(t))
    println("Loaded size: ", size(t))
    println("Loaded values: ", t)
    # Expected: [0.0 1.0 2.0 3.0]
    
    println("t[1]: ", t[1])
    # output of t[2] depends on whether it's (1,4) or (4,1) loaded
    if length(t) > 1
        println("t[2]: ", t[2])
    end
    
    # Check reshape
    t_r = reshape(t, reverse(size(t)))
    println("Reshaped size: ", size(t_r))
    println("Reshaped values: ", t_r)
    println("t_r[1]: ", t_r[1])
    println("t_r[2]: ", t_r[2])
end

debug_order()
