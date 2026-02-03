using Lux
using Random
using ComponentArrays
using NNlib

function verify_conv()
    rng = Random.default_rng()
    
    # 1D example for simplicity (logic holds for 3D)
    # Input: [0, 1, 2, 3, 0]
    # Kernel: [1, 0, 0] (Left-most is 1)
    
    # Correlation:
    # At pos 1 (center 1): Window [0, 1, 2]. Kern [1, 0, 0]. Sum = 0*1 + 1*0 + 2*0 = 0?
    # Wait, correlation slides without flipping.
    # Kern = [a, b, c]
    # Input = [x1, x2, x3]
    # Out = x1*a + x2*b + x3*c
    
    # Convolution:
    # Flips the kernel.
    # Kern_flipped = [c, b, a]
    # Out = x1*c + x2*b + x3*a
    
    # Let's use 3D to be exact with our use case.
    # Input: 3x3x3 volume. Center is 1, others 0.
    input = zeros(Float32, 3, 3, 3, 1, 1)
    input[2, 2, 2, 1, 1] = 1.0
    
    # Kernel: 3x3x3. Center-Left (1, 2, 2) is 1.0, others 0.
    weight = zeros(Float32, 3, 3, 3, 1, 1)
    weight[1, 2, 2, 1, 1] = 1.0
    
    # Create Conv layer
    # Kernel dims: (3, 3, 3), In: 1, Out: 1
    # Lux weights for Conv are (K1, K2, K3, In, Out)
    
    layer = Conv((3, 3, 3), 1 => 1; pad=SamePad(), use_bias=false)
    ps, st = Lux.setup(rng, layer)
    
    # Manually set weight
    ps = ComponentArray(ps)
    ps.weight .= weight
    
    # Run
    output, _ = layer(input, ps, st)
    
    println("Input center: ", input[2, 2, 2, 1, 1])
    println("Kernel peak at: (1, 2, 2)")
    
    # Analyze output
    # If Correlation:
    # At output (2,2,2): input window is input[1:3, 1:3, 1:3].
    # Element-wise multiply with kernel [1,2,2] is 1.
    # input[1,2,2] * weight[1,2,2] + ...
    # input[1,2,2] is 0.
    # input[2,2,2] is 1. (corresponds to weight[2,2,2] which is 0)
    # So if we slide kernel [1,0; 0,0] over [0,0; 1,0]...
    # When kernel center is at (2,2,2), kernel[1,2,2] overlaps input[1,2,2].
    # When kernel center is at (3,2,2), kernel[1,2,2] overlaps input[2,2,2] (which is 1).
    # So output at (3,2,2) should be 1.
    
    # If Convolution (Flipped):
    # Kernel flipped becomes peak at (3, 2, 2).
    # When kernel center is at (1,2,2), kernel_flipped[3,2,2] overlaps input[2,2,2].
    # So output at (1,2,2) should be 1.
    
    pos = findfirst(x -> x > 0.5, output)
    println("Output peak at: ", pos)
    
    if pos[1] == 3
        println("Result: Correlation (Matches PyTorch default)")
    elseif pos[1] == 1
        println("Result: Convolution (Mathematically flipped)")
    else
        println("Result: Confusing")
    end
end

verify_conv()
