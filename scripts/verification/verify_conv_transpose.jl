using Lux
using Random
using ComponentArrays
using NNlib

function verify_conv_transpose()
    rng = Random.default_rng()
    
    # ConvTranspose 1D equivalent for simplicity
    # Input: [0, 1, 0]
    # Kernel: [1, 2, 3]
    # Output logic?
    
    # Let's use 3D
    # Input: 3x3x3, center is 1.
    input = zeros(Float32, 3, 3, 3, 1, 1)
    input[2, 2, 2, 1, 1] = 1.0
    
    # Kernel: 3x3x3. Center-Left (1, 2, 2) is 1.0.
    weight = zeros(Float32, 3, 3, 3, 1, 1) # (W, H, D, Out, In) - Out=1, In=1
    weight[1, 2, 2, 1, 1] = 1.0
    
    # Layer: ConvTranspose
    # stride=1, padding=Same?
    layer = ConvTranspose((3,3,3), 1=>1; stride=1, pad=SamePad(), use_bias=false)
    
    ps, st = Lux.setup(rng, layer)
    ps = ComponentArray(ps)
    ps.weight .= weight
    
    output, _ = layer(input, ps, st)
    
    println("Input center: ", input[2, 2, 2, 1, 1])
    println("Kernel peak at: (1, 2, 2)")
    
    pos = findfirst(x -> x > 0.5, output)
    println("Output peak at: ", pos)
    
    # Logic:
    # Transpose Conv "splats" the kernel onto the input.
    # Input[2,2,2] * Kernel is added to Output centered at 2,2,2?
    # If Kernel has peak at (1,2,2) relative to center (2,2,2).
    # Then Output should have peak at (2+ (1-2), ...) = (1, ...)?
    # Or does it use the unflipped kernel?
    
    if pos[1] == 1
        println("Result: Splats weights directly (No Flip relative to placement)")
    elseif pos[1] == 3
        println("Result: Flipped Splat?")
    end
end

verify_conv_transpose()
