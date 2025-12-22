using Lux
using Random

include("layers.jl")

function UnetSupervisor(in_channels=2, out_channels=2)
    # Simple 3D U-Net equivalent
    return Chain(
        Conv((3,3,3), in_channels => 16, pad=1, relu),
        Conv((3,3,3), 16 => 32, pad=1, relu),
        Conv((3,3,3), 32 => out_channels, pad=1)
        # In real scenario, use skip connections and upsampling
    )
end

function OrdinalSupervisor(in_channels=2, num_classes=5)
    return Chain(
        Conv((3,3,3), in_channels => 16, pad=1, relu),
        GlobalMaxPool(),
        FlattenLayer(),
        Dense(16 => num_classes - 1)
    )
end

function SurvivalSupervisor(in_channels=2)
    return Chain(
        Conv((3,3,3), in_channels => 16, pad=1, relu),
        GlobalMaxPool(),
        FlattenLayer(),
        Dense(16 => 1)
    )
end
