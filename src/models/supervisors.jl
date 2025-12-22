using Lux
using Random

"""
    UnetSupervisor(in_channels=2, out_channels=2)

Constructs a U-Net style segmentation model.
- `in_channels`: Number of input image channels (e.g., T2W + ADC = 2).
- `out_channels`: Number of output classes (e.g., Background, Prostate, Tumor).
"""
function UnetSupervisor(in_channels=2, out_channels=2)
    # Simple 3D U-Net equivalent
    return Chain(
        Conv((3,3,3), in_channels => 16, pad=1, relu),
        Conv((3,3,3), 16 => 32, pad=1, relu),
        Conv((3,3,3), 32 => out_channels, pad=1)
        # In real scenario, use skip connections and upsampling
    )
end

"""
    OrdinalSupervisor(in_channels=2, num_classes=5)

Constructs an ordinal regression model for grading (e.g., Gleason score).
- `num_classes`: Total ordinal ranks. Output size is `num_classes - 1` (binary cutpoints).
"""
function OrdinalSupervisor(in_channels=2, num_classes=5)
    return Chain(
        Conv((3,3,3), in_channels => 16, pad=1, relu),
        GlobalMaxPool(),
        FlattenLayer(),
        Dense(16 => num_classes - 1)
    )
end

"""
    SurvivalSupervisor(in_channels=2)

Constructs a DeepSurv-like survival risk predictor.
Outputs a single scalar risk score.
"""
function SurvivalSupervisor(in_channels=2)
    return Chain(
        Conv((3,3,3), in_channels => 16, pad=1, relu),
        GlobalMaxPool(),
        FlattenLayer(),
        Dense(16 => 1)
    )
end
