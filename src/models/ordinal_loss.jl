# Ordinal Loss Functions for Julia/Lux.jl
# Implements CORAL (Consistent Rank Logits) for ordinal regression

using Lux
using NNlib
using Statistics
using Zygote

"""
CORAL Loss for Ordinal Regression.

Decomposes K-class ordinal problem into K-1 binary tasks:
"Is the class > k?" for k = 1, 2, ..., K-1

Args:
    logits: (K-1, B) raw scores for each binary threshold
    labels: (B,) integer class labels (1 to K, Julia 1-indexed)
    num_classes: K total classes

Returns:
    Scalar loss value
"""
function coral_loss(logits, labels, num_classes::Int)
    K = num_classes
    
    # Create binary targets (not tracked by AD)
    targets = Zygote.ignore() do
        # labels: (B,)
        # levels: (K-1, 1) to broadcast against (1, B)
        levels = Float32.(collect(1:(K-1)))
        
        # Broadcasting: (K-1) vs (B) -> (K-1, B)
        # We need labels as row vectors: (1, B)
        # labels is Vector, so reshape to (1, B)
        labels_row = reshape(labels, 1, :)
        
        # Compare: 1 if label > k
        # target_cpu is BitMatrix or Matrix{Bool} on CPU
        target_cpu = Float32.(labels_row .> levels)
        
        # Move to same device/type as logits
        return typeof(logits)(target_cpu)
    end
    
    # Binary cross-entropy with logits (differentiable)
    # Both logits and targets are same type (e.g. CuArray)
    loss = mean(NNlib.logitbinarycrossentropy.(logits, targets))
    
    return loss
end

"""
Predict class from CORAL logits.

Args:
    logits: (K-1, B) raw scores
    
Returns:
    (B,) predicted class labels (1 to K)
"""
function coral_predict(logits)
    probs = sigmoid.(logits)  # (K-1, B)
    # Predicted class = 1 + number of thresholds exceeded (prob > 0.5)
    # Sum along first dimension
    preds = 1 .+ sum(probs .> 0.5f0, dims=1)  # (1, B)
    return vec(preds)  # (B,)
end

"""
Ordinal Head for CORAL output.

Maps feature vector to K-1 logits with shared weight and separate biases (cutpoints).
"""
struct OrdinalHead{D} <: Lux.AbstractLuxContainerLayer{(:fc,)}
    fc::D
    num_classes::Int
end

function OrdinalHead(in_features::Int, num_classes::Int)
    # K-1 outputs for K classes
    fc = Dense(in_features => num_classes - 1)
    return OrdinalHead(fc, num_classes)
end

function (m::OrdinalHead)(x, ps, st)
    # x: (in_features, B)
    logits, st_fc = m.fc(x, ps.fc, st.fc)
    return logits, (fc=st_fc,)
end

"""
Regression Head for continuous output (e.g., PSA).
"""
struct RegressionHead{D} <: Lux.AbstractLuxContainerLayer{(:fc,)}
    fc::D
end

function RegressionHead(in_features::Int)
    fc = Dense(in_features => 1)
    return RegressionHead(fc)
end

function (m::RegressionHead)(x, ps, st)
    out, st_fc = m.fc(x, ps.fc, st.fc)
    return vec(out), (fc=st_fc,)  # (B,)
end

# Constants for label encoding
const NUM_T_CLASSES = 9
const NUM_GLEASON_CLASSES = 5

# Label mappings (Julia 1-indexed)
const T_STAGE_MAP = Dict(
    "1a" => 1, "T1a" => 1, "t1a" => 1,
    "1b" => 2, "T1b" => 2, "t1b" => 2,
    "1c" => 3, "T1c" => 3, "t1c" => 3,
    "2a" => 4, "T2a" => 4, "t2a" => 4,
    "2b" => 5, "T2b" => 5, "t2b" => 5,
    "2c" => 6, "T2c" => 6, "t2c" => 6,
    "3a" => 7, "T3a" => 7, "t3a" => 7,
    "3b" => 8, "T3b" => 8, "t3b" => 8,
    "4" => 9, "T4" => 9, "t4" => 9,
)

const GLEASON_MAP = Dict(
    6 => 1, 6.0 => 1,
    7 => 2, 7.0 => 2,
    8 => 3, 8.0 => 3,
    9 => 4, 9.0 => 4,
    10 => 5, 10.0 => 5,
)

function encode_t_stage(val)
    if ismissing(val) || val === nothing
        return nothing
    end
    s = strip(string(val))
    return get(T_STAGE_MAP, s, nothing)
end

function encode_gleason(val)
    if ismissing(val) || val === nothing
        return nothing
    end
    try
        g = parse(Float64, string(val))
        return get(GLEASON_MAP, g, get(GLEASON_MAP, Int(g), nothing))
    catch
        return nothing
    end
end

# Test
if abspath(PROGRAM_FILE) == @__FILE__
    println("Testing CORAL Loss...")
    
    # Simulate 5-class problem (Gleason)
    num_classes = 5
    batch_size = 8
    
    # Random logits (K-1, B)
    logits = randn(Float32, num_classes - 1, batch_size)
    # Random labels 1 to K
    labels = rand(1:num_classes, batch_size)
    
    loss = coral_loss(logits, labels, num_classes)
    preds = coral_predict(logits)
    
    println("Labels: $labels")
    println("Predictions: $preds")
    println("Loss: $loss")
    
    println("\n✓ CORAL Loss test passed!")
end
