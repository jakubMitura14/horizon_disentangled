# Multi-Task Classifier for T-Stage, Gleason, and PSA Prediction
# Uses Lux.jl

using Lux
using Random

include(joinpath(@__DIR__, "ordinal_loss.jl"))
include(joinpath(@__DIR__, "vae.jl")) # Import GaussianEncoder

"""
Multi-Task Classifier combining:
- VAE-based Gaussian Encoder (shared latent space)
- Ordinal heads for T-Stage and Gleason
- Regression head for PSA
"""
struct MultiTaskClassifier{E, S, T, G, P} <: Lux.AbstractLuxContainerLayer{(:encoder, :shared_fc, :t_stage_head, :gleason_head, :psa_head)}
    encoder::E
    shared_fc::S
    t_stage_head::T
    gleason_head::G
    psa_head::P
end

function MultiTaskClassifier(;
    in_channels::Int=1,
    encoder_features::Int=256, # This roughly maps to latent_dim * flattened spatial if not pooled, but GaussianEncoder outputs flat latent_dim
    shared_features::Int=128,
    latent_dim::Int=128 # New arg for VAE latent dimension
)
    # Using GaussianEncoder from vae.jl
    # Note: GaussianEncoder expects (in_channels, latent_dim)
    # and outputs (z, mu, logvar)
    encoder = GaussianEncoder(in_channels, latent_dim)
    
    # Shared FC layer takes the latent vector z
    shared_fc = Chain(
        Dense(latent_dim => shared_features),
        x -> relu.(x),
        Dropout(0.3f0)
    )
    
    t_stage_head = OrdinalHead(shared_features, NUM_T_CLASSES)
    gleason_head = OrdinalHead(shared_features, NUM_GLEASON_CLASSES)
    psa_head = RegressionHead(shared_features)
    
    return MultiTaskClassifier(encoder, shared_fc, t_stage_head, gleason_head, psa_head)
end

function (m::MultiTaskClassifier)(x, ps, st)
    # x: (W, H, D, C, B)
    # Encoder returns ((z, mu, logvar), st_enc)
    (z, mu, logvar), st_enc = m.encoder(x, ps.encoder, st.encoder)
    
    # Use z (sampled latent) for forward pass during training
    # For inference/eval, one might prefer mu, but z is standard for VAE-based tasks
    
    shared, st_shared = m.shared_fc(z, ps.shared_fc, st.shared_fc)
    
    t_logits, st_t = m.t_stage_head(shared, ps.t_stage_head, st.t_stage_head)
    g_logits, st_g = m.gleason_head(shared, ps.gleason_head, st.gleason_head)
    psa_pred, st_psa = m.psa_head(shared, ps.psa_head, st.psa_head)
    
    outputs = (
        t_stage_logits = t_logits,
        gleason_logits = g_logits,
        psa_pred = psa_pred,
        # Optional: return VAE internals if needed for auxiliary losses
        vae_mu = mu,
        vae_logvar = logvar
    )
    
    new_st = (
        encoder = st_enc,
        shared_fc = st_shared,
        t_stage_head = st_t,
        gleason_head = st_g,
        psa_head = st_psa
    )
    
    return outputs, new_st
end

"""
Compute multi-task loss. Returns scalar only for AD compatibility.

Args:
    outputs: NamedTuple from forward pass
    labels: NamedTuple with :T_label, :Gleason_label, :PSA_target
    weights: (w_t, w_g, w_psa) loss weights
    
Returns:
    total_loss (scalar)
"""
function compute_multitask_loss(outputs, labels; weights=(1.0f0, 1.0f0, 0.01f0))
    w_t, w_g, w_psa = weights
    total_loss = 0.0f0
    
    # T-Stage loss (skip invalid labels = 0)
    t_mask = labels.T_label .> 0
    if any(t_mask)
        t_logits = outputs.t_stage_logits[:, t_mask]
        t_labels = labels.T_label[t_mask]
        t_loss = coral_loss(t_logits, t_labels, NUM_T_CLASSES)
        total_loss += w_t * t_loss
    end
    
    # Gleason loss
    g_mask = labels.Gleason_label .> 0
    if any(g_mask)
        g_logits = outputs.gleason_logits[:, g_mask]
        g_labels = labels.Gleason_label[g_mask]
        g_loss = coral_loss(g_logits, g_labels, NUM_GLEASON_CLASSES)
        total_loss += w_g * g_loss
    end
    
    # PSA loss (skip NaN)
    psa_mask = .!isnan.(labels.PSA_target)
    if any(psa_mask)
        # Using simple indexing for 1D arrays
        psa_pred = outputs.psa_pred[psa_mask]
        psa_true = labels.PSA_target[psa_mask]
        psa_loss = mean(abs2, psa_pred .- psa_true)
        total_loss += w_psa * psa_loss
    end
    
    return total_loss, NamedTuple()
end

# Test
if abspath(PROGRAM_FILE) == @__FILE__
    using Random
    
    println("Testing MultiTaskClassifier...")
    
    rng = Random.default_rng()
    model = MultiTaskClassifier(in_channels=1)
    ps, st = Lux.setup(rng, model)
    
    # Test input
    x = randn(Float32, 48, 48, 16, 1, 4)  # Batch of 4
    
    outputs, st_new = model(x, ps, st)
    
    println("Input shape: $(size(x))")
    println("T-Stage logits: $(size(outputs.t_stage_logits))")  # (8, 4)
    println("Gleason logits: $(size(outputs.gleason_logits))")  # (4, 4)
    println("PSA predictions: $(size(outputs.psa_pred))")       # (4,)
    
    # Test loss
    labels = (
        T_label = [3, 5, 0, 7],  # 0 = invalid
        Gleason_label = [2, 3, 4, 0],
        PSA_target = [10.5f0, NaN32, 25.0f0, 8.3f0]
    )
    
    loss, loss_dict = compute_multitask_loss(outputs, labels)
    println("\nTotal Loss: $loss")
    println("Loss Dict: $loss_dict")
    
    println("\n✓ MultiTaskClassifier test passed!")
end
