import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu_p, log_p, mu_s, log_s, s, mask, beta=1.0):
    """
    Composite Loss for Causal VAE.
    L = Recon + KL(z_p) + KL(z_s) + AnatomyConstraint + Disentangle(not imp here yet)
    """
    # 1. Reconstruction Loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # 2. KL Divergence
    kl_p = -0.5 * torch.sum(1 + log_p - mu_p.pow(2) - log_p.exp())
    kl_s = -0.5 * torch.sum(1 + log_s - mu_s.pow(2) - log_s.exp())

    # 3. Anatomy Constraint (Mask Reconstruction)
    # Ideally we'd have a separate decoder head to reconstruct mask from s
    # For now, we assume s ITSELF should resemble the mask (identity constraint)
    # since SpatialEncoder is shallow.
    # Resize s to match mask if needed
    if s.shape != mask.shape:
        s_resized = F.interpolate(s, size=mask.shape[2:])
    else:
        s_resized = s

    # Simple constraint: s channels should map to mask classes
    # If mask is 1 channel (labels), s is 4 channels.
    # Let's skip explicit anatomy loss for this pilot mock-up unless we add an aux decoder.

    return recon_loss + beta * (kl_p + kl_s)
