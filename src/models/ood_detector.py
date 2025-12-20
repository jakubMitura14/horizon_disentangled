import torch
import torch.nn as nn

class OODDetector(nn.Module):
    """
    Stage 3: OOD Detection.
    A simple VAE trained on the latent vectors (z_p) to learn the 'normal' manifold.
    Uses reconstruction error and distance in latent space.
    """
    def __init__(self, input_dim=16, latent_dim=4):
        super(OODDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12), nn.ReLU(),
            nn.Linear(12, 8), nn.ReLU()
        )
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, 12), nn.ReLU(),
            nn.Linear(12, input_dim)
        )

        # We will store reference latent points for KNN
        self.ref_latents = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def fit_knn(self, x_train):
        """Store training latents for KNN."""
        with torch.no_grad():
             h = self.encoder(x_train)
             mu, _ = self.fc_mu(h), self.fc_logvar(h)
             self.ref_latents = mu # Use mean

    def get_ood_score(self, x, k=5):
        """
        Returns OOD score = Reconstruction Error + alpha * KNN Distance
        """
        recon, mu, _ = self(x)

        # 1. Reconstruction Error
        recon_error = torch.mean((x - recon)**2, dim=1)

        # 2. KNN Distance (Latent Space Distance)
        if self.ref_latents is not None:
            # Compute distance to all ref points
            # x: (B, L), ref: (N, L)
            dists = torch.cdist(mu, self.ref_latents) # (B, N)
            # Get k nearest
            knn_dists, _ = torch.topk(dists, k=k, dim=1, largest=False)
            knn_score = torch.mean(knn_dists, dim=1)
        else:
            knn_score = torch.zeros_like(recon_error)

        return recon_error + 0.1 * knn_score

def train_ood_detector(latent_vectors):
    """
    latent_vectors: Tensor of shape (N, 16) extracted from Causal VAE.
    """
    model = OODDetector(input_dim=latent_vectors.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        optimizer.zero_grad()
        recon, mu, logvar = model(latent_vectors)
        loss = nn.MSELoss()(recon, latent_vectors) + 0.01 * torch.mean(mu**2 + logvar.exp() - 1 - logvar)
        loss.backward()
        optimizer.step()

    # Fit KNN
    model.fit_knn(latent_vectors)

    return model
