import torch
import torch.nn as nn

class OODDetector(nn.Module):
    """
    Stage 3: OOD Detection.
    A simple VAE trained on the latent vectors (z_p) to learn the 'normal' manifold.
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

    def get_ood_score(self, x):
        """
        Returns reconstruction error as OOD score.
        Higher score = More likely OOD.
        """
        recon, _, _ = self(x)
        error = torch.mean((x - recon)**2, dim=1)
        return error

def train_ood_detector(latent_vectors):
    """
    latent_vectors: Tensor of shape (N, 16) extracted from Causal VAE.
    """
    model = OODDetector(input_dim=latent_vectors.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        optimizer.zero_grad()
        recon, mu, logvar = model(latent_vectors)
        loss = nn.MSELoss()(recon, latent_vectors)
        loss.backward()
        optimizer.step()

    return model
