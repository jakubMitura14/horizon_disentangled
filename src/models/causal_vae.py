import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetEncoder(nn.Module):
    """
    3D ResNet-18 Encoder.
    """
    def __init__(self, in_channels, latent_dim):
        super(ResNetEncoder, self).__init__()
        # Simplified ResNet-like block
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.InstanceNorm3d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(32, 64)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, 2, 1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, 1, 1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class AnatomyEncoder(nn.Module):
    """
    Encodes Segmentation Mask into Spatial Tensor 's'.
    Input: (B, 1, H, W, D) Mask
    Output: (B, C, H//8, W//8, D//8) Spatial Tensor
    """
    def __init__(self, in_channels=1, out_channels=4):
        super(AnatomyEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 2, 1), nn.InstanceNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 32, 3, 2, 1), nn.InstanceNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, out_channels, 3, 2, 1), nn.InstanceNorm3d(out_channels), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class SPADE(nn.Module):
    """Spatially-Adaptive Normalization."""
    def __init__(self, norm_nc, label_nc):
        super(SPADE, self).__init__()
        self.param_free_norm = nn.InstanceNorm3d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv3d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv3d(128, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        # Resize segmap to match x spatial dim
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, label_nc):
        super(SPADEResnetBlock, self).__init__()
        self.norm1 = SPADE(fin, label_nc)
        self.conv1 = nn.Conv3d(fin, fout, kernel_size=3, padding=1)
        self.norm2 = SPADE(fout, label_nc)
        self.conv2 = nn.Conv3d(fout, fout, kernel_size=3, padding=1)

        self.learned_shortcut = (fin != fout)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)
            self.conv_s = nn.Conv3d(fin, fout, kernel_size=1, bias=False)

    def forward(self, x, seg):
        dx = self.conv1(F.relu(self.norm1(x, seg)))
        dx = self.conv2(F.relu(self.norm2(dx, seg)))
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s + dx

class Discriminator(nn.Module):
    """
    Adversarial Discriminator: Tries to predict Mask M from Pathology Vector z_p.
    If E_p works well, D should fail (random guess).
    """
    def __init__(self, latent_dim, mask_shape=(1, 96, 96, 32)):
        super(Discriminator, self).__init__()
        self.mask_shape = mask_shape
        # Input: z_p (B, latent_dim)
        # Output: Predicted Mask (B, 1, H, W, D) - Simplified: Predict low-res mask
        self.fc = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 16*6*6*2), # Map to low res volume
            nn.ReLU(),
            nn.Unflatten(1, (16, 6, 6, 2)),
            nn.ConvTranspose3d(16, 8, 4, 2, 1), nn.ReLU(), # 12, 12, 4
            nn.ConvTranspose3d(8, 1, 4, 2, 1) # 24, 24, 8 (Still small, but acts as proxy)
        )
    def forward(self, z):
        h = F.relu(self.fc(z))
        return self.decoder(h)

class CausalVAE(nn.Module):
    """
    Stage 2: Causal Disentangled VAE (SDNet-like).
    """
    def __init__(self, latent_dim=16, anatomy_dim=4):
        super(CausalVAE, self).__init__()

        # Encoders
        self.enc_a = AnatomyEncoder(in_channels=1, out_channels=anatomy_dim) # Input: Mask
        self.enc_p = ResNetEncoder(in_channels=2, latent_dim=latent_dim) # Input: Image
        self.enc_s = ResNetEncoder(in_channels=2, latent_dim=latent_dim) # Input: Image

        # Decoder (Generator)
        # Starts from z_p + z_s (Vector)
        self.fc_start = nn.Linear(latent_dim * 2, 128 * 6 * 6 * 2) # Small base spatial dim

        # SPADE Blocks taking 's' as guidance
        self.spade1 = SPADEResnetBlock(128, 64, label_nc=anatomy_dim)
        self.spade2 = SPADEResnetBlock(64, 32, label_nc=anatomy_dim)
        self.spade3 = SPADEResnetBlock(32, 16, label_nc=anatomy_dim)
        self.conv_out = nn.Conv3d(16, 2, 3, 1, 1) # Output: T2W, ADC

        # Discriminator for Adversarial Loss (z_p independent of anatomy)
        self.discriminator = Discriminator(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img, mask):
        # 1. Anatomy Code (Spatial Tensor) - Driven by Mask
        s = self.enc_a(mask) # (B, 4, H/8, W/8, D/8)

        # 2. Pathology & Style Codes (Vectors) - Driven by Image
        mu_p, log_p = self.enc_p(img)
        z_p = self.reparameterize(mu_p, log_p)

        mu_s, log_s = self.enc_s(img)
        z_s = self.reparameterize(mu_s, log_s)

        # 3. Decode
        z_combined = torch.cat([z_p, z_s], dim=1)
        x = self.fc_start(z_combined).view(-1, 128, 6, 6, 2) # Base spatial

        # Upsample and SPADE
        x = F.interpolate(x, scale_factor=2) # 12, 12, 4
        x = self.spade1(x, s)
        x = F.interpolate(x, scale_factor=2) # 24, 24, 8
        x = self.spade2(x, s)
        x = F.interpolate(x, scale_factor=2) # 48, 48, 16
        x = self.spade3(x, s)

        recon = self.conv_out(x)

        # 4. Adversarial Check
        pred_mask_from_pathology = self.discriminator(z_p)

        return recon, mu_p, log_p, mu_s, log_s, s, pred_mask_from_pathology
