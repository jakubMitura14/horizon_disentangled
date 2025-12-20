import torch
from src.models.ood_detector import train_ood_detector

def main():
    # Mock latent vectors (e.g., 100 samples, 16 dim)
    # In real pipeline, these come from CausalVAE encoder
    z_p = torch.randn(100, 16)

    print("\n--- Training OOD Detector ---")
    ood_model = train_ood_detector(z_p)

    # Test on In-Distribution
    score_in = ood_model.get_ood_score(z_p[:5])
    print(f"ID Scores: {score_in.detach().numpy()}")

    # Test on Out-of-Distribution (Noise)
    z_ood = torch.randn(5, 16) * 5.0 # Large shift
    score_out = ood_model.get_ood_score(z_ood)
    print(f"OOD Scores: {score_out.detach().numpy()}")

    assert score_out.mean() > score_in.mean(), "OOD detection failed to separate noise."
    print("OOD Detection Validated.")

if __name__ == "__main__":
    main()
