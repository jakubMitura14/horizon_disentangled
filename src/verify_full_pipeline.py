import os
import shutil
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"Running {script_path}...")
    print(f"{'='*50}")
    # Run with PYTHONPATH=. to ensure src imports work from root
    ret = os.system(f"PYTHONPATH=. python3 {script_path}")
    if ret != 0:
        print(f"FAILED: {script_path}")
        sys.exit(1)
    else:
        print(f"SUCCESS: {script_path}")

def main():
    print("Starting Full Pipeline Verification on Synthetic Data...")

    # 1. Cleanup old artifacts
    if os.path.exists("src/mock_data"):
        shutil.rmtree("src/mock_data")

    # 2. Stage 1: Supervisors (Segmentation, Ordinal, Survival)
    # This script handles Data Gen internally if missing, but we can rely on it.
    run_script("src/train_supervisor.py")

    # 3. Stage 2: Causal VAE (Disentanglement)
    run_script("src/train_vae.py")

    # 4. Stage 3: OOD Detection
    run_script("src/test_ood.py")

    # 5. Stage 4: Neural Jump ODE (Temporal)
    run_script("src/train_njde.py")

    # 6. Validation (Counterfactual)
    run_script("src/validate_counterfactual.py")

    print("\n\nAll Stages Verified Successfully!")

    # Cleanup
    print("Cleaning up artifacts...")
    if os.path.exists("src/mock_data"):
        shutil.rmtree("src/mock_data")

    checkpoints = ["src/supervisor_checkpoint.pth", "src/vae_checkpoint.pth", "src/njde_checkpoint.pth", "src/counterfactual_plot.png"]
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            os.remove(ckpt)

    print("Done.")

if __name__ == "__main__":
    main()
