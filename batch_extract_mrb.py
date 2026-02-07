import os
import subprocess
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================
SLICER_PATH = "/home/jm/Slicer-5.11.0-2025-11-10-linux-amd64/Slicer"
DATA_DIR = "./data"
LOGIC_SCRIPT = os.path.abspath("slicer_extract_logic.py")

# =============================================================================
# RUNNER
# =============================================================================

def run_extraction():
    # Find all .mrb files in the data directory
    mrb_pattern = os.path.join(DATA_DIR, "Pat*", "verification_scene.mrb")
    mrb_files = glob.glob(mrb_pattern)
    
    print(f"Found {len(mrb_files)} MRB files to process.")
    
    for i, mrb_path in enumerate(mrb_files):
        mrb_abs = os.path.abspath(mrb_path)
        output_dir = os.path.dirname(mrb_abs)
        case_name = os.path.basename(output_dir)
        
        print(f"\n[{i+1}/{len(mrb_files)}] Processing {case_name}...")
        
        # Prepare environment for the Slicer subprocess
        env = os.environ.copy()
        env["MRB_INPUT"] = mrb_abs
        env["MRB_OUTPUT"] = output_dir
        
        # Build command
        # Note: On headless Linux, you might need xvfb-run if Slicer crashes
        cmd = [
            SLICER_PATH,
            "--no-main-window",
            "--no-splash",
            "--python-script", LOGIC_SCRIPT
        ]
        
        try:
            # Run Slicer for this single case
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"DONE: {case_name}")
            else:
                print(f"ERROR: Slicer exited with code {result.returncode} for {case_name}")
                print(result.stderr)
        except Exception as e:
            print(f"EXCEPTION: Failed to run Slicer for {case_name}: {e}")

if __name__ == "__main__":
    if not os.path.exists(SLICER_PATH):
        print(f"ERROR: Slicer executable not found at {SLICER_PATH}")
    else:
        run_extraction()
        print("\nAll conversions finished.")
