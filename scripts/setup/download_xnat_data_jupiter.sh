#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# download_xnat_data_jupiter.sh
# ═══════════════════════════════════════════════════════════════════════════
# Frontend / login node script to download the HORIZON_HDF5 dataset
# (~22 GB) and HORIZON_CHECKPOINTS (163 GB) from XNAT onto JUPITER
# project storage.
#
# Runs on the login node — NO GPU or Slurm allocation needed.
#
# Usage:
#   bash scripts/setup/download_xnat_data_jupiter.sh [--data-only | --checkpoints-only | --all]
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ─── Storage Layout (JUPITER project filesystem) ────────────────────────
DATA_DIR="/e/project1/numedai-scale/mitura1/horizon_data/hdf5"
CKPT_DIR="/e/project1/numedai-scale/mitura1/horizon_data/checkpoints"
LOGS_DIR="/e/scratch/numedai-scale/mitura1/horizon_logs"

# ─── XNAT Credentials ──────────────────────────────────────────────────
CRED_FILE="$REPO_ROOT/scripts/xnat_credentials.txt"
XNAT_SERVER="https://imaging-platform.diz-ag.med.ovgu.de"
XNAT_ALIAS=""
XNAT_SECRET=""

if [ -f "$CRED_FILE" ]; then
    XNAT_ALIAS=$(grep "^alias" "$CRED_FILE" | cut -d'=' -f2 | tr -d ' ')
    XNAT_SECRET=$(grep "^secret" "$CRED_FILE" | cut -d'=' -f2 | tr -d ' ')
fi

if [ -z "$XNAT_ALIAS" ] || [ -z "$XNAT_SECRET" ]; then
    echo "ERROR: Missing XNAT credentials. Create $CRED_FILE with:"
    echo "  server = $XNAT_SERVER"
    echo "  alias = <your_alias>"
    echo "  secret = <your_secret>"
    exit 1
fi

# ─── Parse Arguments ───────────────────────────────────────────────────
MODE="${1:---all}"

echo "═══════════════════════════════════════════════════════════"
echo "  HORIZON — XNAT DATA DOWNLOAD (JUPITER)"
echo "═══════════════════════════════════════════════════════════"
echo "XNAT Server:    $XNAT_SERVER"
echo "Data Target:    $DATA_DIR"
echo "Ckpt Target:    $CKPT_DIR"
echo "Mode:           $MODE"
echo "═══════════════════════════════════════════════════════════"

# ─── Activate Environment ─────────────────────────────────────────────
if [ -f "$REPO_ROOT/activate_jupiter.sh" ]; then
    source "$REPO_ROOT/activate_jupiter.sh"
elif [ -f "/e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh" ]; then
    source /e/project1/numedai-scale/mitura1/venvs/activate_lymph.sh
fi

mkdir -p "$DATA_DIR" "$CKPT_DIR" "$LOGS_DIR"

# ─── Download HDF5 Datasets ───────────────────────────────────────────
if [ "$MODE" = "--all" ] || [ "$MODE" = "--data-only" ]; then
    echo ""
    echo ">>> Downloading HDF5 dataset from XNAT (HORIZON_HDF5)..."
    python3 "$REPO_ROOT/scripts/data/xnat_download.py" \
        --download-dataset \
        --project temp \
        --resource-label HORIZON_HDF5 \
        --output "$DATA_DIR" \
        --credentials "$CRED_FILE" \
        2>&1 | tee "$LOGS_DIR/xnat_hdf5_download.log"
    echo ""
    echo "✔ HDF5 download complete. Contents:"
    ls -lhS "$DATA_DIR"/*.h5 2>/dev/null || echo "  (no .h5 files found)"
    echo "  Total: $(du -sh "$DATA_DIR" | cut -f1)"
fi

# ─── Download Model Checkpoints ───────────────────────────────────────
if [ "$MODE" = "--all" ] || [ "$MODE" = "--checkpoints-only" ]; then
    echo ""
    echo ">>> Downloading model checkpoints from XNAT (HORIZON_CHECKPOINTS)..."

    # Use Python + requests for checkpoint download (xnat library may not be installed)
    python3 - << 'PYEOF' "$CKPT_DIR" "$CRED_FILE"
import sys, os, json, time
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
cred_file = Path(sys.argv[2])

# Parse credentials
creds = {}
for line in cred_file.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        creds[k.strip()] = v.strip()

server = creds.get("server", "https://imaging-platform.diz-ag.med.ovgu.de")
alias = creds["alias"]
secret = creds["secret"]

try:
    import xnat
    print(f"Connecting to {server}...")
    with xnat.connect(server=server, user=alias, password=secret) as session:
        project = session.projects["temp"]
        # Try resource by label
        target_res = None
        for r in project.resources.values():
            if r.label == "HORIZON_CHECKPOINTS":
                target_res = r
                break
        if target_res is None:
            print("WARNING: HORIZON_CHECKPOINTS resource not found. Skipping.")
            sys.exit(0)

        remote_files = list(target_res.files.values())
        print(f"Found {len(remote_files)} file(s) in HORIZON_CHECKPOINTS")

        for i, f in enumerate(remote_files, 1):
            dest = ckpt_dir / f.name
            # Create subdirectory structure if file name contains /
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == f.size:
                print(f"[{i}/{len(remote_files)}] Skipping {f.name} (exists)")
                continue
            print(f"[{i}/{len(remote_files)}] Downloading {f.name} ({f.size/(1024**2):.1f} MB)...")
            t0 = time.time()
            f.download(str(dest))
            print(f"  ✓ Done in {time.time()-t0:.1f}s")

except ImportError:
    import requests
    print("xnat library not available; using requests REST API fallback...")
    session_url = f"{server}/data/JSESSION"
    r = requests.post(session_url, auth=(alias, secret), verify=False)
    if r.status_code != 200:
        print(f"ERROR: XNAT auth failed ({r.status_code})")
        sys.exit(1)
    jsession = r.text.strip()
    cookies = {"JSESSIONID": jsession}

    # List files in HORIZON_CHECKPOINTS resource
    list_url = f"{server}/data/projects/temp/resources/HORIZON_CHECKPOINTS/files"
    resp = requests.get(list_url, cookies=cookies, params={"format": "json"}, verify=False)
    if resp.status_code != 200:
        print(f"WARNING: Could not list HORIZON_CHECKPOINTS ({resp.status_code}). Skipping.")
        sys.exit(0)
    files = resp.json().get("ResultSet", {}).get("Result", [])
    print(f"Found {len(files)} file(s) via REST API")

    for i, finfo in enumerate(files, 1):
        fname = finfo.get("Name", "")
        fsize = int(finfo.get("Size", 0))
        dest = ckpt_dir / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == fsize:
            print(f"[{i}/{len(files)}] Skipping {fname} (exists)")
            continue
        dl_url = f"{server}{finfo['URI']}"
        print(f"[{i}/{len(files)}] Downloading {fname} ({fsize/(1024**2):.1f} MB)...")
        t0 = time.time()
        with requests.get(dl_url, cookies=cookies, stream=True, verify=False) as dr:
            dr.raise_for_status()
            with open(dest, "wb") as out:
                for chunk in dr.iter_content(chunk_size=8*1024*1024):
                    out.write(chunk)
        print(f"  ✓ Done in {time.time()-t0:.1f}s")

    # Logout
    requests.delete(f"{server}/data/JSESSION", cookies=cookies, verify=False)

print("\n✔ Checkpoint download complete.")
PYEOF

    echo ""
    echo "✔ Checkpoint download complete. Contents:"
    ls -lhS "$CKPT_DIR/" 2>/dev/null | head -20
    echo "  Total: $(du -sh "$CKPT_DIR" | cut -f1)"

    # Reconstruct split files
    echo ""
    echo ">>> Reconstructing split .partXX files..."
    cd "$CKPT_DIR"
    for base in $(ls *.part00 2>/dev/null | sed 's/\.part00$//' || true); do
        echo "  Reconstructing $base ..."
        cat ${base}.part* > ${base} && rm -f ${base}.part*
    done
    echo "✔ Split file reconstruction complete."
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✔ ALL DOWNLOADS COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo "HDF5 Data:     $(du -sh "$DATA_DIR" 2>/dev/null | cut -f1) at $DATA_DIR"
echo "Checkpoints:   $(du -sh "$CKPT_DIR" 2>/dev/null | cut -f1) at $CKPT_DIR"
echo "═══════════════════════════════════════════════════════════"
