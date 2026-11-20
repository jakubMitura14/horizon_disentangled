#!/usr/bin/env python3
"""
Upload newly downloaded models (Hulu-Med-32B, STU-Net, VoCo missing files)
to XNAT HORIZON_CHECKPOINTS after the main checkpoint upload finishes.

This script:
1. Waits for the HF downloads to complete (checks file presence)
2. Uploads each model to XNAT with auto-split for files >3 GB
3. Updates the manifest

Usage:
    python scripts/upload_new_models.py [--dry-run]

Environment:
    LD_LIBRARY_PATH=/DATA/conda/envs/nukl-to-ct/lib:$LD_LIBRARY_PATH
    /DATA/conda/envs/nukl-to-ct/bin/python scripts/upload_new_models.py
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import xnat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
log = logging.getLogger(__name__)

# ── XNAT config ───────────────────────────────────────────────────────────────
ALIAS    = "ee21adf2-44a6-4a97-a53c-34c889273e90"
SECRET   = "dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0"
SERVER   = "https://imaging-platform.diz-ag.med.ovgu.de"
PROJECT  = "temp"
RES_ID   = "40121"

BASE      = Path("/DATA/horizon_disentangled")
SPLIT_MB  = 3072   # 3 GB split threshold in MiB
SPLIT_DIR = BASE / ".xnat_splits"

# ── New model files to upload ─────────────────────────────────────────────────
# Hulu-Med-32B: 13 weight shards + all configs
HULU_BASE = BASE / "models/Hulu-Med-32B"
HULU_WEIGHT_FILES = [
    (HULU_BASE / f"model-{i:05d}-of-00013.safetensors",
     f"Hulu-Med-32B/model-{i:05d}-of-00013.safetensors")
    for i in range(1, 14)
]
HULU_CONFIG_FILES = [
    (HULU_BASE / "config.json",                        "Hulu-Med-32B/config.json"),
    (HULU_BASE / "configuration.json",                 "Hulu-Med-32B/configuration.json"),
    (HULU_BASE / "configuration_hulumed_encoder.py",   "Hulu-Med-32B/configuration_hulumed_encoder.py"),
    (HULU_BASE / "configuration_hulumed_qwen2.py",     "Hulu-Med-32B/configuration_hulumed_qwen2.py"),
    (HULU_BASE / "generation_config.json",             "Hulu-Med-32B/generation_config.json"),
    (HULU_BASE / "image_processing_hulumed.py",        "Hulu-Med-32B/image_processing_hulumed.py"),
    (HULU_BASE / "modeling_hulumed_encoder.py",        "Hulu-Med-32B/modeling_hulumed_encoder.py"),
    (HULU_BASE / "modeling_hulumed_qwen2.py",          "Hulu-Med-32B/modeling_hulumed_qwen2.py"),
    (HULU_BASE / "model.safetensors.index.json",       "Hulu-Med-32B/model.safetensors.index.json"),
    (HULU_BASE / "preprocessor_config.json",           "Hulu-Med-32B/preprocessor_config.json"),
    (HULU_BASE / "processing_hulumed.py",              "Hulu-Med-32B/processing_hulumed.py"),
    (HULU_BASE / "processor_config.json",              "Hulu-Med-32B/processor_config.json"),
    (HULU_BASE / "tokenizer_config.json",              "Hulu-Med-32B/tokenizer_config.json"),
    (HULU_BASE / "special_tokens_map.json",            "Hulu-Med-32B/special_tokens_map.json"),
    (HULU_BASE / "added_tokens.json",                  "Hulu-Med-32B/added_tokens.json"),
    (HULU_BASE / "chat_template.json",                 "Hulu-Med-32B/chat_template.json"),
    (HULU_BASE / "vocab.json",                         "Hulu-Med-32B/vocab.json"),
    (HULU_BASE / "merges.txt",                         "Hulu-Med-32B/merges.txt"),
]

# STU-Net: 4 .model files (NNUNet format, ~15.8 GB total)
STU_BASE = BASE / "models/STU-Net"
STU_FILES = [
    (STU_BASE / "small_ep4k.model",  "STU-Net/small_ep4k.model"),
    (STU_BASE / "large_ep4k.model",  "STU-Net/large_ep4k.model"),
    (STU_BASE / "base_ep4k.model",   "STU-Net/base_ep4k.model"),
    (STU_BASE / "huge_ep4k.model",   "STU-Net/huge_ep4k.model"),
]

# VoCo — missing files (H SSL head + others not yet in XNAT)
VOCO_BASE = BASE / "models/VoCo"
VOCO_NEW_FILES = [
    (VOCO_BASE / "VoCo_H_SSL_head.pt",   "VoCo/VoCo_H_SSL_head.pt"),
    (VOCO_BASE / "VoCo_L_SSL_head.pt",   "VoCo/VoCo_L_SSL_head.pt"),
    (VOCO_BASE / "VoComni_B.pt",         "VoCo/VoComni_B.pt"),
    (VOCO_BASE / "VoComni_nnunet.pt",    "VoCo/VoComni_nnunet.pt"),
]

ALL_NEW_FILES = HULU_WEIGHT_FILES + HULU_CONFIG_FILES + STU_FILES + VOCO_NEW_FILES


def split_file(src: Path, remote_name: str) -> list[tuple[Path, str]]:
    """Split a file into 3 GB binary parts."""
    stem = src.name
    out_dir = SPLIT_DIR / remote_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / (stem + ".part")
    result = subprocess.run(
        ["split", "-b", f"{SPLIT_MB}m", "--numeric-suffixes=0", "--suffix-length=2",
         str(src), str(prefix)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"split failed: {result.stderr}")
    parts = sorted(out_dir.glob(stem + ".part*"))
    log.info(f"  Split into {len(parts)} parts: {[p.name for p in parts]}")
    return [(p, remote_name + f".part{i:02d}") for i, p in enumerate(parts)]


def upload_file(res, local_path: Path, remote_name: str, existing: set, dry_run: bool) -> dict:
    """Upload a single file, splitting if >3 GB."""
    sz = local_path.stat().st_size
    sz_mb = sz / 1024**2
    needs_split = sz_mb > SPLIT_MB

    if needs_split:
        parts = split_file(local_path, remote_name)
        entries = []
        for part_path, part_remote in parts:
            part_sz = part_path.stat().st_size
            part_name = Path(part_remote).name
            if part_name in existing:
                log.info(f"    ↷ Skip {part_name} (already on XNAT)")
            elif not dry_run:
                log.info(f"    ↑ Uploading part {part_name} ({part_sz/1e6:.0f} MB)...")
                t0 = time.time()
                res.upload(str(part_path), remotepath=part_name, overwrite=True)
                dt = time.time() - t0
                log.info(f"      ✓ {part_name} in {dt:.1f}s ({part_sz/1e6/max(dt,0.001):.0f} MB/s)")
            else:
                log.info(f"    [DRY] Would split+upload {part_name} ({part_sz/1e6:.0f} MB)")
            entries.append({"filename": part_name, "size_bytes": part_sz, "part_of": remote_name})
        return {
            "remote_name": remote_name, "local_path": str(local_path),
            "size_bytes": sz, "split": True, "parts": len(parts),
            "reconstruct_cmd": f"cat {Path(remote_name).name}.part* > {Path(remote_name).name}"
        }
    else:
        fname = Path(remote_name).name
        if fname in existing:
            log.info(f"  ↷ Skip {fname} (already on XNAT)")
        elif not dry_run:
            log.info(f"  ↑ Uploading {fname} ({sz_mb:.0f} MB)...")
            t0 = time.time()
            res.upload(str(local_path), remotepath=fname, overwrite=True)
            dt = time.time() - t0
            log.info(f"    ✓ {fname} in {dt:.1f}s ({sz_mb/max(dt,0.001):.0f} MB/s)")
        else:
            log.info(f"  [DRY] Would upload {fname} ({sz_mb:.0f} MB)")
        return {"remote_name": remote_name, "local_path": str(local_path),
                "size_bytes": sz, "split": False}


def wait_for_downloads(files: list, timeout_minutes: int = 180) -> list:
    """Wait for files to appear on disk. Returns list of available files."""
    deadline = time.time() + timeout_minutes * 60
    available = []
    pending = list(files)

    while pending and time.time() < deadline:
        still_pending = []
        for lp, rn in pending:
            if lp.exists() and lp.stat().st_size > 0:
                available.append((lp, rn))
                log.info(f"  ✓ Ready: {lp.name} ({lp.stat().st_size/1e9:.2f} GB)")
            else:
                still_pending.append((lp, rn))
        pending = still_pending
        if pending:
            log.info(f"  Waiting for {len(pending)} file(s): {[p.name for p,_ in pending[:3]]}...")
            time.sleep(60)

    if pending:
        log.warning(f"  TIMEOUT: {len(pending)} file(s) never appeared: {[p.name for p,_ in pending]}")
    return available


def main():
    parser = argparse.ArgumentParser(description="Upload new models to XNAT after HF download")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true",
                        help="Don't wait for downloads — only upload already-present files")
    args = parser.parse_args()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which files are ready now
    ready_now = [(lp, rn) for lp, rn in ALL_NEW_FILES if lp.exists()]
    not_ready = [(lp, rn) for lp, rn in ALL_NEW_FILES if not lp.exists()]

    log.info("=" * 70)
    log.info(f"NEW MODEL UPLOAD {'[DRY RUN] ' if args.dry_run else ''}")
    log.info(f"  Total files : {len(ALL_NEW_FILES)}")
    log.info(f"  Ready now   : {len(ready_now)}")
    log.info(f"  Downloading : {len(not_ready)}")
    log.info("=" * 70)

    if not_ready and not args.no_wait:
        log.info(f"Waiting for {len(not_ready)} files to finish downloading...")
        extra = wait_for_downloads(not_ready)
        all_files = ready_now + extra
    else:
        if not_ready:
            log.warning(f"--no-wait: skipping {len(not_ready)} not-yet-downloaded files")
        all_files = ready_now

    if not all_files:
        log.error("No files ready to upload. Run HF downloads first.")
        return

    total_bytes = sum(lp.stat().st_size for lp, _ in all_files)
    log.info(f"\nUploading {len(all_files)} files ({total_bytes/1e9:.1f} GB) to XNAT...")

    if args.dry_run:
        for lp, rn in all_files:
            sz = lp.stat().st_size
            split = sz / 1024**2 > SPLIT_MB
            log.info(f"  {'[SPLIT] ' if split else '        '}{rn}  ({sz/1e9:.2f} GB)")
        return

    with xnat.connect(server=SERVER, user=ALIAS, password=SECRET) as session:
        res = session.projects[PROJECT].resources[RES_ID]
        log.info("Fetching existing files...")
        existing = {Path(f.name).name for f in res.files.values()}
        log.info(f"Resource currently has {len(existing)} file(s).")

        manifest_entries = []
        for i, (local_path, remote_name) in enumerate(all_files, 1):
            log.info(f"\n[{i}/{len(all_files)}] {remote_name} ({local_path.stat().st_size/1e9:.2f} GB)")
            entry = upload_file(res, local_path, remote_name, existing, args.dry_run)
            manifest_entries.append(entry)

        # Update manifest
        manifest_path = BASE / "experiments" / "new_models_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "resource": "HORIZON_CHECKPOINTS", "resource_id": RES_ID, "project": PROJECT,
                "upload_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "models": ["Hulu-Med-32B", "STU-Net", "VoCo-extended"],
                "files": manifest_entries
            }, f, indent=2)
        log.info(f"\nUploading new_models_manifest.json...")
        res.upload(str(manifest_path), remotepath="new_models_manifest.json", overwrite=True)

    log.info("\n" + "=" * 70)
    log.info("NEW MODELS SUCCESSFULLY SYNCED TO XNAT!")
    log.info(f"  Hulu-Med-32B : {sum(1 for _,rn in all_files if 'Hulu' in rn)} files")
    log.info(f"  STU-Net      : {sum(1 for _,rn in all_files if 'STU' in rn)} files")
    log.info(f"  VoCo (new)   : {sum(1 for _,rn in all_files if 'VoCo' in rn)} files")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
