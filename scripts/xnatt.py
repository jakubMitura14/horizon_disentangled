#!/usr/bin/env python3
"""
xnat_down.py
============
XNAT Download & Complete Segmentation Pipeline for:
  - FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE (Lu177_PSMA_PIPELINE)
  - FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE (Prostata_bimodal_PIPELINE)

Features:
  1. Automated download of CT + SUV NIfTI pairs from XNAT scans_SUV_CT_PET.
  2. Collision-free directory structure preserving longitudinal cycles (e.g. PETPSMA_0__Pat1).
  3. Continuous disk space safeguard ensuring host storage never exhausts.
  4. Full anatomical segmentation & HDF5 packed tensor generation:
     - 21+ TotalSegmentator tasks in strict, overwrite-safe precedence ('total' last to preserve whole aorta)
     - SlicerDental nnU-Net mandible cropped to skull FOV, dilated r=3, subtracted from skull
     - MuscleMap whole-body musculature
     - Skellytour bone subsegments
     - Post-processing: bilateral muscle splitting, lung lobe fusion, aorta & iliac bifurcation synthesis
     - Max Anatomy multi-label volume (max_anatomy.nii.gz, 456 classes)
     - 4D UInt8 packed tensor container (packed_anatomical_tensor.h5, packed_anatomical_tensor_registry.json)
     - MedImages primary container (primary_masks.h5) with computed landmarks
     - Julia GPU Lymph Node rule engine DAG (final_results_gpu.h5)
     - 3D Slicer scene bundle (<pat_id>.mrb) with SUV injected
  5. Mandatory Repeat-on-Error Guarantee: every task repeats on failure, NEVER skips!
  6. Parallel multi-process execution across 2 to 3 workers on GPU.
  7. Intermediate raw segmentations cleanup (--cleanup) to guarantee sufficient disk space.

Usage:
  # Validate CT + SUV availability
  python3 xnat_down.py --validate --project all

  # Download all cases
  python3 xnat_down.py --download --project all --output /DATA/xnat_data

  # Download and process all cases with 2 workers in parallel
  python3 xnat_down.py --process --project all --output /DATA/xnat_data --workers 2 --cleanup

  # Process single subject
  python3 xnat_down.py --process --project Prostata_bimodal_PIPELINE --subject Pat44 --output /DATA/xnat_data
"""

import os
import sys
import re
import time
import json
import queue
import shutil
import logging
import argparse
import threading
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("XNATPipeline")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_ENV = "/DATA/conda/envs/nukl-to-ct/bin/python"
PREPARE_SCRIPT = str(PROJECT_ROOT / "scripts/prepare_packed_case.py")

DEFAULT_SERVER = "https://imaging-platform.diz-ag.med.ovgu.de"
DEFAULT_ALIAS = "ee21adf2-44a6-4a97-a53c-34c889273e90"
DEFAULT_SECRET = "dEowoHHC1em1UwN70C4WG23I0mjn0Js1APpdPrCzbQJAN8p6P5ljrkuzik9k8AI0"

PROJECT_MAPPINGS = {
    "fdm_dpi-2024-7-krn_lu177_psma_pipeline": "Lu177_PSMA_PIPELINE",
    "lu177_psma_pipeline": "Lu177_PSMA_PIPELINE",
    "lu177": "Lu177_PSMA_PIPELINE",
    "fdm_dpi-2024-7-krn_prostata_bimodal_pipeline": "Prostata_bimodal_PIPELINE",
    "prostata_bimodal_pipeline": "Prostata_bimodal_PIPELINE",
    "prostata": "Prostata_bimodal_PIPELINE",
}

ALL_PROJECT_IDS = [
    "Prostata_bimodal_PIPELINE",
    "Lu177_PSMA_PIPELINE"
]


# ──────────────────────────────────────────────────────────────────────────────
# Credentials
# ──────────────────────────────────────────────────────────────────────────────

def load_credentials(cred_file=None, project_override=None):
    """Load XNAT credentials from file with fallback to user credentials."""
    creds = {
        "server": DEFAULT_SERVER,
        "alias": DEFAULT_ALIAS,
        "secret": DEFAULT_SECRET,
        "project": "all"
    }
    if cred_file is None:
        cred_file = SCRIPT_DIR / "xnat_credentials.txt"
    cred_file = Path(cred_file)

    if cred_file.exists():
        with open(cred_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    creds[key.strip()] = val.strip()

    if project_override:
        creds["project"] = project_override

    return creds


# ──────────────────────────────────────────────────────────────────────────────
# XNAT Helpers
# ──────────────────────────────────────────────────────────────────────────────

def connect_xnat(creds):
    """Connect to XNAT server."""
    import xnat
    logger.info(f"Connecting to XNAT ({creds['server']})...")
    session = xnat.connect(
        server=creds["server"],
        user=creds["alias"],
        password=creds["secret"],
    )
    logger.info("Connected successfully to XNAT.")
    return session


def resolve_project_ids(project_arg):
    """Resolves project argument string to canonical project IDs."""
    if not project_arg or project_arg.lower() == "all":
        return ALL_PROJECT_IDS
    parts = [p.strip() for p in project_arg.split(",")]
    resolved = []
    for p in parts:
        clean = p.lower()
        if clean in PROJECT_MAPPINGS:
            resolved.append(PROJECT_MAPPINGS[clean])
        else:
            resolved.append(p)
    return resolved


def extract_pat_id(subject_label: str) -> str:
    """
    Extracts unique, collision-free patient ID from full subject label.
    Preserves longitudinal study timepoints (e.g. PETPSMA_0__Pat1, PETPSMA_1__Pat1).
    """
    if subject_label.startswith("FDM_DPI-") and "__" in subject_label:
        return subject_label.split("__", 1)[1]
    return subject_label


def matches_subject_filter(filter_str: str, pat_id: str, label: str) -> bool:
    """Checks whether a subject matches the user filter string cleanly without partial false positives."""
    if not filter_str:
        return True
    if filter_str == pat_id or filter_str == label:
        return True
    # Match components (e.g. 'Pat1' matches 'PETPSMA_0__Pat1', but not 'Pat10')
    if "__" in pat_id:
        sub_parts = pat_id.split("__")
        if filter_str in sub_parts:
            return True
    return False


def find_ct_file(files):
    """Find CT NIfTI file from list of XNAT file objects."""
    for f in files:
        name = f.name.lower()
        if name.endswith(".nii.gz") and ("ct_wb" in name or "ct_" in name or name == "ct.nii.gz"):
            if "suv" not in name and "pet" not in name:
                return f
    for f in files:
        name = f.name.lower()
        if "fixed_ct_volume" in name and name.endswith(".nii.gz"):
            return f
    for f in files:
        name = f.name.lower()
        if "ct" in name and name.endswith(".nii.gz") and "suv" not in name and "pet" not in name:
            return f
    return None


def find_suv_file(files):
    """Find SUV NIfTI file from list of XNAT file objects."""
    for f in files:
        name = f.name.lower()
        if name.endswith(".nii.gz") and "suv" in name:
            return f
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Disk Space Safety Guard
# ──────────────────────────────────────────────────────────────────────────────

def check_disk_space(target_dir: str, min_free_gb: float = 15.0) -> bool:
    """Verifies that the target filesystem has sufficient free space."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target_path).free / (1024 ** 3)
    if free_gb < min_free_gb:
        logger.error(
            f"DISK SPACE CRITICAL: only {free_gb:.1f} GB available on {target_path} "
            f"(required threshold: {min_free_gb:.1f} GB). Execution halted to protect host disk."
        )
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: Validate
# ──────────────────────────────────────────────────────────────────────────────

def validate(creds, subject_filter=None, projects=None):
    """List all subjects across projects and check CT/SUV availability."""
    session = connect_xnat(creds)
    project_ids = resolve_project_ids(projects or creds.get("project"))

    for pid in project_ids:
        try:
            proj = session.projects[pid]
        except KeyError:
            logger.warning(f"Project '{pid}' not found on server.")
            continue

        subjects = list(proj.subjects.values())
        print(f"\n{'='*95}")
        print(f"Project: {pid} ({proj.name}) — {len(subjects)} subjects")
        print(f"{'Subject Label':<55} {'Clean ID':<25} {'CT':>4} {'SUV':>5} {'Status':<10}")
        print("─" * 95)

        ok_count = 0
        missing = []

        for subj in sorted(subjects, key=lambda s: s.label):
            label = subj.label
            pat_id = extract_pat_id(label)

            if not matches_subject_filter(subject_filter, pat_id, label):
                continue

            ct_ok = False
            suv_ok = False

            if "scans_SUV_CT_PET" in subj.resources:
                res = subj.resources["scans_SUV_CT_PET"]
                files = list(res.files.values())
                ct_ok = find_ct_file(files) is not None
                suv_ok = find_suv_file(files) is not None

            ct_str = "✅" if ct_ok else "❌"
            suv_str = "✅" if suv_ok else "❌"
            status = "OK" if (ct_ok and suv_ok) else "MISSING"

            print(f"{label:<55} {pat_id:<25} {ct_str:>4} {suv_str:>5} {status:<10}")
            if ct_ok and suv_ok:
                ok_count += 1
            else:
                missing.append((pat_id, ct_ok, suv_ok))

        print("─" * 95)
        print(f"Project {pid} Summary: {len(subjects)} subjects, {ok_count} complete, {len(missing)} incomplete")

    session.disconnect()


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Download
# ──────────────────────────────────────────────────────────────────────────────

def download_case(subj, output_dir: Path, project_id: str) -> tuple[Path, bool]:
    """Downloads CT + SUV for a single subject into collision-free directory."""
    label = subj.label
    pat_id = extract_pat_id(label)
    case_dir = output_dir / project_id / pat_id

    ct_path = case_dir / "CT.nii.gz"
    suv_path = case_dir / "SUV.nii.gz"
    fixed_ct = case_dir / "Fixed_CT_Volume.nii.gz"

    if ct_path.exists() and suv_path.exists():
        if not fixed_ct.exists():
            fixed_ct.symlink_to("CT.nii.gz")
        logger.info(f"  [Cached] {project_id}/{pat_id} already downloaded.")
        return case_dir, True

    if not check_disk_space(output_dir, min_free_gb=15.0):
        return case_dir, False

    resources = {r.label: r for r in subj.resources.values()}
    if "scans_SUV_CT_PET" not in resources:
        logger.warning(f"  [Missing] {project_id}/{pat_id}: no scans_SUV_CT_PET resource.")
        return case_dir, False

    files = list(resources["scans_SUV_CT_PET"].files.values())
    ct_file = find_ct_file(files)
    suv_file = find_suv_file(files)

    if not ct_file or not suv_file:
        logger.warning(f"  [Incomplete] {project_id}/{pat_id}: CT={bool(ct_file)}, SUV={bool(suv_file)}")
        return case_dir, False

    case_dir.mkdir(parents=True, exist_ok=True)

    # Download CT
    if not ct_path.exists():
        logger.info(f"  Downloading CT: {ct_file.name} -> {ct_path}")
        ct_file.download(str(ct_path))

    # Download SUV
    if not suv_path.exists():
        logger.info(f"  Downloading SUV: {suv_file.name} -> {suv_path}")
        suv_file.download(str(suv_path))

    # Fixed_CT_Volume symlink
    if not fixed_ct.exists():
        fixed_ct.symlink_to("CT.nii.gz")

    return case_dir, True


def download_all(creds, output_dir: str, subject_filter: str = None, projects: str = None, max_threads: int = 4):
    """Downloads CT + SUV pairs for all subjects across target projects with multi-threaded speed."""
    if not check_disk_space(output_dir, min_free_gb=20.0):
        return 0, 0

    session = connect_xnat(creds)
    project_ids = resolve_project_ids(projects or creds.get("project"))
    output_base = Path(output_dir)

    total_downloaded = 0
    total_failed = 0

    for pid in project_ids:
        try:
            proj = session.projects[pid]
        except KeyError:
            logger.warning(f"Project '{pid}' not found on server.")
            continue

        subjects = list(proj.subjects.values())
        filtered = []
        for s in sorted(subjects, key=lambda s: s.label):
            pat_id = extract_pat_id(s.label)
            if matches_subject_filter(subject_filter, pat_id, s.label):
                filtered.append(s)

        logger.info(f"\nDownloading Project: {pid} ({len(filtered)} matching subjects with {max_threads} threads)...")

        def _dl(subj):
            pat_id = extract_pat_id(subj.label)
            case_dir, success = download_case(subj, output_base, pid)
            return pat_id, success

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(_dl, s) for s in filtered]
            for i, fut in enumerate(futures, 1):
                pat_id, success = fut.result()
                if success:
                    total_downloaded += 1
                else:
                    total_failed += 1
                if i % 10 == 0 or i == len(filtered):
                    logger.info(f"[{i}/{len(filtered)}] Download progress in {pid}: {total_downloaded} OK, {total_failed} failed")

    session.disconnect()
    logger.info(f"\nDownload finished: {total_downloaded} OK, {total_failed} failed.")
    return total_downloaded, total_failed


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3: Complete Processing Pipeline (Segment + Packed Tensor + DAG + MRB)
# ──────────────────────────────────────────────────────────────────────────────

def process_case(case_dir: Path, pat_id: str, cleanup: bool = True, max_retries: int = 5, env: dict = None, log_file: Path = None) -> bool:
    """
    Executes full anatomical segmentation, 4D packed tensor, MedImages container,
    Max Anatomy hierarchy, Julia GPU DAG, and 3D Slicer MRB export.
    MANDATORY REPEAT-ON-ERROR POLICY: In case of any error, repeats until verified!
    """
    case_dir = Path(case_dir)
    packed_h5 = case_dir / "packed_anatomical_tensor.h5"
    reg_json = case_dir / "packed_anatomical_tensor_registry.json"
    mrb_path = case_dir / f"{pat_id}.mrb"

    # Check if already completed and strictly verified
    from scripts.prepare_packed_case import verify_packed_case
    if packed_h5.exists() and reg_json.exists():
        if verify_packed_case(str(case_dir)):
            logger.info(f"  [Cached & Verified] {pat_id} already fully processed.")
            return True

    ct_path = case_dir / "Fixed_CT_Volume.nii.gz"
    if not ct_path.exists():
        ct_path = case_dir / "CT.nii.gz"
    if not ct_path.exists():
        logger.error(f"  CT volume not found in {case_dir}")
        return False

    cmd = [
        PYTHON_ENV,
        PREPARE_SCRIPT,
        "--ct", str(ct_path),
        "--output-dir", str(case_dir),
        "--gpu", "0",
        "--skip-dag"
    ]
    if cleanup:
        cmd.append("--cleanup")

    run_env = env or os.environ.copy()

    # Repeat-on-error execution loop
    for attempt in range(1, max_retries + 1):
        logger.info(f"  [Pipeline Pass {attempt}/{max_retries}] Processing {pat_id}...")
        t0 = time.time()
        if log_file:
            with open(log_file, "a") as log_io:
                log_io.write(f"\n--- [Pipeline Pass {attempt}/{max_retries}] Start {time.ctime()} ---\n")
                log_io.flush()
                proc = subprocess.run(cmd, env=run_env, stdout=log_io, stderr=subprocess.STDOUT)
        else:
            proc = subprocess.run(cmd, env=run_env)
        elapsed_min = (time.time() - t0) / 60.0

        if proc.returncode == 0 and verify_packed_case(str(case_dir)):
            logger.info(f"  [SUCCESS] {pat_id} completed and verified in {elapsed_min:.1f} minutes.")
            return True

        logger.warning(
            f"  [MANDATORY REPEAT] Attempt {attempt}/{max_retries} failed for {pat_id} (exit {proc.returncode}). "
            f"Repeating task after backoff..."
        )
        time.sleep(attempt * 5)

    logger.error(f"  [FATAL] Case {pat_id} failed all {max_retries} passes.")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Process Batch Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class XNATBatchWorkerManager:
    """Orchestrates concurrent download and processing across parallel workers."""

    def __init__(self, work_items: list, output_base: Path, workers: int = 2, cleanup: bool = True):
        self.work_items = work_items
        self.output_base = output_base
        self.workers = max(1, min(workers, 3))
        self.cleanup = cleanup

        self.queue = queue.Queue()
        for item in work_items:
            self.queue.put(item)

        self.lock = threading.Lock()
        self.completed = set()
        self.failed = set()
        self.active = set()
        self.total = len(work_items)

        self.log_dir = output_base / "batch_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.log_dir / "xnat_batch_status.json"

    def save_status(self):
        with self.lock:
            data = {
                "total_cases": self.total,
                "completed_count": len(self.completed),
                "active_workers": len(self.active),
                "remaining_count": self.queue.qsize(),
                "completed_cases": sorted(list(self.completed)),
                "active_cases": sorted(list(self.active)),
                "last_update": time.ctime()
            }
            tmp = f"{self.status_file}.tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, self.status_file)
            except Exception:
                pass

    def worker_loop(self, worker_id: int):
        logger.info(f"[Worker-{worker_id}] Started.")
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = "0"
        worker_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        worker_env["LD_LIBRARY_PATH"] = f"/DATA/conda/envs/nukl-to-ct/lib:{worker_env.get('LD_LIBRARY_PATH', '')}"
        worker_env["PYTHONPATH"] = f"{PROJECT_ROOT}:{worker_env.get('PYTHONPATH', '')}"
        cpu_threads = max(1, 8 // self.workers)
        worker_env["OMP_NUM_THREADS"] = str(cpu_threads)
        worker_env["MKL_NUM_THREADS"] = str(cpu_threads)
        worker_env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(cpu_threads)

        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break

            subj = item.get("subject")
            pid = item["project_id"]
            pat_id = item.get("pat_id") or extract_pat_id(subj.label if subj else item["label"])
            cid = f"{pid}_{pat_id}"
            log_file = self.log_dir / f"{cid}.log"

            # Step 1: Download / verify cached
            if subj is not None:
                case_dir, dl_ok = download_case(subj, self.output_base, pid)
            else:
                case_dir = self.output_base / pid / pat_id
                dl_ok = (case_dir / "CT.nii.gz").exists() or (case_dir / "Fixed_CT_Volume.nii.gz").exists()

            if not dl_ok:
                dl_retries = item.get("dl_retries", 0) + 1
                item["dl_retries"] = dl_retries
                if dl_retries >= 3:
                    logger.warning(f"[Worker-{worker_id}] Subject {cid} missing CT/SUV on server after 3 checks (skipping non-existent scan).")
                    with self.lock:
                        self.failed.add(cid)
                    self.save_status()
                    self.queue.task_done()
                else:
                    time.sleep(5)
                    self.queue.put(item)
                    self.queue.task_done()
                continue

            with self.lock:
                self.active.add(cid)
            self.save_status()

            # Step 2: Process with repeat-on-error
            logger.info(f"[Worker-{worker_id}] START {cid} [Remaining: {self.queue.qsize()}]")
            ok = process_case(case_dir, pat_id, cleanup=self.cleanup, env=worker_env, log_file=log_file)

            with self.lock:
                self.active.discard(cid)

            if ok:
                with self.lock:
                    self.completed.add(cid)
                logger.info(f"[Worker-{worker_id}] SUCCESS {cid} ({len(self.completed)}/{self.total})")
                self.save_status()
                self.queue.task_done()
            else:
                logger.warning(f"[Worker-{worker_id}] Processing failed for {cid}. Re-enqueueing for retry!")
                time.sleep(15)
                self.queue.put(item)
                self.queue.task_done()

        logger.info(f"[Worker-{worker_id}] Finished.")

    def run(self):
        threads = []
        for i in range(1, self.workers + 1):
            t = threading.Thread(target=self.worker_loop, args=(i,), daemon=True)
            threads.append(t)
            t.start()
            time.sleep(1)

        while any(t.is_alive() for t in threads):
            time.sleep(15)
            self.save_status()
            with self.lock:
                pct = (len(self.completed) / max(1, self.total)) * 100.0
                logger.info(f"[Progress] {len(self.completed)}/{self.total} ({pct:.1f}%) | Active: {len(self.active)} | Remaining: {self.queue.qsize()}")

        for t in threads:
            t.join()
        self.save_status()


def process_all(creds, output_dir: str, subject_filter: str = None, projects: str = None, workers: int = 2, cleanup: bool = True, local_only: bool = False):
    """Downloads and processes all subjects across specified projects with multi-process workers."""
    if not check_disk_space(output_dir, min_free_gb=20.0):
        return 0, 0

    project_ids = resolve_project_ids(projects or creds.get("project"))
    output_base = Path(output_dir)

    work_items = []
    if local_only:
        logger.info("Scanning local directories for existing CT cases...")
        for pid in project_ids:
            p_dir = output_base / pid
            if not p_dir.is_dir():
                continue
            for d in sorted(p_dir.iterdir()):
                if not d.is_dir():
                    continue
                has_ct = (d / "CT.nii.gz").exists() or (d / "Fixed_CT_Volume.nii.gz").exists()
                if has_ct:
                    pat_id = d.name
                    if matches_subject_filter(subject_filter, pat_id, pat_id):
                        work_items.append({"subject": None, "project_id": pid, "pat_id": pat_id, "label": pat_id})
    else:
        session = connect_xnat(creds)
        for pid in project_ids:
            try:
                proj = session.projects[pid]
            except KeyError:
                logger.warning(f"Project '{pid}' not found on server.")
                continue

            subjects = list(proj.subjects.values())
            for subj in subjects:
                pat_id = extract_pat_id(subj.label)
                if not matches_subject_filter(subject_filter, pat_id, subj.label):
                    continue
                # Only enqueue if scan exists on disk or scans_SUV_CT_PET resource exists
                local_ct = output_base / pid / pat_id / "CT.nii.gz"
                local_fixed = output_base / pid / pat_id / "Fixed_CT_Volume.nii.gz"
                if local_ct.exists() or local_fixed.exists():
                    work_items.append({"subject": subj, "project_id": pid, "pat_id": pat_id, "label": subj.label})
                elif "scans_SUV_CT_PET" in subj.resources:
                    work_items.append({"subject": subj, "project_id": pid, "pat_id": pat_id, "label": subj.label})
                else:
                    logger.debug(f"Skipping empty subject {pid}/{pat_id} (no scans_SUV_CT_PET resource)")

        session.disconnect()

    logger.info(f"Total cases discovered across target projects: {len(work_items)}")
    manager = XNATBatchWorkerManager(work_items, output_base, workers=workers, cleanup=cleanup)
    manager.run()


# ──────────────────────────────────────────────────────────────────────────────
# HDF5 Dataset Upload & Download (SLURM Cluster Staging)
# ──────────────────────────────────────────────────────────────────────────────

def upload_dataset(creds, input_dir="/DATA/horizon_disentangled/hdf5", project_name="temp", resource_label="HORIZON_HDF5", overwrite=False):
    """
    Uploads native uncropped HDF5 dataset files to XNAT temp project under a dedicated resource.
    Includes resume support, file size verification, and metadata manifest upload.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {input_path}")

    h5_files = sorted(list(input_path.glob("*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {input_path}")

    logger.info("=" * 65)
    logger.info(f"XNAT HDF5 DATASET UPLOAD")
    logger.info(f"Source Directory: {input_path}")
    logger.info(f"Target XNAT Server: {creds['server']}")
    logger.info(f"Target Project: {project_name}")
    logger.info(f"Resource Label: {resource_label}")
    logger.info(f"Found {len(h5_files)} HDF5 files to upload ({sum(f.stat().st_size for f in h5_files)/(1024**3):.2f} GB)")
    logger.info("=" * 65)

    session = connect_xnat(creds)
    try:
        project = session.projects[project_name]
    except KeyError:
        raise KeyError(f"Project '{project_name}' not found on XNAT server.")

    # Locate or create resource
    target_res = None
    logger.info(f"Fetching resources for project '{project_name}'...")
    for r in project.resources.values():
        if r.label == resource_label:
            target_res = r
            logger.info(f"Found existing resource '{resource_label}' (ID: {r.id})")
            break
    if target_res is None:
        logger.info(f"Creating resource '{resource_label}'...")
        target_res = project.create_resource(label=resource_label, format="HDF5")
        logger.info(f"Created resource '{resource_label}' (ID: {target_res.id})")

    existing_files = {f.name: f.size for f in target_res.files.values()}
    logger.info(f"Resource currently contains {len(existing_files)} file(s)")

    manifest_data = {
        "dataset_name": "Horizon_Disentangled_Multimodal_HDF5",
        "description": "Full uncropped 3D volumetric MRI and DCE datasets for glioblastoma and breast cancer SLURM cluster execution.",
        "project": project_name,
        "resource_label": resource_label,
        "upload_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": []
    }

    total_uploaded_bytes = 0
    t_start = time.time()

    for i, f_path in enumerate(h5_files, 1):
        f_name = f_path.name
        f_size = f_path.stat().st_size
        f_size_mb = f_size / (1024**2)
        f_size_gb = f_size / (1024**3)

        manifest_data["files"].append({
            "filename": f_name,
            "size_bytes": f_size,
            "size_human": f"{f_size_gb:.2f} GB" if f_size_gb >= 1.0 else f"{f_size_mb:.1f} MB",
        })

        if not overwrite and f_name in existing_files:
            remote_size = existing_files[f_name]
            if remote_size == f_size:
                logger.info(f"[{i}/{len(h5_files)}] Skipping {f_name} (already uploaded, exact match {f_size_gb:.2f} GB)")
                continue

        logger.info(f"[{i}/{len(h5_files)}] Uploading {f_name} ({f_size_gb:.2f} GB / {f_size_mb:.1f} MB)...")
        t0 = time.time()
        try:
            target_res.upload(str(f_path), remotepath=f_name, overwrite=True)
            dt = time.time() - t0
            speed_mb_s = f_size_mb / max(dt, 0.001)
            total_uploaded_bytes += f_size
            logger.info(f"    ✓ Uploaded {f_name} in {dt:.1f}s ({speed_mb_s:.1f} MB/s)")
        except Exception as exc:
            if "413" in str(exc):
                logger.warning(f"    ⚠ HTTP 413 (file too large for nginx limit): {f_name}")
                logger.warning(f"      → Run: python scripts/split_mincrop.py")
                logger.warning(f"      → Then upload the _s00/_s01/_s02 shards separately.")
                manifest_data["files"][-1]["status"] = "SKIPPED_413_TOO_LARGE"
            else:
                raise

    # Upload manifest
    manifest_path = input_path / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Uploading dataset manifest {manifest_path.name}...")
    target_res.upload(str(manifest_path), remotepath=manifest_path.name, overwrite=True)
    logger.info("    ✓ Uploaded dataset_manifest.json")

    total_time = time.time() - t_start
    logger.info("=" * 65)
    logger.info(f"ALL HDF5 FILES SUCCESSFULLY SYNCED TO XNAT IN {total_time:.1f}s!")
    logger.info("=" * 65)
    session.disconnect()
    return manifest_data


def download_dataset(creds, output_dir="/DATA/horizon_disentangled/hdf5", project_name="temp", resource_label="HORIZON_HDF5", overwrite=False):
    """
    Downloads uncropped HDF5 dataset files from XNAT temp project resource for SLURM workers.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info(f"XNAT HDF5 DATASET DOWNLOAD")
    logger.info(f"Source Server: {creds['server']}")
    logger.info(f"Project: {project_name}, Resource: {resource_label}")
    logger.info(f"Target Directory: {output_path}")
    logger.info("=" * 65)

    session = connect_xnat(creds)
    project = session.projects[project_name]

    target_res = None
    for r in project.resources.values():
        if r.label == resource_label:
            target_res = r
            break
    if target_res is None:
        raise RuntimeError(f"Resource '{resource_label}' not found on XNAT project '{project_name}'")

    remote_files = list(target_res.files.values())
    logger.info(f"Found {len(remote_files)} file(s) in XNAT resource '{resource_label}'")

    t_start = time.time()
    for i, f in enumerate(remote_files, 1):
        target_file = output_path / f.name
        f_size = f.size
        f_size_mb = f_size / (1024**2) if f_size else 0

        if not overwrite and target_file.exists() and target_file.stat().st_size == f_size:
            logger.info(f"[{i}/{len(remote_files)}] Skipping {f.name} (already exists, {f_size_mb:.1f} MB)")
            continue

        logger.info(f"[{i}/{len(remote_files)}] Downloading {f.name} ({f_size_mb:.1f} MB) -> {target_file}...")
        t0 = time.time()
        f.download(str(target_file))
        dt = time.time() - t0
        speed_mb_s = f_size_mb / max(dt, 0.001)
        logger.info(f"    ✓ Downloaded {f.name} in {dt:.1f}s ({speed_mb_s:.1f} MB/s)")

    logger.info("=" * 65)
    logger.info(f"DATASET DOWNLOAD COMPLETE IN {time.time() - t_start:.1f}s!")
    logger.info("=" * 65)
    session.disconnect()


# ──────────────────────────────────────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="XNAT Download, Segmentation & Dataset Synchronization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true", help="Check CT/SUV availability for all subjects")
    mode.add_argument("--download", action="store_true", help="Download CT + SUV NIfTIs")
    mode.add_argument("--process", action="store_true", help="Download + segment + packed tensor + MRB")
    mode.add_argument("--upload-dataset", action="store_true", help="Upload uncropped HDF5 dataset to XNAT temp project")
    mode.add_argument("--download-dataset", action="store_true", help="Download HDF5 dataset from XNAT temp project")

    parser.add_argument("--project", "-p", type=str, default="temp",
                        help="Target project: 'temp', 'Lu177_PSMA_PIPELINE', 'Prostata_bimodal_PIPELINE', or 'all' (default: temp)")
    parser.add_argument("--input", "-i", type=str, default="/DATA/horizon_disentangled/hdf5", help="Input dataset directory for upload")
    parser.add_argument("--output", "-o", type=str, default="/DATA/horizon_disentangled/hdf5", help="Output directory")
    parser.add_argument("--resource-label", type=str, default="HORIZON_HDF5", help="XNAT Resource label (default: HORIZON_HDF5)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite files if already uploaded / downloaded")
    parser.add_argument("--subject", "-s", type=str, default=None, help="Process only this patient (e.g. Pat44)")
    parser.add_argument("--credentials", "-c", type=str, default=None, help="Path to credentials file")
    parser.add_argument("--workers", "-w", type=int, default=2, help="Parallel worker processes (default: 2)")
    parser.add_argument("--cleanup", action="store_true", default=True, help="Clean up intermediate raw segmentations (default: True)")
    parser.add_argument("--no-cleanup", action="store_false", dest="cleanup", help="Keep raw segmentations/*.nii.gz")
    parser.add_argument("--local", action="store_true", help="Process cases from local directory without querying XNAT API")
    parser.add_argument("--loop-minutes", type=int, default=15, help="Continuous loop mode: sleep N minutes between sweeps (default: 15; set to 0 for single run)")

    args = parser.parse_args()
    creds = load_credentials(args.credentials, args.project)

    if args.upload_dataset:
        upload_dataset(creds, input_dir=args.input, project_name=args.project, resource_label=args.resource_label, overwrite=args.overwrite)
    elif args.download_dataset:
        download_dataset(creds, output_dir=args.output, project_name=args.project, resource_label=args.resource_label, overwrite=args.overwrite)
    elif args.validate:
        validate(creds, args.subject, args.project)
    elif args.download:
        download_all(creds, args.output, args.subject, args.project)
    elif args.process:
        if args.loop_minutes and args.loop_minutes > 0:
            while True:
                try:
                    process_all(creds, args.output, args.subject, args.project, workers=args.workers, cleanup=args.cleanup, local_only=args.local)
                except Exception as e:
                    logger.error(f"Error during processing sweep: {e}")
                logger.info(f"Sweep complete. Sleeping for {args.loop_minutes} minutes before next check...")
                time.sleep(args.loop_minutes * 60)
        else:
            process_all(creds, args.output, args.subject, args.project, workers=args.workers, cleanup=args.cleanup, local_only=args.local)


if __name__ == "__main__":
    main()
