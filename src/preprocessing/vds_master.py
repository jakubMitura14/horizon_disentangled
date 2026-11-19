import os
import glob
from datetime import datetime
import numpy as np
import h5py
from typing import Dict, Any, List, Tuple
from src.preprocessing.ledger import IngestionLedger

PATIENT_DTYPE = np.dtype([
    ('uid', 'S32'),
    ('domain', 'S8'),
    ('cohort', 'S24'),
    ('split', np.uint8),      # 0: train, 1: val, 2: test
    ('os_weeks', np.float32), # overall survival in weeks (or NaN)
    ('event', np.uint8),      # 0: censored, 1: event/death/progression, 255: missing
    ('pcr', np.int8),         # -1: N/A, 0: non-pCR, 1: pCR
    ('num_visits', np.uint8),
    ('target_shard', 'S32'),  # e.g. "breast_ispy2_s02.h5"
])

def assign_stratified_splits(records: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """
    Assigns stratified 70% train / 15% val / 15% test splits per cohort,
    preserving benchmark splits if already assigned (e.g. BreastDCEDL Duke & SPY1).
    """
    rng = np.random.RandomState(seed)

    # Group by (cohort, target_label) for unassigned records
    groups = {}
    for idx, r in enumerate(records):
        if r['split'] is not None:
            continue
        c = r['cohort']
        if r['domain'] == 'breast':
            target = r['pcr'] if r['pcr'] in (0, 1) else -1
        else:
            target = r['event'] if r['event'] in (0, 1) else -1
        key = (c, target)
        if key not in groups:
            groups[key] = []
        groups[key].append(idx)

    for key, idxs in groups.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(np.round(0.70 * n))
        n_val = int(np.round(0.15 * n))
        # Ensure at least 1 per set if n >= 3
        if n >= 3 and n_train == n:
            n_train = n - 2
            n_val = 1
        elif n == 2:
            n_train = 1
            n_val = 1

        train_idx = idxs[:n_train]
        val_idx = idxs[n_train:n_train + n_val]
        test_idx = idxs[n_train + n_val:]

        for i in train_idx:
            records[i]['split'] = 0
        for i in val_idx:
            records[i]['split'] = 1
        for i in test_idx:
            records[i]['split'] = 2

    # Catch any remaining
    for r in records:
        if r['split'] is None:
            r['split'] = 0
    return records

def build_master_vds(hdf5_dir: str = "/DATA/horizon_disentangled/hdf5",
                     vds_filename: str = "horizon_master.vds.h5"):
    vds_path = os.path.join(hdf5_dir, vds_filename)
    shard_files = [
        "gbm_tabular_cohorts.h5",
        "breast_mincrop.h5",
        "gbm_lumiere.h5",
    ] + [f"breast_ispy2_s0{i}.h5" for i in range(8)]

    records = []
    print("Scanning HDF5 shards to assemble master patient registry...")

    for sf in shard_files:
        full_sf = os.path.join(hdf5_dir, sf)
        if not os.path.exists(full_sf):
            print(f"Warning: shard {sf} not found on disk, skipping.")
            continue

        with h5py.File(full_sf, 'r') as h5f:
            if 'patients' not in h5f:
                continue
            for uid in sorted(h5f['patients'].keys()):
                p_grp = h5f[f'patients/{uid}']
                sc = p_grp.get('static_clinical')
                attrs = dict(sc.attrs) if sc is not None else {}

                # Decode domain
                raw_dom = attrs.get('domain', b'gbm' if 'GBM' in sf or 'gbm' in sf or 'lumiere' in sf else b'breast')
                domain = raw_dom.decode('ascii') if isinstance(raw_dom, bytes) else str(raw_dom)

                # Decode cohort
                raw_coh = attrs.get('cohort', sf.replace('.h5', ''))
                cohort = raw_coh.decode('ascii') if isinstance(raw_coh, bytes) else str(raw_coh)

                # Split (preserve if in attrs)
                split = int(attrs['split']) if 'split' in attrs else None

                # Outcomes
                pcr = int(attrs['pcr']) if 'pcr' in attrs and attrs['pcr'] is not None else -1
                event = int(attrs['event']) if 'event' in attrs and attrs['event'] in (0, 1) else 255

                os_w = float(attrs.get('os_weeks', attrs.get('survival_weeks', np.nan)))

                # Num visits
                v_grp = p_grp.get('visits')
                num_v = len(v_grp) if v_grp is not None else 0

                records.append({
                    'uid': uid,
                    'domain': domain,
                    'cohort': cohort,
                    'split': split,
                    'os_weeks': os_w,
                    'event': event,
                    'pcr': pcr,
                    'num_visits': num_v,
                    'target_shard': sf
                })

    print(f"Discovered {len(records)} total patients across all shards.")

    # Assign stratified splits for any unassigned patients
    records = assign_stratified_splits(records)

    # Build numpy compound array
    reg_array = np.empty(len(records), dtype=PATIENT_DTYPE)
    total_visits = 0

    for i, r in enumerate(records):
        reg_array[i]['uid'] = r['uid'].encode('ascii')
        reg_array[i]['domain'] = r['domain'].encode('ascii')
        reg_array[i]['cohort'] = r['cohort'].encode('ascii')
        reg_array[i]['split'] = r['split']
        reg_array[i]['os_weeks'] = r['os_weeks']
        reg_array[i]['event'] = r['event']
        reg_array[i]['pcr'] = r['pcr']
        reg_array[i]['num_visits'] = r['num_visits']
        reg_array[i]['target_shard'] = r['target_shard'].encode('ascii')
        total_visits += r['num_visits']

    # Create VDS file
    if os.path.exists(vds_path):
        os.remove(vds_path)

    print(f"Creating master VDS file at {vds_path}...")
    with h5py.File(vds_path, 'w', libver='latest') as vds:
        # Root attributes
        vds.attrs['license'] = "CC-BY-NC-4.0"
        vds.attrs['total_patients'] = len(records)
        vds.attrs['total_visits'] = total_visits
        vds.attrs['domains'] = np.array([b"gbm", b"breast"], dtype='S8')
        vds.attrs['created_at'] = datetime.utcnow().isoformat() + "Z"

        # Registry group & dataset
        reg_grp = vds.create_group("registry")
        reg_dset = reg_grp.create_dataset("patients", data=reg_array)

        # Transparent patient external links
        pat_grp = vds.create_group("patients")
        for r in records:
            pat_grp[r['uid']] = h5py.ExternalLink(r['target_shard'], f"patients/{r['uid']}")

        # Cohort external links
        coh_grp = vds.create_group("cohorts")
        for sf in shard_files:
            cname = sf.replace('.h5', '')
            coh_grp[cname] = h5py.ExternalLink(sf, "/")

    print(f"Master VDS successfully built! Total patients: {len(records)}, Total visits: {total_visits}")
    return vds_path

if __name__ == "__main__":
    build_master_vds()
