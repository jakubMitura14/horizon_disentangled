import os
import glob
import torch
import pandas as pd
from torch.utils.data import DataLoader
from monai.data import Dataset, CacheDataset
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from src.data.mock_data import generate_longitudinal_dataset
from src.preprocessing.pipeline import get_preprocessing_pipeline
from src.models.supervisor import SegmentationSupervisor
from src.models.ordinal import OrdinalSupervisor, ordinal_loss
from src.models.survival import SurvivalSupervisor, cox_loss

def main():
    set_determinism(seed=42)
    data_dir = "src/mock_data"

    # 1. Generate Mock Data if needed
    # We force regeneration to ensure CSV exists matching the new structure
    if os.path.exists(data_dir):
        import shutil
        shutil.rmtree(data_dir)
    generate_longitudinal_dataset(data_dir, num_patients=10)

    # 2. Load Clinical CSV
    df = pd.read_csv(os.path.join(data_dir, "clinical_data.csv"))

    # Create Data List
    data_dicts = []
    for idx, row in df.iterrows():
        data_dicts.append({
            "t2w": row['t2w_path'],
            "adc": row['adc_path'],
            "seg": row['seg_path'],
            "gleason": int(row['gleason']), # Map logic needed later if not 0,6,7,8,9
            "time": float(row['time_to_event']),
            "event": float(row['event_occurred'])
        })

    # 3. Setup Dataset
    transforms = get_preprocessing_pipeline(roi_size=(96, 96, 32))
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.0) # Disable cache for speed in mock
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- A. Train Segmentation Supervisor ---
    print("\n--- Training Segmentation Supervisor ---")
    # Mock data has 3 classes: 0=BG, 1=Prostate, 2=Tumor
    seg_model = SegmentationSupervisor(in_channels=2, out_channels=3).to(device)
    seg_opt = torch.optim.Adam(seg_model.parameters(), lr=1e-3)
    seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    for epoch in range(2):
        seg_model.train()
        for batch in dataloader:
            inputs = torch.cat([batch["t2w"], batch["adc"]], dim=1).to(device)
            labels = batch["seg"].to(device)
            seg_opt.zero_grad()
            out = seg_model(inputs)
            loss = seg_loss_fn(out, labels)
            loss.backward()
            seg_opt.step()
        print(f"Seg Epoch {epoch+1} Loss: {loss.item():.4f}")

    # --- B. Train Ordinal Supervisor ---
    print("\n--- Training Ordinal Supervisor ---")
    # Mapping Gleason: 0->0, 6->1, 7->2, 8->3, 9->4
    gleason_map = {0:0, 6:1, 7:2, 8:3, 9:4}

    ord_model = OrdinalSupervisor(in_channels=2, num_classes=5).to(device)
    ord_opt = torch.optim.Adam(ord_model.parameters(), lr=1e-3)

    for epoch in range(2):
        ord_model.train()
        for batch in dataloader:
            inputs = torch.cat([batch["t2w"], batch["adc"]], dim=1).to(device)
            raw_g = batch["gleason"].tolist()
            targets = torch.tensor([gleason_map.get(g, 0) for g in raw_g]).to(device)

            ord_opt.zero_grad()
            out = ord_model(inputs)
            loss = ordinal_loss(out, targets)
            loss.backward()
            ord_opt.step()
        print(f"Ordinal Epoch {epoch+1} Loss: {loss.item():.4f}")

    # --- C. Train Survival Supervisor ---
    print("\n--- Training Survival Supervisor ---")
    surv_model = SurvivalSupervisor(in_channels=2).to(device)
    surv_opt = torch.optim.Adam(surv_model.parameters(), lr=1e-3)

    for epoch in range(2):
        surv_model.train()
        for batch in dataloader:
            inputs = torch.cat([batch["t2w"], batch["adc"]], dim=1).to(device)
            times = batch["time"].float().to(device)
            events = batch["event"].float().to(device)

            surv_opt.zero_grad()
            risk = surv_model(inputs)
            # Flatten risk
            loss = cox_loss(risk.view(-1), times, events)
            loss.backward()
            surv_opt.step()
        print(f"Survival Epoch {epoch+1} Loss: {loss.item():.4f}")

    print("All Supervisors Trained Successfully.")

if __name__ == "__main__":
    main()
