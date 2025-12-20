import os
import glob
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset, CacheDataset
from monai.losses import DiceCELoss
from monai.utils import set_determinism
import matplotlib.pyplot as plt

from src.data.mock_data import generate_mock_dataset
from src.preprocessing.pipeline import get_preprocessing_pipeline
from src.models.supervisor import SegmentationSupervisor

def main():
    set_determinism(seed=42)
    data_dir = "src/mock_data"

    # 1. Generate Mock Data if not exists
    if not os.path.exists(data_dir):
        print("Generating mock data...")
        generate_mock_dataset(data_dir, num_patients=10)

    # 2. Create Data List
    data_dicts = []
    patient_dirs = glob.glob(os.path.join(data_dir, "*"))
    for p_dir in patient_dirs:
        p_id = os.path.basename(p_dir)
        data_dicts.append({
            "t2w": os.path.join(p_dir, f"{p_id}_t2w.nii.gz"),
            "adc": os.path.join(p_dir, f"{p_id}_adc.nii.gz"),
            "seg": os.path.join(p_dir, f"{p_id}_seg.nii.gz"),
        })

    # 3. Setup Transforms and Dataset
    transforms = get_preprocessing_pipeline(roi_size=(96, 96, 32))
    # Using CacheDataset for faster training
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 4. Initialize Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SegmentationSupervisor(in_channels=2, out_channels=2).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training Loop (Short run for verification)
    max_epochs = 5
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in dataloader:
            step += 1
            inputs = torch.cat([batch_data["t2w"], batch_data["adc"]], dim=1).to(device)
            labels = batch_data["seg"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"{step}/{len(dataset) // dataloader.batch_size}, train_loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} average loss: {epoch_loss / step:.4f}")

    print("Training finished successfully.")

    # Save a checkpoint
    torch.save(model.state_dict(), "src/supervisor_checkpoint.pth")
    print("Checkpoint saved.")

if __name__ == "__main__":
    main()
