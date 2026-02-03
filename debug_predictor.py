import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join, isfile
import os

def debug_predictor():
    trainer_path = "external_sources/weights/moose/Dataset100_Brain_v1/nnUNetTrainer_2000epochs_NoMirroring__nnUNetPlans__3d_fullres"
    os.environ["nnUNet_raw"] = "/tmp/nnunet_raw"
    os.environ["nnUNet_preprocessed"] = "/tmp/nnunet_preprocessed"
    os.environ["nnUNet_results"] = "/tmp/nnunet_results"
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True if torch.cuda.is_available() else False,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    args_kwargs = {}
    checkpoint_path = join(trainer_path, "fold_0", "checkpoint_final.pth")
    folds = [0]
    if not isfile(checkpoint_path):
        checkpoint_path = join(trainer_path, "fold_all", "checkpoint_final.pth")
        if isfile(checkpoint_path):
            folds = ["all"]
        else:
            print(f"Cannot find checkpoint in {trainer_path}")
            return
            
    predictor.initialize_from_trained_model_folder(
        trainer_path,
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )
    
    print("Predictor Attributes:")
    print(dir(predictor))
    
    print("\nPredictor.data_preprocessing Attributes (if exists):")
    # try to access common names
    if hasattr(predictor, 'data_preprocessing'):
        print(dir(predictor.data_preprocessing))
    elif hasattr(predictor, 'preprocessor'):
        print(dir(predictor.preprocessor))
    elif hasattr(predictor, 'preprocessing'):
        print(dir(predictor.preprocessing))
        
    import inspect
    print("\nSignature of get_data_iterator_from_raw_npy_data:")
    print(inspect.signature(predictor.get_data_iterator_from_raw_npy_data))
        
if __name__ == "__main__":
    debug_predictor()
