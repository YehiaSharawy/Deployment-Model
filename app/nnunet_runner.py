import os
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

NNUNET_RAW = "/Users/yehiasharawy/Documents/GitHub/Deployment-Model/nnUNet_raw"
NNUNET_PREPROCESSED = "/Users/yehiasharawy/Documents/GitHub/Deployment-Model/nnUNet_preprocessed"
NNUNET_RESULTS = "/Users/yehiasharawy/Documents/GitHub/Deployment-Model/nnUNet_results"

def run_nnunet_inference_on_slice(input_dir, output_dir=None):
    """
    Run nnUNetV2 inference on a single NIfTI slice directory.
    Returns path to the segmentation file.
    """
    os.environ['nnUNet_raw'] = NNUNET_RAW
    os.environ['nnUNet_preprocessed'] = NNUNET_PREPROCESSED
    os.environ['nnUNet_results'] = NNUNET_RESULTS

    if output_dir is None:
        output_dir = input_dir

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,  # Should be True for CUDA, but False for CPU!
        device=torch.device('cpu'),  # Change to cuda if available and desired
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=f"{NNUNET_RESULTS}/Dataset004_COCA2Dv3/nnUNetTrainer__nnUNetPlans__2d",
        use_folds=(1,),
        checkpoint_name="checkpoint_best.pth"
    )

    predictor.predict_from_files(
        input_dir,
        output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )

    seg_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith("_seg.nii.gz")
    ]
    if not seg_files:
        raise FileNotFoundError(f"Segmentation output not found in {output_dir}")
    return seg_files[0]
