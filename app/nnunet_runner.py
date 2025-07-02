import os
import subprocess

NNUNET_RAW = "/Users/yehiasharawy/Desktop/Deployment/nnUNet_raw"
NNUNET_PREPROCESSED = "/Users/yehiasharawy/Desktop/Deployment/nnUNet_preprocessed"
NNUNET_RESULTS = "/Users/yehiasharawy/Desktop/Deployment/nnUNet_results"

def run_nnunet_inference(input_dir):
    """
    Run nnUNetV2 inference on the input_dir.
    Returns path to the segmentation file, or raises exception.
    """
    os.environ['nnUNet_raw'] = NNUNET_RAW
    os.environ['nnUNet_preprocessed'] = NNUNET_PREPROCESSED
    os.environ['nnUNet_results'] = NNUNET_RESULTS

    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", input_dir,
        "-d", "004",
        "-c", "2d",
        "-f", "1",
        "-chk", "checkpoint_best.pth",
        "-device", "cpu"
    ]
    print("[INFO] Running command:", " ".join(cmd))
    subprocess.check_call(cmd)
    for f in os.listdir(input_dir):
        if f.endswith("_seg.nii.gz"):
            return os.path.join(input_dir, f)
    raise FileNotFoundError("Segmentation output not found in " + input_dir)
