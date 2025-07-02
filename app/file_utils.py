import shutil
import os

def prepare_nnunet_input(folder, nifti_file_path):
    """Rename/copy the NIfTI file for nnUNet single channel (_0000.nii.gz)."""
    nnunet_input_path = os.path.join(folder, "patient_0000.nii.gz")
    shutil.copy(nifti_file_path, nnunet_input_path)
    return nnunet_input_path

def save_uploaded_file(upload_file, destination):
    """Save an UploadFile object to disk."""
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
