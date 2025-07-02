import os
import SimpleITK as sitk

def convert_dicom_to_nifti_slices(dicom_folder, output_folder):
    """
    Converts a DICOM series to multiple NIfTI single-slice files.
    Returns a list of output NIfTI slice paths.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    if not dicom_names:
        raise ValueError("No DICOM files found in folder: " + dicom_folder)

    reader.SetFileNames(dicom_names)
    image3d = reader.Execute()
    n_slices = image3d.GetDepth()
    out_paths = []
    for i in range(n_slices):
        img_slice = image3d[:, :, i]
        out_path = os.path.join(output_folder, f"slice_{i:03d}.nii.gz")
        sitk.WriteImage(img_slice, out_path)
        out_paths.append(out_path)
    return out_paths
