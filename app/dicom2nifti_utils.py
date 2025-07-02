import SimpleITK as sitk

def convert_dicom_to_nifti(dicom_folder, output_nifti):
    """Convert DICOM folder to NIfTI using SimpleITK."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    if not dicom_names:
        raise ValueError("No DICOM files found in folder: " + dicom_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, output_nifti)
    return output_nifti
