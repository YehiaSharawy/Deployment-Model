import dicom2nifti

dicom_folder = "patient/IM-6033-0001.dcm"  # Use the same folder you uploaded
output_folder = "tmp/dicom2nifti_test"

import os
os.makedirs(output_folder, exist_ok=True)

try:
    dicom2nifti.convert_directory(dicom_folder, output_folder)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
