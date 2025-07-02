import os
import uuid
import shutil
import zipfile
import pydicom
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import HTMLResponse

from app.dicom2nifti_utils import convert_dicom_to_nifti
from app.nnunet_runner import run_nnunet_inference
from app.file_utils import prepare_nnunet_input, save_uploaded_file



app = FastAPI()

UPLOADS_ROOT = "uploads"

@app.post("/upload/")
async def upload_and_segment(file: UploadFile = File(...)):
    upload_id = str(uuid.uuid4())
    upload_folder = os.path.join(UPLOADS_ROOT, upload_id)
    os.makedirs(upload_folder, exist_ok=True)
    zip_path = os.path.join(upload_folder, "dicoms.zip")
    save_uploaded_file(file, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(upload_folder)
    os.remove(zip_path)

    dicom_folder = upload_folder
    output_nifti = os.path.join(upload_folder, "output.nii.gz")

    try:
        convert_dicom_to_nifti(dicom_folder, output_nifti)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"DICOM to NIfTI failed: {e}"})

    # --- HERE: prepare input folder for nnUNet ---
    nnunet_input_dir = prepare_nnunet_input(upload_folder, output_nifti)
    # ---------------------------------------------

    try:
        seg_path = run_nnunet_inference(nnunet_input_dir)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"nnUNet inference failed: {e}"})

    return FileResponse(seg_path, media_type="application/gzip", filename=os.path.basename(seg_path))

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload DICOMs for nnUNet Inference</title>
    </head>
    <body>
        <h2>Upload DICOM files (zip file, one patient at a time)</h2>
        <form action="/upload/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".zip" required>
            <input type="submit" value="Upload and Run">
        </form>
    </body>
    </html>
    """

def find_dicom_folder(root_folder):
    # Walk the directory tree, look for first folder with .dcm files (or any file that looks like DICOM)
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            try:
                # Try to open the file as DICOM
                pydicom.dcmread(os.path.join(dirpath, fname), stop_before_pixels=True)
                return dirpath
            except Exception:
                continue
    raise Exception(f"No DICOM files found in folder: {root_folder}")


def prepare_nnunet_input(upload_folder, output_nifti):
    # Make a clean input folder for nnUNet
    nnunet_input_dir = os.path.join(upload_folder, "nnunet_case")
    os.makedirs(nnunet_input_dir, exist_ok=True)

    # nnUNet expects <caseid>_0000.nii.gz, use a generic name
    nnunet_case = os.path.join(nnunet_input_dir, "case_0000.nii.gz")
    shutil.copy(output_nifti, nnunet_case)
    return nnunet_input_dir