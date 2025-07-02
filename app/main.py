import os
import uuid
import shutil
import zipfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

from app.dicom2nifti_utils import convert_dicom_to_nifti_slices
from app.nnunet_runner import run_nnunet_inference_on_slice

app = FastAPI()

UPLOADS_ROOT = "uploads"

@app.post("/upload/")
async def upload_and_segment(file: UploadFile = File(...)):
    upload_id = str(uuid.uuid4())
    upload_folder = os.path.join(UPLOADS_ROOT, upload_id)
    os.makedirs(upload_folder, exist_ok=True)
    zip_path = os.path.join(upload_folder, "dicoms.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(upload_folder)
    os.remove(zip_path)

    single_slice_folder = os.path.join(upload_folder, "slices")
    try:
        niftis = convert_dicom_to_nifti_slices(upload_folder, single_slice_folder)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"DICOM to NIfTI failed: {e}"})

    output_slices_folder = os.path.join(upload_folder, "nnunet_out")
    os.makedirs(output_slices_folder, exist_ok=True)
    outputs = []

    for nifti_path in niftis:
        nnunet_input_dir = os.path.join(upload_folder, "nnunet_input", os.path.basename(nifti_path)[:-7])
        os.makedirs(nnunet_input_dir, exist_ok=True)
        nnunet_case = os.path.join(nnunet_input_dir, "case_0000.nii.gz")
        shutil.copy(nifti_path, nnunet_case)
        try:
            seg_path = run_nnunet_inference_on_slice(nnunet_input_dir)
            output_path = os.path.join(output_slices_folder, os.path.basename(seg_path))
            shutil.move(seg_path, output_path)
            outputs.append(output_path)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"nnUNet inference failed on {nifti_path}: {e}"})

    output_zip = os.path.join(upload_folder, "all_segmentations.zip")
    with zipfile.ZipFile(output_zip, "w") as zipf:
        for out_file in outputs:
            zipf.write(out_file, os.path.basename(out_file))

    return FileResponse(output_zip, media_type="application/zip", filename="all_segmentations.zip")

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
