# app.py

import os
import io
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from predict_knn import load_feature_extractor, DfuRecommender, TREATMENT_MAP, transform

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup
model_path = 'model.pth'
feats_path = 'dfu_feats.npy'
grades_path = 'dfu_grades.npy'
paths_path = 'dfu_paths.npy'

extractor = load_feature_extractor(model_path, device=DEVICE)
recommender = DfuRecommender(feats_path, grades_path, paths_path)

app = FastAPI()

@app.post("/predict")
async def predict_and_generate_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_t = transform(img)

    recs = recommender.recommend(img_t, extractor, k=5, device=DEVICE)

    # Generate PDF
    pdf_path = "treatment_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "DFU Treatment Recommendation Report")
    c.setFont("Helvetica", 12)
    y = height - 80

    for idx, rec in enumerate(recs, 1):
        if y < 180:  # Add new page if needed
            c.showPage()
            y = height - 50

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"{idx}. Similar Case: {os.path.basename(rec['path'])}")
        y -= 20

        # Load and embed image
        try:
            img_path = rec['path']
            case_img = Image.open(img_path).convert('RGB')
            case_img.thumbnail((120, 120))
            img_reader = ImageReader(case_img)
            c.drawImage(img_reader, 50, y - 120, width=120, height=120)
        except Exception as e:
            c.drawString(50, y - 20, f"[Image load failed: {e}]")
        y -= 130

        # Grade and treatment
        c.setFont("Helvetica", 12)
        c.drawString(180, y + 100, f"Wagner Grade: {rec['grade']}")
        c.drawString(180, y + 80, "Recommended Treatments:")
        offset_y = 60
        for step in rec['treatment_plan']:
            c.drawString(200, y + offset_y, f"- {step}")
            offset_y -= 15

        y = y + offset_y - 40  # Prepare for next block

    c.save()

    return FileResponse(path=pdf_path, filename="treatment_report.pdf", media_type='application/pdf')
