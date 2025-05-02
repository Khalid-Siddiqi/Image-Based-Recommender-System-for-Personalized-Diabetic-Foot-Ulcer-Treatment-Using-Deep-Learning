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
from predict_knn import load_feature_extractor, DfuRecommender, TREATMENT_MAP, transform, predict_grade
from predict_knn import generate_gradcam

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = 'best_convnext.pth'
feats_path = 'npy_folder/dfu_feats.npy'
grades_path = 'npy_folder/dfu_grades.npy'
paths_path = 'npy_folder/dfu_paths.npy'

# Load model and recommender
extractor = load_feature_extractor(model_path, device=DEVICE)
recommender = DfuRecommender(feats_path, grades_path, paths_path)

app = FastAPI()

@app.post("/predict")
async def predict_and_generate_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_t = transform(img)
    gradcam_img = generate_gradcam(img_t, model_path, device=DEVICE)

    # Predict grade
    predicted_grade = predict_grade(img_t, model_path, device=DEVICE)

    # Get recommendations
    recs = recommender.recommend(img_t, extractor, k=2, device=DEVICE)

    # Generate PDF
    pdf_path = "treatment_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Medical Research Institute of ADPM")
    c.setFont("Helvetica", 12)
    y = height - 90

    # Input image and predicted grade
    img_orig = img.copy()
    img_orig.thumbnail((150, 150))
    gradcam_img.thumbnail((150, 150))

    c.drawString(50, y, "Patient Input Image (Left) and Grad-CAM Visualization (Right):")
    c.drawImage(ImageReader(img_orig), 50, y - 160, width=150, height=150)
    c.drawImage(ImageReader(gradcam_img), 220, y - 160, width=150, height=150)
    
    y -= 180
    c.drawString(50, y, f"Predicted Wagner Grade: {predicted_grade}")
    y -= 20
    # Treatment plan for predicted grade
    c.drawString(50, y, "Recommended Treatment Plan:")
    for step in TREATMENT_MAP[predicted_grade]:
        c.drawString(70, y - 15, f"- {step}")
        y -= 15
    y -= 20

    # Show 2 recommendations
        # Show 2 recommendations
    for idx, rec in enumerate(recs, 1):
        if y < 180:
            c.showPage()
            y = height - 50
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"{idx}. Similar Case: {os.path.basename(rec['path'])}")
        y -= 20

        try:
            # Reconstruct path: data/Grade {rec['grade']}/<filename>
            grade_folder = f"Grade {int(rec['grade'])}"
            img_filename = os.path.basename(rec['path'])
            corrected_path = os.path.join("data", grade_folder, img_filename)

            case_img = Image.open(corrected_path).convert('RGB')
            case_img.thumbnail((120, 120))
            img_reader = ImageReader(case_img)
            c.drawImage(img_reader, 50, y - 120, width=120, height=120)
        except Exception as e:
            c.drawString(50, y - 20, f"[Image load failed: {e}]")

        y -= 130
        c.setFont("Helvetica", 12)
        c.drawString(180, y + 100, f"Wagner Grade: {rec['grade']}")
        y -= 20

    c.save()
    return FileResponse(path=pdf_path, filename="treatment_report.pdf", media_type='application/pdf')
