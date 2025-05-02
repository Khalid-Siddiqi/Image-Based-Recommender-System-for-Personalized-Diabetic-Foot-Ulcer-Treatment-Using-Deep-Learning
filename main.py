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
from ultralytics import YOLO
import cv2
import numpy as np
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

seg_model = YOLO("models/Instance_Segementation_Model.pt")
model_path = 'models/best_convnext.pth'
feats_path = 'npy_folder/dfu_feats.npy'
grades_path = 'npy_folder/dfu_grades.npy'
paths_path = 'npy_folder/dfu_paths.npy'

# Load model and recommender
extractor = load_feature_extractor(model_path, device=DEVICE)
recommender = DfuRecommender(feats_path, grades_path, paths_path)
def run_instance_segmentation(pil_image):
    # Convert PIL to OpenCV format
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Predict
    results = seg_model.predict(image)
    annotated_image = image.copy()
    alpha = 0.5

    for result in results:
        for mask, box in zip(result.masks, result.boxes):
            mask_array = mask.data.cpu().numpy()
            if mask_array.shape[1:] != annotated_image.shape[:2]:
                mask_array = cv2.resize(mask_array[0], (annotated_image.shape[1], annotated_image.shape[0]))

            green_overlay = np.zeros_like(annotated_image, dtype=np.uint8)
            green_overlay[mask_array.astype(bool)] = (0, 0, 255)  # Blue overlay
            annotated_image = cv2.addWeighted(green_overlay, alpha, annotated_image, 1 - alpha, 0)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Convert back to PIL
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_image)

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
    # Generate instance segmentation image
    segmented_img = run_instance_segmentation(img)
    segmented_img.thumbnail((150, 150))

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

    c.drawString(50, y, "Patient Input (Left), Grad-CAM (Middle), and Segmentation (Right):")
    c.drawImage(ImageReader(img_orig), 50, y - 160, width=150, height=150)
    c.drawImage(ImageReader(gradcam_img), 220, y - 160, width=150, height=150)
    c.drawImage(ImageReader(segmented_img), 390, y - 160, width=150, height=150)
    
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
