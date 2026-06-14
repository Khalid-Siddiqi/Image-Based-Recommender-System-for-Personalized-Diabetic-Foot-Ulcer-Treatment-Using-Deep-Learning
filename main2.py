import io
import os
import cv2
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

# Import your custom modules
from dfu_predictor import load_feature_extractor, DfuRecommender, TREATMENT_MAP, transform, predict_grade

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models and paths
seg_model = YOLO("models/Instance_Segementation_Model.pt")
model_path = 'models/best_convnext.pth'
feats_path = 'npy_folder/dfu_feats.npy'
grades_path = 'npy_folder/dfu_grades.npy'
paths_path = 'npy_folder/dfu_paths.npy'

# Load feature extractor and recommender
extractor = load_feature_extractor(model_path, device=DEVICE)
recommender = DfuRecommender(feats_path, grades_path, paths_path)

app = FastAPI()

def run_instance_segmentation(pil_image):
    image = np.array(pil_image)
    # Convert RGB to BGR for YOLO/OpenCV
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Run the YOLO segmentation model
    results = seg_model.predict(bgr_image)

    # Check if the model found anything
    if len(results) == 0 or results[0].masks is None:
        return None, False

    # Use YOLO's native plotting engine. 
    annotated_bgr = results[0].plot() 
    
    # Convert back to RGB for the PIL Image pipeline
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(annotated_rgb), True

@app.post("/predict_image")
async def predict_and_return_image(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    
    # 1. Read image into PIL
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # 2. Run Segmentation
    heatmap_img, found_ulcer = run_instance_segmentation(img)

    if not found_ulcer:
        # If no ulcer is found, prepare to write on the original image
        result_img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        display_text = "No DFU Detected"
        text_color = (0, 0, 255) # Red text
    else:
        # 3. If ulcer found, run classification
        img_t = transform(img)
        predicted_grade = predict_grade(img_t, model_path, device=DEVICE)
        
        # Prepare to write on the segmented heatmap image
        result_img_cv = cv2.cvtColor(np.array(heatmap_img), cv2.COLOR_RGB2BGR)
        display_text = f"Wagner Grade: {predicted_grade}"
        text_color = (0, 255, 0) # Green text

    # 4. Write the text onto the image using OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    # Calculate text size to place it nicely at the top left with a dark background box for readability
    (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
    cv2.rectangle(result_img_cv, (10, 10), (10 + text_width + 10, 10 + text_height + 15), (0, 0, 0), cv2.FILLED)
    cv2.putText(result_img_cv, display_text, (15, 10 + text_height + 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # 5. Convert back to PIL Image
    final_pil_image = Image.fromarray(cv2.cvtColor(result_img_cv, cv2.COLOR_BGR2RGB))

    # --- NEW CODE: Save the image locally ---
    # Create the output directory if it doesn't exist
    os.makedirs("output_images", exist_ok=True)
    
    # Generate a unique filename using the original name and a timestamp
    safe_filename = os.path.splitext(file.filename)[0] # Extracts the name without the extension
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"output_images/predicted_{safe_filename}_{timestamp}.jpg"
    
    # Save to disk
    final_pil_image.save(save_path, format='JPEG')
    print(f"✅ Image successfully saved to: {save_path}")
    # ----------------------------------------

    # 6. Save image to a BytesIO object in memory for the API response
    img_byte_arr = io.BytesIO()
    final_pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # 7. Return the image directly to the client
    return StreamingResponse(img_byte_arr, media_type="image/jpeg")