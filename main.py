# step 01
# Import necessary libraries
# step 02
import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from fpdf import FPDF

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and setup feature extractor
def load_model():
    model = torch.load('swin_model.pth')
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    return model, feature_extractor

model, feature_extractor = load_model()

# Preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load stored features and metadata (precomputed from your dataset)
stored_features = np.load('stored_features.npy')  # Shape: [num_samples, embedding_dim]
stored_metadata = np.load('stored_metadata.npy', allow_pickle=True)  # Contains Wagner grades and case IDs

# Treatment mapping dictionary
treatment_plans = {
    1: "Offloading, local wound care, monitor for progression.",
    2: "Surgical debridement, antibiotics, specialized dressings.",
    3: "IV antibiotics, hospitalization, bone infection imaging.",
    4: "Emergency surgery, vascular assessment, multidisciplinary care."
}

def extract_features(image_path):
    """Extract features using Swin Transformer"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img_tensor).squeeze().numpy()
    return features

def find_similar_cases(query_features, k=5):
    """Find similar cases using KNN"""
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(stored_features)
    distances, indices = nn.kneighbors([query_features])
    return indices[0]

def create_pdf_report(grade, treatment, image_path, output_path="report.pdf"):
    """Generate PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add content
    pdf.cell(200, 10, txt=f"Predicted Wagner Grade: {grade}", ln=True)
    pdf.cell(200, 10, txt=f"Recommended Treatment:", ln=True)
    pdf.multi_cell(0, 10, txt=treatment)
    
    # Add image
    pdf.image(image_path, x=10, y=50, w=100)
    
    pdf.output(output_path)
    return output_path

@app.post("/process-dfu")
async def process_dfu(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # 1. Feature extraction
        features = extract_features(temp_path)
        
        # 2. Predict Wagner grade
        img_tensor = preprocess(Image.open(temp_path).convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
        grade = int(torch.argmax(output).item()) + 1  # Assuming grades are 1-4
        
        # 3. Find similar cases
        similar_indices = find_similar_cases(features)
        similar_cases = stored_metadata[similar_indices].tolist()
        
        # 4. Get treatment plan
        treatment = treatment_plans.get(grade, "Unknown grade")
        
        # 5. Generate PDF report
        report_path = create_pdf_report(grade, treatment, temp_path)
        
        # 6. Grad-CAM (implement your Grad-CAM logic here)
        # grad_cam_path = generate_grad_cam(temp_path)
        
        return {
            "grade": grade,
            "treatment": treatment,
            "similar_cases": similar_cases,
            "report_path": report_path
        }
    
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/get-report")
async def get_report(path: str):
    return FileResponse(path, media_type='application/pdf', filename="treatment_report.pdf")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



