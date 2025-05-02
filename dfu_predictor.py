import numpy as np
import torch
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

# ========================
# Mapping Wagner grades to treatment plans
# ========================
TREATMENT_MAP = {
    1: [
        "Offloading (reduce pressure on foot)",
        "Local wound care (cleaning, basic dressings)",
        "Monitor for progression",
        "Antibiotics if cellulitis is present"
    ],
    2: [
        "Aggressive wound care",
        "Specialised dressings",
        "Consider minor surgical debridement",
        "Start antibiotics prophylactically"
    ],
    3: [
        "Surgical debridement required",
        "Broad-spectrum IV antibiotics",
        "Imaging to assess bone infection (MRI)",
        "Possible hospitalisation"
    ],
    4: [
        "Emergency surgical intervention (e.g., amputation)",
        "Vascular assessment for blood flow",
        "IV antibiotics",
        "Hospitalisation and multidisciplinary team management"
    ]
}

# ========================
# Image Preprocessing
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========================
# Feature Extractor
# ========================
def load_feature_extractor(model_path, device='cpu'):
    try:
        # Load the ConvNeXt model and its weights
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()  # Set the model to evaluation mode
        
        # Extract feature extractor part of the model
        trunk = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
        return trunk
    except Exception as e:
        raise Exception(f"Failed to load feature extractor: {str(e)}")

# ========================
# KNN Recommender Class
# ========================
class DfuRecommender:
    def __init__(self, feats_np, grades_np, paths_np, n_neighbors=6):
        self.feats = np.load(feats_np)
        self.grades = np.load(grades_np)
        self.paths = np.load(paths_np)
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.knn.fit(self.feats)

    def recommend(self, img_tensor, feature_extractor, k=5, device='cuda'):
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            f = feature_extractor(img_tensor.unsqueeze(0)).flatten(1).cpu().numpy()
        dists, idxs = self.knn.kneighbors(f, n_neighbors=k+1)
        results = []
        for dist, idx in zip(dists[0][1:], idxs[0][1:]):
            grade = int(self.grades[idx])
            results.append({
                'path': self.paths[idx],
                'grade': grade,
                'distance': float(dist),
                'treatment_plan': TREATMENT_MAP.get(grade, [])
            })
        return results

# ========================
# Grade Prediction
# ========================
def predict_grade(img_tensor, model_path, device='cpu'):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 4)  # 4 classes for Wagner grades
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device).eval()

    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
        pred = torch.argmax(logits, dim=1).item()
    return pred + 1  # Convert 0-indexed to Grade 1-4

# ========================
# Grad-CAM Generation
# ========================
def generate_gradcam(img_tensor, model_path, device='cpu'):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 4)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device).eval()

    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Register hooks to capture activations and gradients
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]  # Last convolution layer
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    pred_class = output.argmax(dim=1)
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()

    grad = gradients[0].detach()
    act = activations[0].detach()

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()
    cam = F.relu(cam)

    # Normalize and apply heatmap
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Convert original tensor back to image and overlay heatmap
    img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    img_np = np.uint8(255 * img_np)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay)

# ========================
# Optional: CLI runner for testing
# ========================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict DFU recommendations')
    parser.add_argument('model_path', type=str)
    parser.add_argument('feats_path', type=str)
    parser.add_argument('grades_path', type=str)
    parser.add_argument('paths_path', type=str)
    parser.add_argument('image_path', type=str)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    extractor = load_feature_extractor(args.model_path, device=args.device)
    recommender = DfuRecommender(args.feats_path, args.grades_path, args.paths_path)

    img = Image.open(args.image_path).convert('RGB')
    img_t = transform(img)
    recs = recommender.recommend(img_t, extractor, k=args.k, device=args.device)

    print("Top similar cases and recommended treatments:\n")
    for idx, r in enumerate(recs, start=1):
        print(f"{idx}. Path: {r['path']}")
        print(f"   Wagner Grade: {r['grade']}")
        print("   Treatment Plan:")
        for step in r['treatment_plan']:
            print(f"     - {step}")
        print()
