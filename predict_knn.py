# predict_knn.py
# Requires: pip install torch torchvision scikit-learn pillow numpy

import os
import io
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import convnext_tiny
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# Mapping Wagner grades to treatment plans
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

# Load ConvNext trunk as feature extractor
def load_feature_extractor(model_path, device='cpu'):
    model = convnext_tiny(pretrained=False)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 4)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    # strip final head
    trunk = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    return trunk

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict DFU recommendations')
    parser.add_argument('model_path', type=str, help='Path to ConvNext .pth')
    parser.add_argument('feats_path', type=str, help='Path to dfu_feats.npy')
    parser.add_argument('grades_path', type=str, help='Path to dfu_grades.npy')
    parser.add_argument('paths_path', type=str, help='Path to dfu_paths.npy')
    parser.add_argument('image_path', type=str, help='Path to test DFU image')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu or cuda)')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
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
