

# predict_knn.py
import os
import io
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import convnext_tiny
from PIL import Image
from sklearn.neighbors import NearestNeighbors

def load_feature_extractor(model_path, device='cuda'):
    model = convnext_tiny(pretrained=False)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 4)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    trunk = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    return trunk

# Build recommender
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
            results.append({
                'path': self.paths[idx],
                'grade': int(self.grades[idx]),
                'distance': float(dist)
            })
        return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict DFU recommendations')
    parser.add_argument('--model_path', required=True, help='Path to ConvNext .pth')
    parser.add_argument('--feats', required=True, help='Path to dfu_feats.npy')
    parser.add_argument('--grades', required=True, help='Path to dfu_grades.npy')
    parser.add_argument('--paths', required=True, help='Path to dfu_paths.npy')
    parser.add_argument('--image', required=True, help='Path to new DFU image')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Setup
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    extractor = load_feature_extractor(args.model_path, device=args.device)
    recommender = DfuRecommender(args.feats, args.grades, args.paths)

    # Process new image
    img = Image.open(args.image).convert('RGB')
    img_t = transform(img)
    recs = recommender.recommend(img_t, extractor, k=args.k, device=args.device)
    print("Top similar cases:")
    for r in recs:
        print(r)
