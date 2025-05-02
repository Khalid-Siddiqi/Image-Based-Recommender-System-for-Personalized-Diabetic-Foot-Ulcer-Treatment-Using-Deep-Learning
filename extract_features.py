# extract_features.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_tiny
from PIL import Image

class DfuDataset(Dataset):
    """
    Dataset for DFU images organized in subfolders per Wagner grade.
    Expects root_dir/GradeX/*.jpg where X is the grade (1-4).
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for entry in os.listdir(root_dir):
            folder = os.path.join(root_dir, entry)
            if not os.path.isdir(folder):
                continue
            try:
                grade = int(''.join(filter(str.isdigit, entry)))
            except ValueError:
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    path = os.path.join(folder, fname)
                    self.samples.append((path, grade))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, grade = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, grade, path

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature extraction function
def extract_features(root_dir, model_path, output_dir, batch_size=32, device='cuda'):
    # Prepare dataset and loader
    dataset = DfuDataset(root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load fine-tuned ConvNext
    model = convnext_tiny(pretrained=False)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Remove classification head
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)

    feats, grades, paths = [], [], []
    with torch.no_grad():
        for imgs, gs, ps in loader:
            imgs = imgs.to(device)
            out = feature_extractor(imgs).flatten(1).cpu().numpy()
            feats.append(out)
            grades.extend(gs)
            paths.extend(ps)

    feats = np.vstack(feats)
    grades = np.array(grades)
    paths = np.array(paths)

    # Save .npy files
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'dfu_feats.npy'), feats)
    np.save(os.path.join(output_dir, 'dfu_grades.npy'), grades)
    np.save(os.path.join(output_dir, 'dfu_paths.npy'), paths)
    print(f"Saved {len(feats)} feature vectors to {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract DFU image features')
    parser.add_argument('--data_dir', required=True, help='Root folder of DFU images')
    parser.add_argument('--model_path', required=True, help='Path to fine-tuned ConvNext .pth')
    parser.add_argument('--output_dir', required=True, help='Directory to save .npy files')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    extract_features(args.data_dir, args.model_path, args.output_dir, args.batch_size, args.device)
