import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
from torch import nn

import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),
        nn.ReLU(),
        nn.Dropout(2),
        nn.Linear(512, 512),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    image = Image.open(image_path)

    # Resize and crop
    image = image.resize((256, 256)).crop((16, 16, 240, 240))

    # Normalize
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    return torch.tensor(np_image, dtype=torch.float32)

def predict(image_path, model, topk=5, device='cpu'):
    model.to(device)
    model.eval()

    image = process_image(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        logps = model(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i.item()] for i in top_class[0]]

    return top_p[0].tolist(), top_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    model = load_checkpoint(args.checkpoint)

    probs, classes = predict(args.image_path, model, topk=args.top_k, device=device)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]

    for prob, cls in zip(probs, classes):
        print(f"{cls}: {prob*100:.2f}%")

if __name__ == "__main__":
    main()
