"""
Inference script to detect whether an input image is REAL or FAKE/AI.
Compatible with models trained on CIFAKE or custom datasets.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import build_model

def get_inference_transform(img_size=32):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

def predict(image_path, model_path="best_model.pth"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}. Train the model first using train.py!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes", ["FAKE", "REAL"])
    model_type = checkpoint.get("model_type", "resnet18")
    img_size = checkpoint.get("img_size", 32)

    model = build_model(model_type=model_type, num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = get_inference_transform(img_size=img_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    predicted_label = classes[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    print("=" * 45)
    print(f" Image:      {image_path}")
    print(f" Prediction: {predicted_label.upper()}")
    print(f" Confidence: {confidence_score:.2f}%")
    print("-" * 45)
    print(" Class Probabilities:")
    for idx, cls in enumerate(classes):
        print(f"  • {cls}: {probabilities[idx].item() * 100:.2f}%")
    print("=" * 45)

    return predicted_label, confidence_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Image Detector")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Checkpoint file path")
    args = parser.parse_args()

    predict(image_path=args.image, model_path=args.model)
