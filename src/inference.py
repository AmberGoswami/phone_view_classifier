import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import argparse
import os
from datetime import datetime

MODEL_PATH = "weights/model.pth"
IMG_SIZE = 224
CLASS_NAMES = ['back', 'front', 'none']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    model = load_model()
    with torch.no_grad():
        logits = model(image_tensor)                          
        probs = F.softmax(logits, dim=1)[0]           
        conf, idx = probs.max(0)                     
        predicted_class = CLASS_NAMES[idx.item()]

    return predicted_class, conf.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    prediction, confidence_score = predict_image(args.image_path)
    print(f"✅ Predicted class: {prediction}, Confidence Score: {confidence_score}")
