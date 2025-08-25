import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import numpy as np
import os

# ---------- Config ----------
CLASSES = ["Normal", "Pneumonia"]  # Change if your dataset has different labels
MODEL_PATH = "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image transforms (common for ResNet)
IMG_TFMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------- Model ----------
def load_model():
    model = models.resnet50(weights=None)
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, len(CLASSES))
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model

MODEL = load_model()

# ---------- Inference ----------
@torch.inference_mode()
def predict(img):
    inputs = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred = torch.argmax(probs).item()
    result = CLASSES[pred]
    prob_str = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return result, str(prob_str)

# ---------- UI ----------
disclaimer = "⚠️ This is a demo. Not for medical use."

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload X-ray"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="Probabilities (per class)"),
    ],
    title="Chest X-ray Classifier",
    description="ResNet50 model trained to classify Normal vs Pneumonia X-rays. ⚠️ Not for medical use.",
)



if __name__ == "__main__":
    demo.launch()

