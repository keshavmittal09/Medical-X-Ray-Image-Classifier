mport gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# -------------------------
# Load Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same model architecture you trained
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes: Normal vs Pneumonia

# Load trained weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# -------------------------
# Define Image Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # resize to match ResNet input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Prediction Function
# -------------------------
def predict(img):
    img = Image.fromarray(img)  # Ensure PIL format
    inputs = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    classes = ["Normal", "Pneumonia"]  # Change if you trained on more classes
    return {classes[0]: float(torch.softmax(outputs, 1)[0][0]),
            classes[1]: float(torch.softmax(outputs, 1)[0][1])}

# -------------------------
# Gradio Interface
# -------------------------
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload an X-ray"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Medical X-Ray Classifier",
    description="Upload a chest X-ray to check if it looks Normal or Pneumonia (AI-based)."
)

if __name__ == "__main__":
    interface.launch()
