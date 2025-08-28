# mport gradio as gr
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# from PIL import Image

# # -------------------------
# # Load Model
# # -------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define the same model architecture you trained
# model = models.resnet50(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes: Normal vs Pneumonia

# # Load trained weights
# model.load_state_dict(torch.load("best_model.pth", map_location=device))
# model.to(device)
# model.eval()

# # -------------------------
# # Define Image Transform
# # -------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),   # resize to match ResNet input
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# # -------------------------
# # Prediction Function
# # -------------------------
# def predict(img):
#     img = Image.fromarray(img)  # Ensure PIL format
#     inputs = transform(img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)

#     classes = ["Normal", "Pneumonia"]  # Change if you trained on more classes
#     return {classes[0]: float(torch.softmax(outputs, 1)[0][0]),
#             classes[1]: float(torch.softmax(outputs, 1)[0][1])}

# # -------------------------
# # Gradio Interface
# # -------------------------
# interface = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="numpy", label="Upload an X-ray"),
#     outputs=gr.Label(num_top_classes=2, label="Prediction"),
#     title="Medical X-Ray Classifier",
#     description="Upload a chest X-ray to check if it looks Normal or Pneumonia (AI-based)."
# )

# if __name__ == "__main__":
#     interface.launch()



# ========================================================================================================================================================

# -----------------------------------------------------------------------------------------------------------------
# 3rd


import os
import warnings
from typing import Tuple

import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import google.generativeai as genai

# ------------------- CONFIG -------------------
# 1) Gemini API Key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found in environment. Set it before running the app."
    )

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-1.5-flash"
model_ai = genai.GenerativeModel(GEMINI_MODEL_NAME)

# 2) Torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) Load your trained model (ResNet50 with 2 classes)
#    Uses the modern torchvision API (weights=None) to avoid deprecated warnings
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Binary classification: Normal vs Pneumonia

MODEL_PATH = os.getenv("XRAY_MODEL_PATH", "best_model.pth")
if not os.path.exists(MODEL_PATH):
    warnings.warn(
        f"Model checkpoint '{MODEL_PATH}' not found. The app will run, but predictions will fail until the file is provided.")
else:
    state = torch.load(MODEL_PATH, map_location=device)
    # Support both plain state_dict and checkpoint dicts
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    # If the keys are prefixed (e.g., 'module.'), strip them
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=False)

model.to(device)
model.eval()

# Image transforms (match ImageNet normalization expected by ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Labels
class_names = ["Normal", "Pneumonia"]

# ------------------- HELPERS -------------------

def _normalize_text(s: str) -> str:
    return (s or "").lower().replace("-", " ").strip()


def _is_chest_xray(label: str) -> bool:
    s = _normalize_text(label)
    return ("chest" in s) and ("xray" in s or "x ray" in s)


def _is_any_xray(label: str) -> bool:
    s = _normalize_text(label)
    return ("xray" in s or "x ray" in s)

# ------------------- GEMINI FUNCTIONS -------------------

def check_image_type_with_gemini(image: Image.Image) -> str:
    """
    Uses Gemini to classify the high-level type in <= 3 words.
    Target categories: "chest x-ray", "other x-ray", "random object".
    Returns a lowercase string.
    """
    try:
        response = model_ai.generate_content([
            (
                "Classify this image into one of these categories only: "
                "'chest x-ray', 'other x-ray', or 'random object'. "
                "Return just the category in <= 3 words."
            ),
            image,
        ])
        return _normalize_text(response.text)
    except Exception as e:
        return f"error: {e}".lower()

chat = model_ai.start_chat(history=[]) # for history

def diagnose_with_gemini(image: Image.Image) -> str:
    """A brief, simple description/diagnosis (max 2 sentences)."""
    try:
        response = chat.send_message([ # changeed to generate content 
                ("you must summerize the final answer in 2 lines at max, DO NOT GIVE MORE THAN 2 LINES RESULT"
                "You are a highly skilled medical imaging assistant specialized in radiology. "
                "Analyze the provided chest X-ray image and produce a clear, structured report. "
                "Include: suspected condition(s), affected region(s), short medical explanation, confidence (low/medium/high), and a patient-friendly summary. "
                "If the image is unclear or inconclusive, say so and explain what additional data would help. "
                "Do not provide definitive clinical diagnosis — recommend follow-up with a qualified radiologist or physician."
                "If it's not an X-ray, briefly describe what the image shows. "
                "you must summerize the final answer in 2 lines at max, DO NOT GIVE MORE THAN 2 LINES RESULT"
            ),
            image,
        ])
        return (response.text or "").strip()
    except Exception as e:
        return f"AI diagnosis failed: {e}"

# ------------------- PREDICTION PIPELINE -------------------

def predict(image: Image.Image) -> str:
    """
    Gradio handler:
    1) Check type with Gemini
    2) If chest X-ray -> run ML model + Gemini description
    3) If other X-ray -> only Gemini description
    4) Else -> app is for X-rays only
    """
    if image is None:
        return "No image received. Please upload a file."

    try:
        img = image.convert("RGB")
    except Exception:
        return "Could not read the image. Try another file."

    # Step 1: Gemini classification of type
    img_type = check_image_type_with_gemini(img)

    # Step 2: Route based on Gemini type
    try:
        if _is_chest_xray(img_type):
            # Run ML model
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                conf, pred_class = torch.max(probs, dim=0)
            ml_result = f"Model Prediction: {class_names[int(pred_class)]} ({conf.item()*100:.2f}%)"
            chat.send_message(ml_result)
            ai_diag = diagnose_with_gemini(img)
            return (
                "✅ X-ray detected.\n\n" +
                ml_result + "\n\n" +
                "AI says: " + ai_diag  
            )

        elif _is_any_xray(img_type):
            ai_diag = diagnose_with_gemini(img)
            return (
                "ℹ️ X-ray detected.\n\n" +
                "AI says: " + ai_diag
            )
        else:
            ai_diag = diagnose_with_gemini(img)
            return (
                "⚠️ This app is only for X-ray images.\n\n" +
                "Detected: " + ai_diag
            )
    except FileNotFoundError:
        return (
            "Model file not found. Ensure 'best_model.pth' exists or set XRAY_MODEL_PATH env var "
            "to your checkpoint path."
        )
    except Exception as e:
        return f"Error processing image: {e}"

        # for chatbot 
def chat_with_gemini(user_message):
    response = chat.send_message(user_message)
    return response.text


# ------------------- GRADIO UI -------------------
# with gr.Blocks() as iface:=======================
# iface = gr.Interface(
#     fn=predict, #oldd
#     inputs=gr.Image(type="pil", label="Upload an image (preferably an X-ray)"),
#     outputs=gr.Textbox(label="Result", lines=8),
#     title="🩻 Medical X-Ray Classifier + AI Assistant",
#     description=(
# #         "Upload an image. If it's a chest X-ray, the ML model + AI will analyze it. "
# #         "For other images, AI will describe it briefly.\n"
# #         "**Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice."
# #     ),
# # )
#     gr.Markdown("## 🤖 Gemini Chatbot with History")===========

#     # old 2
#     # gr.Markdown("### 🩻 Upload an X-ray for Prediction")
#     img = gr.Image(type="pil", label="Upload an image (preferably an X-ray)") ====================
#     # out = gr.Textbox(label="Result", lines=8)
#     # img.upload(predict, img, out)   # or img.submit(...)

    
#     # img.upload(lambda image, history: (history + [("🩻 Uploaded Image", predict(image))]), [img, chatbot], chatbot)
#     chatbot = gr.Chatbot(height=400)========================
# ====================
#     msg = gr.Textbox(placeholder="Type your message here...")
#     clear = gr.Button("Clear Chat")

#     def respond(user_message, chat_history):
#         bot_reply = chat_with_gemini(user_message)
#         chat_history.append((user_message, bot_reply))
#         return "", chat_history

#     def reset_chat():
#         global chat
#         chat = model_ai.start_chat(history=[])   # reset Gemini's memory too
#         return []

#     img.upload(lambda image, history: (history + [("🩻 Uploaded Image", predict(image))]), [img, chatbot], chatbot)
#     msg.submit(respond, [msg, chatbot], [msg, chatbot])
#     clear.click(reset_chat, None, chatbot, queue=False)

# ------------------- GRADIO UI -------------------
with gr.Blocks() as iface:
    gr.Markdown("## 🩻 Medical X-Ray Classifier + 🤖 Gemini Chatbot")

    with gr.Row():
        # ---- LEFT SIDE: Chatbot ----
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=500, label="Gemini Chatbot")
            msg = gr.Textbox(placeholder="Type your message here...")
            clear = gr.Button("Clear Chat")

            def respond(user_message, chat_history):
                bot_reply = chat_with_gemini(user_message)
                chat_history.append((user_message, bot_reply))
                return "", chat_history

            def reset_chat():
                global chat
                chat = model_ai.start_chat(history=[])   # reset Gemini’s memory too
                return []

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(reset_chat, None, chatbot, queue=False)

        # ---- RIGHT SIDE: Image Upload ----
        with gr.Column(scale=1):
            gr.Markdown("### 🩻 Upload an X-ray for Prediction")
            img = gr.Image(type="pil", label="Upload an image (preferably an X-ray)")
            
            # When image is uploaded → send result into chatbot as new message
            img.upload(
                lambda image, history: (history + [("🩻 Uploaded Image", predict(image))]),
                [img, chatbot],
                chatbot
            )


if __name__ == "__main__":
    # Set share=True if you want a public URL (e.g., on Colab). On Spaces, keep default.
    iface.launch()

