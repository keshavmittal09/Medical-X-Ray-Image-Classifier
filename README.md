# Medical X-Ray Image Classifier 🩻

This project is a **Chest X-Ray Pneumonia Classifier** built using **ResNet50** pretrained on ImageNet and fine-tuned on the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

🚀 **Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/qwert01/Medical-X-Ray-Image-Classifier)  
📂 **GitHub Repo**: [Medical-X-Ray-Image-Classifier](https://github.com/keshavmittal09/Medical-X-Ray-Image-Classifier.git)

---

## 📖 Project Overview
Pneumonia is a serious lung infection that can be detected via chest X-rays.  
This model leverages **transfer learning** with ResNet50 to classify X-ray scans into:

- **Normal**
- **Pneumonia**

The goal is to provide a simple, accessible tool to assist in medical diagnosis.

---

## ⚙️ Tech Stack
- **Python**
- **PyTorch**
- **ResNet50 (pretrained)**
- **Hugging Face Spaces (Gradio UI)**
- **Kaggle Dataset**

---

## 📊 Dataset
We used the public dataset:  
👉 [Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- **Train**: Normal (1341), Pneumonia (3875)  
- **Test**: Normal (234), Pneumonia (390)

---

## 🚀 Model Training
1. **Preprocessing**  
   - Image resizing to 224x224  
   - Normalization using ImageNet stats  
   - Data augmentation (flips, rotations)

2. **Model**  
   - Base: ResNet50 (pretrained on ImageNet)  
   - Final FC layer modified for binary classification  
   - Loss: CrossEntropyLoss  
   - Optimizer: Adam  

3. **Evaluation Metrics**  
   - Accuracy  
   - Precision, Recall, F1-score  

---

## ▶️ Deployment
- The model is deployed on **Hugging Face Spaces** using **Gradio**.  
- Users can upload an X-ray image and get instant predictions.

🔗 [Try it here](https://huggingface.co/spaces/qwert01/Medical-X-Ray-Image-Classifier)

---

## 📦 Installation (Local Setup)
```bash
# Clone the repo
git clone https://github.com/keshavmittal09/Medical-X-Ray-Image-Classifier.git
cd Medical-X-Ray-Image-Classifier

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

---

## 📌 Future Improvements
- Train on a larger, more diverse dataset  
- Add multi-class classification (e.g., COVID-19, Tuberculosis)  
- Improve explainability with Grad-CAM visualizations  

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!  
Feel free to open an issue or submit a pull request.

---

## 📝 License
This project is licensed under the **MIT License**.

---

### 👨‍💻 Author
Developed by [Keshav Mittal](https://github.com/keshavmittal09)
