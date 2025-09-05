
# 🩻 AI-Powered Medical X-Ray Classifier + Gemini Chatbot  

An intelligent **web-based AI assistant** that combines **Deep Learning (ResNet50)** and **Generative AI (Google Gemini)** to analyze chest X-rays, detect **Pneumonia vs. Normal**, validate input images, generate **patient-friendly summaries**, and provide an **interactive medical chatbot**.  

🌐 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/qwert01/Medical-X-Ray-Image-Classifier)  
📂 **Source Code:** [GitHub Repo](https://github.com/keshavmittal09/Medical-X-Ray-Image-Classifier)  

---

## 🚨 Problem Statement  
Manual interpretation of chest X-rays is:  
- **Time-Consuming** – Requires detailed review for every scan.  
- **Error-Prone** – Human fatigue and subtle cues can lead to missed diagnoses.  
- **Expertise-Dependent** – Relies on availability of specialized radiologists.  

Traditional automated systems are limited and rigid.  
👉 This project demonstrates how **AI can act as a radiology assistant**, delivering **faster, reliable, and patient-friendly insights**.  

---

## ✨ Key Features  
✅ **Dual AI System** – ResNet50 (classification) + Gemini (validation, summarization, chatbot)  
✅ **Binary Classification** – Detects **Normal vs. Pneumonia** with confidence score  
✅ **Smart Image Validation** – Gemini confirms if input is an X-ray (Chest / Other / Random object)  
✅ **Concise Medical Summaries** – ≤2 lines, simplified for patients  
✅ **Interactive Chatbot** – Gemini-powered Q&A for follow-up queries  
✅ **User-Friendly UI** – Built with **Gradio** for accessibility  

---

## 🏗️ System Architecture  

1. **Image Upload** → User uploads X-ray via Gradio interface  
2. **Gemini AI (Image Validation)** → Classifies as Chest X-ray / Other X-ray / Random Object  
3. **ResNet50 Model** → Predicts Normal or Pneumonia (with confidence score)  
4. **Gemini AI (Summary)** → Generates short, clear medical explanation  
5. **Gemini Chatbot** → Users ask follow-ups & get AI-generated responses  

---

## 🚀 Demo Workflow  

1. Upload an image (preferably chest X-ray)  
2. Gemini validates input type  
3. If chest X-ray → ResNet50 predicts **Normal / Pneumonia**  
4. Gemini generates patient-friendly summary  
5. If non-medical image → User is alerted  
6. Engage with chatbot for further queries  

---

## 🛠️ Tech Stack  

- **Deep Learning:** PyTorch, ResNet50  
- **Generative AI:** Google Gemini (Validation, Summaries, Chatbot)  
- **Frontend:** Gradio (Web UI)  
- **Other Tools:** PIL, TorchVision, Hugging Face Spaces (deployment)  

---

## 📊 Results & Impact  

- Provides **quick, reliable assistance** for chest X-ray analysis  
- Makes results **accessible to both doctors & patients**  
- Reduces workload on radiologists while minimizing diagnostic delays  

---

## 🔮 Future Work  

- **Data Expansion** → Larger, diverse datasets for better accuracy  
- **Multi-Disease Detection** → Extend to TB, COVID-19, and other lung conditions  
- **Clinical Integration** → Embed into hospital PACS systems  
- **Continuous Learning** → Improve via feedback & new cases  

---

## 📌 How to Run Locally  

```bash
# Clone repo
git clone https://github.com/keshavmittal09/Medical-X-Ray-Image-Classifier
cd Medical-X-Ray-Image-Classifier

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```

Then open **http://127.0.0.1:7860/** in your browser.  

---

## 🙌 Acknowledgments  
- **Google Gemini AI** – For image validation, summaries, and chatbot  
- **PyTorch & TorchVision** – For deep learning backbone  
- **Hugging Face Spaces** – For easy deployment  
- **Gradio** – For simple & intuitive UI  
