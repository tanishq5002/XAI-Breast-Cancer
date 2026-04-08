# 🧠 Breast Cancer Detection Using Explainable Deep Learning

## 📌 Overview
This project presents an Explainable AI (XAI) based deep learning system for detecting breast cancer from histopathology images. 

A Convolutional Neural Network (CNN) is used to classify images into **Benign** or **Malignant**, and Grad-CAM is applied to provide visual explanations highlighting the regions responsible for the model’s decision.

---

## 🎯 Key Features
- 📤 Upload histopathology images
- 🤖 CNN-based classification (Benign / Malignant)
- 📊 Confidence score visualization
- 🔥 Grad-CAM heatmap generation
- 🧾 Explainable AI for model transparency
- 🌐 Interactive web interface (Flask)

---

## 🧰 Tech Stack

### 🔹 Backend
- Python
- Flask

### 🔹 Machine Learning
- TensorFlow
- Keras
- OpenCV
- Grad-CAM

### 🔹 Frontend
- HTML
- CSS
- Bootstrap

---

## 📂 Project Structure
XAI_Breast_Cancer/
│
├── app.py
├── config.py
├── model_utils.py
├── gradcam_utils.py
├── requirements.txt
├── README.md
│
├── templates/
│ ├── index.html
│ ├── result.html
│ ├── explain.html
│ ├── metrics.html
│ ├── about.html
│
├── static/
│ ├── uploads/
│ ├── heatmaps/
│ ├── overlays/
│ ├── confusion_matrix.png
│ ├── roc_curve.png


---

## 📊 Model Performance

| Metric    | Value   |
|----------|--------|
| Accuracy | 80.55% |
| Precision| 82.13% |
| Recall   | 91.61% |
| F1 Score | 86.61% |

---

## 📁 Dataset

- **BreakHis Dataset (Breast Cancer Histopathological Database)**
- Source:  
  https://www.kaggle.com/datasets/ambarish/breakhis

---

## ⚠️ Model File

The trained model file (`.h5`) is **not included** in this repository due to GitHub size limitations.

👉 To run the project:
- Place the trained model file in the project root directory  
- Contact the author if access is required  

---

## 🚀 How to Run

### 1️⃣ Install dependencies

---

## 📊 Model Performance

| Metric    | Value   |
|----------|--------|
| Accuracy | 80.55% |
| Precision| 82.13% |
| Recall   | 91.61% |
| F1 Score | 86.61% |

---

## 📁 Dataset

- **BreakHis Dataset (Breast Cancer Histopathological Database)**
- Source:  
  https://www.kaggle.com/datasets/ambarish/breakhis

---

## ⚠️ Model File

The trained model file (`.h5`) is **not included** in this repository due to GitHub size limitations.

👉 To run the project:
- Place the trained model file in the project root directory  
- Contact the author if access is required  

---

## 🚀 How to Run

### 1️⃣ Install dependencies
pip install -r requirements.txt

### 2️⃣ Add model file
Place the trained `.h5` model file in the root directory.

### 3️⃣ Run the application
python app.py

### 4️⃣ Open in browser


---

## 🧠 Explainability (Grad-CAM)

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize the regions of the input image that contributed most to the prediction.

- 🔴 Red/Yellow → High importance  
- 🔵 Blue → Low importance  

This enhances trust and interpretability in medical AI systems.

---

## 📌 Future Improvements
- Deploy on cloud (AWS / Render)
- Add multi-class classification
- Improve model accuracy
- Add user authentication
- Integrate database support

---

## 👨‍💻 Author

**Tanishq Anand**

---

## ⭐ Acknowledgements
- TensorFlow & Keras Documentation  
- Grad-CAM Research Paper  
- BreakHis Dataset Contributors  

---

## 📜 License
This project is for academic and educational purposes.
