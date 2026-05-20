# 🐍 Snake Classifier — Deep Learning Web App

An advanced **Snake Species Classification System** built using **PyTorch**, **EfficientNetV2-L**, and **Streamlit**.

The model identifies **25 snake species** from uploaded images and predicts whether the snake is **venomous or non-venomous** using a modern deep learning pipeline with **Test-Time Augmentation (TTA)**.

---

# 🚀 Live Demo

🌐 **Streamlit App:**  
Add your deployed Streamlit URL here

Example:

```text
https://snake-classifier.streamlit.app
```

---

# 📸 Preview

Add screenshots here after deployment.

Example:

```md
![App Screenshot](assets/demo.png)
```

---

# ✨ Features

✅ Classifies **25 snake species**  
✅ Detects **venomous vs non-venomous** snakes  
✅ Built with **EfficientNetV2-L**  
✅ Uses **Test-Time Augmentation (TTA)**  
✅ Beautiful modern Streamlit UI  
✅ Top-5 prediction probabilities  
✅ Confidence visualization chart  
✅ Automatic model download from Google Drive  
✅ Responsive web interface  

---

# 🧠 Deep Learning Details

| Component | Details |
|---|---|
| Architecture | EfficientNetV2-L |
| Framework | PyTorch |
| Classes | 25 Snake Species |
| Accuracy | 87.38% (TTA) |
| Image Size | 384 × 384 |
| Parameters | 117M |
| Loss Function | Focal Loss + Label Smoothing |
| Augmentations | Mixup + TTA |
| Frontend | Streamlit |

---

# 🐍 Supported Snake Species

The model supports classification of 25 species including:

- Indian Cobra
- Spectacled Cobra
- Common Krait
- Russell’s Viper
- Saw Scaled Viper
- Caspian Cobra
- Himalayan Pit Viper
- Blunt Nosed Viper
- Saharan Horned Viper
- and more...

---

# ⚠️ Venomous Species Detection

The app also flags dangerous species as:

- 🔴 **VENOMOUS**
- 🟢 **NON-VENOMOUS**

> ⚠️ This application is for educational and research purposes only.  
> Never rely solely on AI for real-world snake identification or medical emergencies.

---

# 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| Python | Backend |
| PyTorch | Deep Learning |
| timm | Model Architecture |
| Streamlit | Web App |
| Matplotlib | Visualization |
| PIL | Image Processing |
| NumPy | Numerical Operations |
| gdown | Google Drive Model Download |

---

# 📂 Project Structure

```text
Snake-Classifier/
│
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   └── demo.png
└── .streamlit/
    └── config.toml
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/snake-classifier.git
cd snake-classifier
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

```bash
streamlit run app.py
```

The app will open at:

```text
http://localhost:8501
```

---

# ☁️ Deploy on Streamlit Cloud

## Step 1 — Push to GitHub

Upload these files:

- `app.py`
- `requirements.txt`
- `README.md`

---

## Step 2 — Open Streamlit Cloud

Go to:

```text
https://share.streamlit.io/
```

---

## Step 3 — Deploy

- Select GitHub repository
- Choose branch: `main`
- Main file path:

```text
app.py
```

Click:

```text
Deploy
```

---

# 📦 Model Weights

The trained model weights are downloaded automatically from Google Drive using `gdown`.

```python
GDRIVE_ID = "1EHdaSKNH8ZwuY61wQQgFeVsucyy2f2d4"
```

---

# 📊 Example Output

The app provides:

- Predicted species
- Confidence score
- Venomous classification
- Top-5 predictions
- Confidence bar chart

---

# 🎯 Future Improvements

- Grad-CAM visualization
- Mobile optimization
- Faster inference models
- Snake information database
- Multi-language support
- Real-time camera detection

---

# 👨‍💻 Author

## Laraib Asif

🌐 GitHub:  
[Github link](https://github.com/laraibjutt071)

💼 LinkedIn:  
[LinkedIn link](https://www.linkedin.com/in/laraib-asif-48575a313/)

📊 Kaggle:  
[Kaggle link](https://www.kaggle.com/laraibjutt)

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Acknowledgements

- PyTorch
- Streamlit
- timm Library
- EfficientNetV2 Research
- Open-source Deep Learning Community

---

# ⚠️ Disclaimer

This project is intended strictly for:

- educational purposes
- AI research
- computer vision learning

It should NOT be used for:

- medical decisions
- wildlife handling
- emergency identification

Always consult trained wildlife professionals and medical experts.
