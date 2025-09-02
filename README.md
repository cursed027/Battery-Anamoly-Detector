# 🔋 Battery Anomaly Detector  

An end-to-end **ML/DL + MLOps project** for detecting anomalies in battery behavior using **LSTM Autoencoders**.  
This project demonstrates how to go from **model training → API service → Streamlit UI → CI/CD deployment on AWS EC2** 🚀  

---

## ✨ Features  
- 🧠 **LSTM Autoencoder** for anomaly detection on battery cycles  
- 📊 Preprocessing pipeline (scaling + sequence creation)  
- ⚡ REST API using **FastAPI**  
- 🎨 Interactive **Streamlit UI**  
- 🐳 Dockerized setup for portability  
- 🔄 Automated **CI/CD with GitHub Actions** → builds & deploys to **AWS EC2**  
- 🔑 Secrets managed via GitHub Secrets (secure deployment)  

---

## 🏗️ Project Structure  
```
  Battery-Anamoly-Detector/
  │── app/
  │ ├── fastapi_app.py # FastAPI backend (model inference API)
  │ ├── streamlit_app.py # Streamlit frontend (UI)
  │
  │── src/
  │ ├── model.py # LSTM Autoencoder definition
  │ ├── train.py # Training pipeline
  │ ├── utils.py # Helper functions
  │ ├── config.py # Configs (sequence length, paths, features, etc.)
  │
  │── artifacts/
  │ ├── best_model.pth # Saved trained model
  │ ├── scaler.pkl # Pre-fitted scaler
  │
  │── Dockerfile # Docker build for API + Streamlit
  │── requirements.txt # Dependencies
  │── .github/workflows/cd.yml # CI/CD pipeline
  │── README.md # Project documentation
```

---

## ⚙️ Workflow  

1️⃣ **Training**  
- Run `train.py` to train the model on battery data.  
- Saves `best_model.pth` & `scaler.pkl` in `artifacts/`.  

2️⃣ **API Service (FastAPI)**  
- Exposes endpoints for anomaly detection.  
- Example: `http://<EC2-IP>:8000/docs`  

3️⃣ **Streamlit UI**  
- Interactive frontend for predictions.  
- Example: `http://<EC2-IP>:8501`  

4️⃣ **Dockerization**  
- Single container runs both FastAPI & Streamlit.  
- Lightweight & portable deployment.  

5️⃣ **CI/CD Deployment**  
- On `push` to GitHub:  
  - 🏗️ Build Docker image on GitHub runner  
  - 📦 Push image → DockerHub  
  - ☁️ Connect to AWS EC2 & deploy container  
  - 🔄 Stops old container & runs latest image  

---

## 🚀 Deployment  

### 🔑 GitHub Secrets Required  
- `DOCKERHUB_USERNAME` → your DokcerHUB username
- `DOCKERHUB_TOKEN`  → your login token , generated from DockerHub
- `EC2_HOST` → your instance IP  
- `EC2_USER` → usually `ubuntu` or else the Cloud OS you choosed 
- `EC2_KEY` → your private SSH/RSA key for connecting this github to cloud machine  

### 🌍 Access After Deployment  
- **FastAPI** → `http://<EC2-IP>:8000/docs`  
- **Streamlit** → `http://<EC2-IP>:8501`  

---

## 📈 Tech Stack  
- **Python** 🐍  
- **PyTorch** 🔥  
- **FastAPI** ⚡  
- **Streamlit** 🎨  
- **Docker** 🐳  
- **GitHub Actions** 🤖  
- **AWS EC2** ☁️  

---

## 🤝 Contributions  
Pull requests are welcome! 🎉  
For major changes, open an issue first to discuss what you’d like to change.  

---

## 📜 License  
MIT License © 2025 Milan Kumar Singh  





