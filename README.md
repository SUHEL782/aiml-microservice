# AIML Microservice

This project is an **end-to-end ML microservice** that trains a model on tabular data and exposes predictions through a FastAPI REST API. It is designed to be production-ready with Docker and CI/CD support.

---

## Features
- Train a classification model (RandomForest with preprocessing)
- Save the trained model
- Serve predictions via FastAPI
- Docker and CI/CD ready

---

## Training the Model
```bash
cd training
python train.py
