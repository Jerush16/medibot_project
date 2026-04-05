# 🏥 Medibot – AI Health Assistant

## 🚀 Overview

Medibot is an AI-powered healthcare chatbot that provides basic medical guidance through WhatsApp and SMS.
It is designed for users in rural and low-connectivity environments where access to immediate healthcare is limited.

---

## 🎯 Problem

Limited access to doctors, poor internet connectivity, and language barriers make it difficult for many people to get quick medical advice.

---

## 💡 Solution

Medibot analyzes user symptoms using trained machine learning models and provides basic health suggestions via messaging platforms.

---

## 🔥 Features

* Symptom-based disease prediction
* Works via WhatsApp and SMS
* Lightweight and fast
* Designed for low internet usage
* User-friendly interaction

---

## 🛠 Tech Stack

* Python
* Flask
* Machine Learning
* Twilio API
* Ngrok

---

## 📂 Project Structure

medibot-ai-health-assistant/
│
├── app.py
├── train_model_blood.py
├── train_model_disease.py
├── train_model_symptom.py
├── data/
├── requirements.txt
└── README.md

---

## ⚙️ Setup & Run

### Install dependencies

pip install -r requirements.txt

### Run the application

python app.py

---

## 🧠 Model Training

Run the following files to generate models:

* train_model_blood.py
* train_model_disease.py
* train_model_symptom.py

---

## ⚠️ Note

Trained model files are not included due to size limitations.
Run the training scripts to generate them locally.

---

## 🌍 Use Case

Useful for providing initial medical guidance in rural areas, emergencies, or low-network conditions.

---

## 🚧 Future Improvements

* More accurate AI predictions
* Location-based hospital suggestions
* Emergency alert system
* Voice interaction

---
