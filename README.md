# 🛍️ E-commerce Customer Support Chatbot

A smart, real-time NLP-based chatbot designed to handle common customer support queries in the e-commerce domain. It classifies queries into **intents** and **sub-intents**, enabling faster and more precise support automation.

---

## 🚀 Features

- Intent classification (e.g., `cancel_order`, `return_refund`, `order_status`)
- Sub-intent classification (e.g., `cancel_cod_order`, `refund_not_received`)
- Custom-trained models on realistic, augmented e-commerce queries
- Fast predictions via Flask API
- Easily extendable to include backend logic and multi-turn conversations

---

## 🧠 Model Overview

### Intent Classification
- **Approach**: Transformer (BERT)
- **Accuracy**: ~99%
- **Tooling**: BERT Classifier/ Tensorflow / Keras

### Sub-Intent Classification
- **Approach**:TF-IDF + Logistic Regression
- **Accuracy**: ~92%
- **Tooling**: scikit-learn

---

## 🛠️ Tech Stack

- Python
- Flask (for API)
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy

---

