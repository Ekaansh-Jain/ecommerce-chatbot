from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np
import pickle

# === Set local model paths ===
MODEL_PATH = "model/intent_bert_models"
LABEL_ENCODER_PATH = "model/intent_bert_models/label_encoder.pkl"

# === Load model and tokenizer from local directory ===
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# === Load label encoder (used during training to encode labels) ===
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# === Prediction function ===
def predict_main_intent(text):
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # Run model prediction
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1)

    # Get predicted class and confidence
    pred_class = np.argmax(probs, axis=1).item()
    confidence = probs[0][pred_class].numpy()

    # Decode class label
    intent = label_encoder.inverse_transform([pred_class])[0]

    return intent, confidence