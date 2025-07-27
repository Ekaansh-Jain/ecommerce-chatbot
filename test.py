from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np
import pickle

# ✅ Set correct relative path
MODEL_PATH = "model/intent_bert_models"
LABEL_ENCODER_PATH = "model/intent_bert_models/label_encoder.pkl"  # adjust if it's elsewhere

# Load model and tokenizer from local folder
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# Load label encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Prediction function
def predict_intent(text):
    inputs = tokenizer(
        text,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1)
    pred_class = np.argmax(probs, axis=1).item()
    confidence = probs[0][pred_class].numpy()

    intent = label_encoder.inverse_transform([pred_class])[0]
    print(f"\n📝 Input: {text}")
    print(f"🔮 Predicted Intent: {intent}")
    print(f"📊 Confidence: {confidence:.4f}")

# Run test
if __name__ == "__main__":
    predict_intent("I need to cancel my order because I selected the wrong product.")