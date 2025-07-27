import joblib
import os

# Load sub-intent models
model_paths = {
    "cancel_order": "model/subintent_models/cancel_sub_intent.pkl",
    "return_refund": "model/subintent_models/return_refund_sub_intent.pkl",
    "payment_issue": "model/subintent_models/payment_issue_model.pkl",
    "order_status": "model/subintent_models/order_status.pkl"
}

encoder_paths = {
    "cancel_order": "model/subintent_models/cancel_sub_intent_label_encoder.pkl",
    "return_refund": "model/subintent_models/return_refund_sub_intent_encoder.pkl",
    "payment_issue": "model/subintent_models/payment_issue_model_encoder.pkl",
    "order_status": "model/subintent_models/order_status_encoder.pkl"
}

# Load all sub-intent models and vectorizers
subintent_models = {
    intent: joblib.load(path) for intent, path in model_paths.items() if os.path.exists(path)
}

subintent_encoders = {
    intent: joblib.load(path) for intent, path in encoder_paths.items() if os.path.exists(path)
}

def route_subintent(main_intent, user_input):
    if main_intent not in subintent_models or main_intent not in subintent_encoders:
        return "unknown_subintent"

    model = subintent_models[main_intent]
    encoder = subintent_encoders[main_intent]

    # Predict and decode sub-intent
    encoded_pred = model.predict([user_input])[0]
    subintent = encoder.inverse_transform([encoded_pred])[0]

    return subintent