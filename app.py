from flask import Flask, request, jsonify
from utils.main_intent_router import predict_main_intent  # ✅ Use from utils
from utils.subintent_router import route_subintent
# from utils.fallback import fallback_handler  # Optional

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get("user_input", "")

    # ✅ Correctly unpack the tuple returned by predict_main_intent
    predicted_intent, confidence = predict_main_intent(user_input)

    response = {
        "intent": predicted_intent,
        "confidence": float(confidence)
    }

    # Sub-intents for complex intents
    if predicted_intent in ["cancel_order", "return_refund", "payment_issue", "order_status"]:
        sub_intent = route_subintent(predicted_intent, user_input)
        response["sub_intent"] = sub_intent
        response["message"] = f"Sub-intent '{sub_intent}' triggered for main intent '{predicted_intent}'."

    # Direct responses for simple intents
    elif predicted_intent == "greeting":
        response["message"] = "Hello! How can I assist you today?"

    elif predicted_intent == "goodbye":
        response["message"] = "Goodbye! Feel free to reach out again."

    elif predicted_intent == "language_change":
        response["message"] = "Please tell me which language you'd like to switch to."

    elif predicted_intent == "shopping_list":
        response["message"] = "Here is your current shopping list. (feature coming soon)"

    elif predicted_intent == "complaint":
        response["message"] = "Would you like to raise a ticket or get a callback from support?"

    elif predicted_intent == "fraud_report":
        response["message"] = "We're sorry about this. Would you like us to connect you to the fraud resolution team or raise a formal report?"

    elif predicted_intent == "product_query":
        response["message"] = "Please specify your product question. (feature coming soon)"

    # Optional fallback
    # else:
    #     response = fallback_handler(user_input)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5051)