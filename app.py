from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request
import requests
import os

app = Flask(__name__)

# --- Hugging Face Inference API ---
HF_MODEL_ID = "ST-THOMAS-OF-AQUINAS/SCAM"  # your model repo
HF_API_KEY = os.environ.get("HF_API_KEY")  # ✅ Load from Render env

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


def predict_author(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Error {response.status_code}", 0

    result = response.json()
    # Example: [{"label": "author1", "score": 0.93}, {"label": "author2", "score": 0.07}]
    if isinstance(result, list) and len(result) > 0:
        pred = max(result, key=lambda x: x["score"])
        return pred["label"], round(pred["score"] * 100, 2)
    else:
        return "Unknown", 0


# --- Twilio Webhook ---
@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "").strip()
    resp = MessagingResponse()

    if incoming_msg:
        author, confidence = predict_author(incoming_msg)
        reply = f"Prediction: {author}\nConfidence: {confidence}%"
    else:
        reply = "⚠️ No text detected."

    resp.message(reply)
    return str(resp)


# --- Health check ---
@app.route("/")
def home():
    return "✅ Twilio + HuggingFace Inference API bot is running on Render!"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT env
    app.run(host="0.0.0.0", port=port, debug=False)
