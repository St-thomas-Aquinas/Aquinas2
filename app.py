from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging

logging.set_verbosity_error()  # suppress HF warnings

app = Flask(__name__)

# --- Load latest model from Hugging Face ---
hf_model_id = "ST-THOMAS-OF-AQUINAS/SCAM"  # Replace with your Hugging Face repo
tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example label map (update with your labels)
label_map = {"author1": 0, "author2": 1}


def predict_author(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    predicted_author = list(label_map.keys())[list(label_map.values()).index(pred)]
    return predicted_author, round(confidence * 100, 2)


# --- Twilio Webhook ---
@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "").strip()
    resp = MessagingResponse()

    if incoming_msg:
        author, confidence = predict_author(incoming_msg)
        reply = f"Prediction: {author}\nConfidence: {confidence}%"
    else:
        reply = "No text detected."

    resp.message(reply)
    return str(resp)


# --- Health check route ---
@app.route("/")
def home():
    return "✅ Twilio + HuggingFace WhatsApp bot is running on Render!"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render requires $PORT
    app.run(host="0.0.0.0", port=port)
