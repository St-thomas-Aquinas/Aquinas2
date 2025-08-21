from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging

logging.set_verbosity_error()  # suppress HF warnings

app = Flask(__name__)

# --- Load model from Hugging Face Hub ---
hf_model_id = "ST-THOMAS-OF-AQUINAS/SCAM"  # ✅ your uploaded repo
tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example label map (update to match your training labels)
label_map = {0: "author1", 1: "author2"}


def predict_author(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    predicted_author = label_map[pred]
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
        reply = "⚠️ No text detected."

    resp.message(reply)
    return str(resp)


# --- Health check route ---
@app.route("/")
def home():
    return "✅ Twilio + HuggingFace WhatsApp bot is running on Render!"


if __name__ == "__main__":
    # On Render, Flask binds to 0.0.0.0 and PORT from env
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
