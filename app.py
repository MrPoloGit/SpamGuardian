import os
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("MODEL_PATH", "./model")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return {
        "spam": bool(pred == 1),
        "label": "spam" if pred == 1 else "ham",
        "confidence": round(confidence, 4),
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/classify", methods=["POST"])
def classify_route():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing required field: text"}), 400
    text = data["text"]
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' must be a non-empty string"}), 400
    return jsonify(classify_text(text))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
