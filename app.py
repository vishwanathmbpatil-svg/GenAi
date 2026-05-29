import os, json, time, re
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from openai import OpenAI
from PIL import Image
import io

app = Flask(__name__)

# ── Groq API client ───────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ── Load models ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

corn_model      = load_model(os.path.join(BASE, "corn_model.h5"))
sugarcane_model = load_model(os.path.join(BASE, "sugarcane_model.h5"))

with open(os.path.join(BASE, "corn_model_classes.json"))      as f: corn_classes      = json.load(f)
with open(os.path.join(BASE, "sugarcane_model_classes.json")) as f: sugarcane_classes = json.load(f)

IMG_SIZE     = (128, 128)
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTS

def preprocess(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(img_to_array(img) / 255.0, axis=0)

def predict(model, classes, img_arr):
    preds = model.predict(img_arr)[0]
    idx   = int(np.argmax(preds))
    return classes[str(idx)], float(preds[idx])

def sanitize(text):
    return re.sub(r"[<>]", "", text)

def openai_call(messages, json_mode=False, retries=3):
    for i in range(retries):
        try:
            kwargs = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 600,
                "temperature": 0.7
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
                
            response = client.chat.completions.create(**kwargs)
            return sanitize(response.choices[0].message.content)
        except Exception as e:
            if "429" in str(e) and i < retries - 1:
                time.sleep(10 * (i + 1))
            else:
                raise

def get_treatment(plant, disease):
    if disease.lower() == "healthy":
        return {
            "explanation": "The plant looks healthy! No disease detected.",
            "treatments": "No treatment needed. Maintain regular watering and fertilization.",
            "precautions": "Observe crop growth regularly and ensure clean water supply."
        }
        
    messages = [
        {"role": "system", "content": (
            "You are an expert agricultural assistant specializing in plant diseases. "
            "You must return a JSON object with exactly three keys: "
            "'explanation' (a brief 2-line explanation of what the disease is), "
            "'treatments' (specific chemical and organic control methods with exact dosages), and "
            "'precautions' (preventive actions and precautions for the farmer to avoid spread or future infection). "
            "Make sure the response is a valid JSON object."
        )},
        {"role": "user", "content": f"Analyze a {plant} plant with '{disease}' disease."}
    ]
    
    response_text = openai_call(messages, json_mode=True)
    try:
        return json.loads(response_text)
    except Exception as e:
        return {
            "explanation": f"The {plant} plant has been diagnosed with {disease}.",
            "treatments": response_text,
            "precautions": "Consult local agricultural extension."
        }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Upload JPG, PNG or WEBP."}), 400

        plant_type = request.form.get("plant_type", "corn")
        file_bytes = file.read()
        img_arr    = preprocess(file_bytes)

        if plant_type == "sugarcane":
            disease, confidence = predict(sugarcane_model, sugarcane_classes, img_arr)
        else:
            disease, confidence = predict(corn_model, corn_classes, img_arr)

        treatment = get_treatment(plant_type, disease)

        return jsonify({
            "plant":      plant_type,
            "disease":    disease,
            "confidence": round(confidence * 100, 2),
            "treatment":  treatment,
            "is_diseased": disease.lower() != "healthy"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = data.get("message", "").strip()
    history = data.get("history", [])
    plant   = data.get("plant", "").strip()
    disease = data.get("disease", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        system_content = (
            "You are a helpful agricultural assistant specializing in plant diseases. "
            "Answer questions about leaf diseases, treatments, farming practices, and plant health. "
            "Be concise and practical."
        )
        if plant and disease:
            system_content += f" The user is asking about a {plant} plant diagnosed with {disease}."

        messages = [{"role": "system", "content": system_content}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        reply = sanitize(response.choices[0].message.content)
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply, "history": history})
    except Exception as e:
        if "429" in str(e):
            return jsonify({"error": "Rate limit reached. Please wait a moment and try again."}), 429
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")
