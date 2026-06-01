import os, json, time, re, base64
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from openai import OpenAI
from PIL import Image
import io

app = Flask(__name__)

# ── Groq API client ───────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ── Load models lazily ────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

_models = {}
_classes = {}

def get_model(plant_type):
    if plant_type not in _models:
        _models[plant_type] = load_model(os.path.join(BASE, f"{plant_type}_model.h5"))
        with open(os.path.join(BASE, f"{plant_type}_model_classes.json")) as f:
            _classes[plant_type] = json.load(f)
    return _models[plant_type], _classes[plant_type]

IMG_SIZE     = (128, 128)
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_leaf_image(file_bytes, plant_type="corn"):
    try:
        img_b64 = base64.b64encode(file_bytes).decode("utf-8")
        plant_name = "corn/maize" if plant_type == "corn" else "sugarcane"
        other_plant = "sugarcane" if plant_type == "corn" else "corn/maize"
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a strict plant identification expert. "
                    f"Your only job is to check if the image is a {plant_name} leaf. "
                    f"Corn/maize leaves are wide, flat, long with a prominent midrib. "
                    f"Sugarcane leaves are long, narrow, grass-like with a white midrib and hairy edges. "
                    f"If the image is a {other_plant} leaf or any non-{plant_name} object, you must reply NO. "
                    f"Reply with exactly one word only: YES or NO."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Is this a {plant_name} leaf? Reply YES or NO only."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    }
                ]
            }
        ]
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            max_tokens=5,
            temperature=0.0
        )
        result = response.choices[0].message.content.strip().upper()
        result = "".join(c for c in result if c.isalpha())
        print(f"[Plant check] plant_type={plant_type} | LLM replied: '{result}'")
        return result == "YES"
    except Exception as e:
        print("Leaf verification failed:", str(e))
        return True

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

def openai_call(messages, json_mode=False, retries=2):
    for i in range(retries):
        try:
            kwargs = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 400,
                "temperature": 0.7,
                "timeout": 25
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            return sanitize(response.choices[0].message.content)
        except Exception as e:
            if "429" in str(e) and i < retries - 1:
                time.sleep(3)
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

        plant_type  = request.form.get("plant_type", "corn")
        plant_label = "corn/maize" if plant_type == "corn" else "sugarcane"
        file_bytes  = file.read()
        img_arr     = preprocess(file_bytes)

        # Run leaf check and CNN prediction in parallel
        with ThreadPoolExecutor(max_workers=2) as ex:
            leaf_future    = ex.submit(is_leaf_image, file_bytes, plant_type)
            model, classes = get_model(plant_type)
            predict_future = ex.submit(predict, model, classes, img_arr)
            is_valid = leaf_future.result()
            if not is_valid:
                return jsonify({
                    "error": f"The uploaded image does not appear to be a {plant_label} leaf. Please upload a correct {plant_label} leaf image."
                }), 400
            disease, confidence = predict_future.result()

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
