# 🌿 Leaf Disease Detector & Treatment Advisor

GenAI-powered web app that detects leaf diseases and generates treatment recommendations using Google Gemini.

## Supported Plants & Diseases

| Plant      | Diseases                                          |
|------------|---------------------------------------------------|
| Corn/Maize | Blight, Common_Rust, Gray_Leaf_Spot, Healthy      |
| Sugarcane  | Mosaic, RedRot, Rust, Yellow, Healthy             |

## Project Structure
```
genai_t03code/
├── leaf_disease_detection.ipynb   ← Main notebook (train + predict + GenAI)
├── app.py                         ← Flask web app
├── templates/index.html           ← Web UI
├── requirements.txt
├── corn_model.keras               ← Generated after training
├── sugarcane_model.keras          ← Generated after training
├── corn_classes.json
└── sugarcane_classes.json
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key
https://aistudio.google.com/app/apikey

### 3. Run the Notebook (train models + test predictions)
```bash
jupyter notebook leaf_disease_detection.ipynb
```
Run all blocks in order. Models saved as `corn_model.keras` and `sugarcane_model.keras`.

### 4. Run the Web App
```bash
set GEMINI_API_KEY=your_api_key_here
py app.py
```
Open http://localhost:5000
