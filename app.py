from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import torch
import clip
import requests
import json
import re
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, supports_credentials=True)

load_dotenv()
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

# Załaduj model CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Lista etykiet do wykrycia
labels = ["keyboard", "mechanical switch", "PCB", "tools", "keycaps", "mechanical keyboard stabilizer "]

print("Hello world!")
print("api key = " + TOGETHER_API_KEY)

@app.route("/classify", methods=["POST"])
def classify_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))  # Wczytaj obraz

    # Przetworzenie obrazu
    image = preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(labels).to(device)

    # Predykcja
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        probs = (image_features @ text_features.T).softmax(dim=-1)

    # Wynik w formie JSON
    result = {label: float(prob) for label, prob in zip(labels, probs[0])}
    return jsonify(result)


@app.route("/suggest-build", methods=["POST"])
def suggest_build():
    data = request.get_json()
    layout = data.get("layout")
    switchType = data.get("switchType")
    soundProfile = data.get("soundProfile")
    budget = data.get("budget")

    prompt = f"""
    You are a mechanical keyboard expert. I want you to suggest me a build for a {layout} layout with {switchType} switches. The sound should be {soundProfile}. The budget is {budget}. Please provide me with the best options available.

    Keep the response short and concise. Give a short intro and description of the build. Then list the components. 

    Make sure all the components are compatible with each other.
    Make sure all the components are within the budget.

    Return a JSON object with the following keys
    "layout:", "soundProfile", "switchType", "budget", "description", "intro", "totalPrice", and "components" which contain: case, pcb, swtiches, keycaps, stabilizers.

    Always keep the response in this format, don't any extra keys or valuse to the JSON object. Return only valid JSON and nothing else. Do not include explanations, markdown, or additional comments after the JSON.

    """

    together_response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # good mix of speed + quality
            "messages": [
                {"role": "system", "content": "You are a keyboard build expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }
    )

    result = together_response.json()
    raw_response = result["choices"][0]["message"]["content"]
     # Step 1: Use regex to extract JSON object (greedy match)
    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    if not json_match:
        return jsonify({"error": "No JSON object found", "raw": raw_response})

    try:
        json_data = json.loads(json_match.group(0))
        return jsonify(json_data)
    except Exception as e:
        return jsonify({"error": f"JSON parsing failed: {str(e)}", "raw": raw_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway ustawia PORT
    app.run(host="0.0.0.0", port=port)
