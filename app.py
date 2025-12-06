from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
from symspellpy import SymSpell
import language_tool_python
from test import ImageToWordModel
from mltu.configs import BaseModelConfigs
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Load model and config
configs = BaseModelConfigs.load("Models/04_lines_recognition/202505162357/configs.yaml")
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

# Load LanguageTool once
tool = language_tool_python.LanguageTool('en-US')

# Load SymSpell once
sym_spell = SymSpell(max_dictionary_edit_distance=2)
dictionary_path = "frequency_dictionary_en_500_000.txt"
if not os.path.exists(dictionary_path):
    raise FileNotFoundError(f"Dictionary file '{dictionary_path}' not found.")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding="utf-8")

# --------- Text Correction Function ---------
def correct_text(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    corrected = suggestions[0].term if suggestions else text
    matches = tool.check(corrected)
    return language_tool_python.utils.correct(corrected, matches)

# --------- Preprocessing ---------
def preprocess_and_remove_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    mask = cv2.dilate(detected_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    inpainted = cv2.inpaint(gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted

# --------- Line Segmentation ---------
def segment_lines_from_paragraph(img, padding=15, min_line_height=10):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilated = cv2.dilate(binary, vertical_kernel, iterations=1)
    horizontal_projection = np.sum(dilated, axis=1)
    threshold = np.max(horizontal_projection) * 0.1
    lines = []
    start = None
    for i, val in enumerate(horizontal_projection):
        if val > threshold and start is None:
            start = i
        elif val <= threshold and start is not None:
            end = i
            if end - start > min_line_height:
                lines.append((start, end))
            start = None
    if start is not None and (img.shape[0] - start > min_line_height):
        lines.append((start, img.shape[0]))
    line_images = []
    for y1, y2 in lines:
        y1_pad = max(0, y1 - padding)
        y2_pad = min(img.shape[0], y2 + padding)
        line_img = img[y1_pad:y2_pad, :]
        line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR)
        line_images.append(line_img)
    return line_images

# --------- Routes ---------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/manifest.webmanifest")
def manifest():
    return send_from_directory('.', 'manifest.webmanifest', mimetype='application/manifest+json')

@app.route("/sw.js")
def service_worker():
    return send_from_directory('.', 'sw.js', mimetype='application/javascript')

@app.route("/favicon.ico")
def favicon():
    return send_from_directory('.', 'favicon.ico', mimetype='image/x-icon')

@app.route("/android-chrome-192x192.png")
def icon192():
    return send_from_directory('.', 'android-chrome-192x192.png', mimetype='image/png')

@app.route("/android-chrome-512x512.png")
def icon512():
    return send_from_directory('.', 'android-chrome-512x512.png', mimetype='image/png')

@app.route("/apple-touch-icon.png")
def apple_icon():
    return send_from_directory('.', 'apple-touch-icon.png', mimetype='image/png')

@app.route("/predict", methods=["POST"])
def predict():
    request.get_data()
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image file."}), 400

    try:
        original_size = f"{image.shape[1]}x{image.shape[0]}"
        cleaned_image = preprocess_and_remove_lines(image)
        line_images = segment_lines_from_paragraph(cleaned_image)

        # Use ThreadPoolExecutor for faster prediction
        with ThreadPoolExecutor() as executor:
            raw_predictions = list(executor.map(model.predict, line_images))

        full_raw_text = "\n".join(raw_predictions)
        corrected_text = correct_text(full_raw_text)

        return jsonify({
            "raw_prediction": full_raw_text,
            "corrected_prediction": corrected_text,
            "line_wise_predictions": raw_predictions,
            "processing_info": {
                "original_size": original_size,
                "line_count": len(line_images)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


