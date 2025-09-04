import os
import io
import random
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# ===== TensorFlow / EfficientNetB3 (CLASSIFY) =====
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# ===== YOLOv8 (DETECT) =====
from ultralytics import YOLO
import cv2

# ================== CẤU HÌNH MODEL ==================
CNN_MODEL_PATH = r"cnn_model\cnn_best_model50_1.h5"
CNN_CLASS_NAMES = ['cardboad', 'glass', 'metal', 'paper', 'plastic', 'trash']

YOLO_MODEL_PATH = r"YOLO_garbage\garbage-yolov8\runs\yolov8s_garbage\weights\best.pt"

# ================== APP ==================
app = Flask(__name__)
CORS(app)

# ================== LOAD MODELS ==================
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_class_names = yolo_model.names

# Màu cho từng class
random.seed(42)
colors = {i: (random.randint(0,255), random.randint(0,255), random.randint(0,255))
          for i in range(len(yolo_class_names))}


# ================== TIỀN XỬ LÝ ==================
def preprocess_image_for_cnn(img: Image.Image, target_size=(300, 300)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img)
    arr = effnet_preprocess(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def read_image_from_request(file_storage):
    raw = file_storage.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    np_bytes = np.frombuffer(raw, np.uint8)
    cv_img = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    return pil_img, cv_img


# ================== SUY LUẬN ==================
def run_classify(pil_img: Image.Image):
    arr = preprocess_image_for_cnn(pil_img)
    preds = cnn_model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    return {
        "label": CNN_CLASS_NAMES[idx] if 0 <= idx < len(CNN_CLASS_NAMES) else str(idx),
        "confidence": round(conf * 100, 2)
    }

def run_detect(cv_img, conf_threshold=0.5):
    """YOLOv8 -> vừa trả về list, vừa vẽ box"""
    result = yolo_model(cv_img, verbose=False)[0]
    outs = []

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        label = yolo_class_names.get(cls_id, str(cls_id))
        outs.append({
            "label": label,
            "confidence": round(conf * 100, 2),
            "box": [x1, y1, x2, y2]
        })

        # Vẽ box
        color = colors[cls_id]
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(cv_img, f"{label} {conf*100:.1f}%", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Encode ảnh kết quả sang base64 để trả về
    _, buffer = cv2.imencode(".jpg", cv_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return outs, img_base64


# ================== ROUTES ==================
@app.route("/predict", methods=["POST"])
def predict():
    mode = request.args.get("mode") or request.form.get("mode") or "classify"
    mode = mode.lower().strip()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        pil_img, cv_img = read_image_from_request(file)

        if mode == "classify":
            result = run_classify(pil_img)
            return jsonify(result)

        elif mode == "detect":
            results, img_base64 = run_detect(cv_img)
            return jsonify({
                "objects": results,
                "image": img_base64  # ảnh có box
            })

        else:
            return jsonify({"error": "Invalid mode"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
