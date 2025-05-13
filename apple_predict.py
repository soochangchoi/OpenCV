from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
from PIL import Image
from io import BytesIO

app = Flask(__name__)
model = joblib.load("svm_model.pkl")  # 업로드한 SVM 모델 사용

def preprocess_image(file):
    image = Image.open(file).convert("RGB")
    image = np.array(image)
    rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), 345, 1), (image.shape[1], image.shape[0]))
    flipped = cv2.flip(rotated, 1)
    resized = cv2.resize(flipped, (64, 64))
    flat = resized.flatten().reshape(1, -1)
    return flat

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image_data = preprocess_image(file)
    prediction = model.predict(image_data)[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
