from flask import Flask, request, render_template
import os
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classifier = load_model('KidneyStone_model.h5')

base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

IMG_SIZE = 224

def extract_vgg16_features_from_uploaded_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img.convert('RGB')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Step 1: Preprocess and extract features
    features = extract_vgg16_features_from_uploaded_image(file_path)

    # Step 2: Predict
    prediction = classifier.predict(features)[0][0]

    # Step 3: Return result
    result = "Kidney Stone Detected" if prediction > 0.5 else "Normal"
    return f"Prediction: {result}"

if __name__ == '__main__':
    app.run(debug=True)
