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
valid_classifier = load_model('ValidClassifier.h5')

base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

IMG_SIZE = 224

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE)).convert('RGB')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return feature_extractor.predict(img_array)

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', result="No file uploaded.")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', result="No selected file.")

    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        features = extract_features(file_path)

        is_valid = valid_classifier.predict(features)[0][0]
        if is_valid > 0.5:
            result = "Invalid image. Please upload a valid CT scan."
        else:
            prediction = classifier.predict(features)[0][0]
            result = "Kidney Stone Detected" if prediction > 0.5 else "Normal"

        return render_template('index.html', result=result)

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/privacypolicy')
def privacypolicy():
    return render_template('privacypolicy.html')

@app.route('/termscondition')
def termscondition():
    return render_template('termscondition.html')

@app.route('/contactus')
def contactus():
    return render_template('contactus.html')

if __name__ == '__main__':
    app.run(debug=True)
