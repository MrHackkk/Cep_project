import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os

class PPEDetectionModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = (224, 224)
        self.class_names = [
            'person', 'helmet', 'safety_vest', 'gloves', 'goggles',
            'no_helmet', 'no_vest', 'no_gloves'
        ]
    
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image {image_path} not found.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)
        image = image.astype('float32') / 255.0
        return np.expand_dims(image, axis=0)
    
    def predict(self, image_path):
        img = self.preprocess_image(image_path)
        preds = self.model.predict(img)
        class_id = preds.argmax(axis=-1)[0]
        confidence = preds[0][class_id]
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
        return class_name, confidence

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = "ppe_model/best_model.h5"
detector = PPEDetectionModel(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            class_name, confidence = detector.predict(filepath)
            return render_template(
                'detection_result.html',  # updated template name
                image_url=url_for('static', filename=f'uploads/{file.filename}'),
                class_name=class_name,
                confidence=confidence
            )
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
