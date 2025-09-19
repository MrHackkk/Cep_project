import tensorflow as tf
import cv2
import numpy as np

class PPEDetectionModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.input_size = (224, 224)
        # Example PPE classes; adjust as per your dataset
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

if __name__ == "__main__":
    model_path = "ppe_model/best_model.h5"
    detector = PPEDetectionModel(model_path)
    
    test_image = r"D:\Cep_Prj\construction_ppe_dataset\run\R1.jpg"
    
    try:
        class_name, confidence = detector.predict(test_image)
        print(f"Predicted class: {class_name} with confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
