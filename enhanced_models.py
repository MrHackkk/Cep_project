import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import cv2

class EnhancedPPEDetector:
    def __init__(self, num_classes=9, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.class_names = [
            'person', 'helmet', 'safety_vest', 'gloves', 'safety_glasses',
            'no_helmet', 'no_vest', 'no_gloves', 'no_glasses'
        ]
        
    def create_enhanced_model(self):
        """Create an enhanced model with better architecture for PPE detection"""
        # Use EfficientNetB0 as base model for better accuracy
        base_model = EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze early layers, fine-tune later layers
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Add custom head for PPE detection
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Add dropout for regularization
        x = layers.Dropout(0.3)(x)
        
        # Dense layers with batch normalization
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        self.model = model
        return model
    
    def create_mobilenet_model(self):
        """Create MobileNetV2 based model for faster inference"""
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate optimizer and loss function"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return self.model
    
    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1),
        ])
        return data_augmentation
    
    def preprocess_image(self, image_path, augment=False):
        """Preprocess image for model input"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.input_shape[:2])
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_with_confidence(self, image_path, threshold=0.5):
        """Make prediction with confidence scores"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[-4:][::-1]
            results = {}
            
            for idx in top_indices:
                class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
                confidence = float(predictions[0][idx])
                if confidence > threshold:
                    results[class_name] = confidence
            
            return results
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {}
    
    def analyze_safety_compliance(self, predictions):
        """Analyze safety compliance based on predictions"""
        safety_items = {
            'helmet': {'name': 'Hard Hat', 'required': True},
            'safety_vest': {'name': 'Safety Vest', 'required': True},
            'gloves': {'name': 'Safety Gloves', 'required': True},
            'safety_glasses': {'name': 'Safety Glasses', 'required': True}
        }
        
        compliance = {}
        missing_items = []
        present_items = []
        
        for item, data in safety_items.items():
            is_present = item in predictions and predictions[item] > 0.6
            is_missing = f"no_{item}" in predictions and predictions[f"no_{item}"] > 0.6
            
            if is_present:
                present_items.append({
                    'name': data['name'],
                    'confidence': predictions[item],
                    'status': 'present'
                })
                compliance[item] = True
            elif is_missing or not is_present:
                missing_items.append({
                    'name': data['name'],
                    'confidence': predictions.get(f"no_{item}", 0.8),
                    'status': 'missing'
                })
                compliance[item] = False
        
        return {
            'compliance': compliance,
            'present_items': present_items,
            'missing_items': missing_items,
            'is_compliant': len(missing_items) == 0,
            'compliance_percentage': (len(present_items) / len(safety_items)) * 100
        }

class PPEDataPreprocessor:
    def __init__(self, dataset_path, image_size=(224, 224)):
        self.dataset_path = dataset_path
        self.image_size = image_size
    
    def create_tensorflow_dataset(self, batch_size=32, validation_split=0.2):
        """Create TensorFlow dataset with proper preprocessing"""
        def load_and_preprocess_image(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.image_size)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        
        def augment_image(image, label):
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            # Random rotation
            image = tf.image.random_brightness(image, 0.1)
            # Random contrast
            image = tf.image.random_contrast(image, 0.9, 1.1)
            return image, label
        
        datasets = {}
        
        for split in ['train', 'validation', 'test']:
            image_dir = f"{self.dataset_path}/{split}/images"
            if not os.path.exists(image_dir):
                continue
            
            image_paths = []
            labels = []
            
            for filename in os.listdir(image_dir):
                if filename.endswith('.jpg'):
                    image_paths.append(os.path.join(image_dir, filename))
                    class_id = int(filename.split('_')[-1].split('.')[0])
                    labels.append(class_id)
            
            if not image_paths:
                continue
            
            # Create dataset
            path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
            label_ds = tf.data.Dataset.from_tensor_slices(labels)
            dataset = tf.data.Dataset.zip((path_ds, label_ds))
            dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            
            # Apply augmentation for training set
            if split == 'train':
                dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.shuffle(1000)
                dataset = dataset.repeat()
            
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            datasets[split] = dataset
        
        return datasets

def create_callbacks(save_path="enhanced_ppe_model"):
    """Create training callbacks"""
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"{save_path}/best_model.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(f"{save_path}/training_log.csv")
    ]
    return callbacks

if __name__ == "__main__":
    print("Enhanced PPE Detection Models")
    print("=" * 40)
    
    # Test model creation
    detector = EnhancedPPEDetector()
    model = detector.create_enhanced_model()
    print(f"✅ Enhanced model created with {model.count_params():,} parameters")
    
    # Test MobileNet model
    mobilenet_detector = EnhancedPPEDetector()
    mobilenet_model = mobilenet_detector.create_mobilenet_model()
    print(f"✅ MobileNet model created with {mobilenet_model.count_params():,} parameters")
    
    print("Models ready for training!")
