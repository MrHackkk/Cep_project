<<<<<<< HEAD
import os
import tensorflow as tf

class PPEDataPreprocessor:
    def __init__(self, dataset_path, image_size=(224, 224)):
        self.dataset_path = dataset_path
        self.image_size = image_size
    
    def create_tensorflow_dataset(self):
        def load_and_preprocess_image(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.image_size)
            image = tf.cast(image, tf.float32) / 255.0
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
            
            path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
            label_ds = tf.data.Dataset.from_tensor_slices(labels)
            dataset = tf.data.Dataset.zip((path_ds, label_ds))
            dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            
            if split == 'train':
                dataset = dataset.shuffle(80)
                dataset = dataset.repeat()
            
            dataset = dataset.batch(32)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            datasets[split] = dataset
        
        return datasets


class PPEDetectionModel:
    def __init__(self, num_classes=9, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
    
    def create_mobilenet_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def train(self, train_dataset, val_dataset, epochs=50, save_path="ppe_model", train_steps=None, val_steps=None):
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
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        tf_data_exp = tf.data.experimental
        
        if train_steps is None or val_steps is None:
            train_card = tf_data_exp.cardinality(train_dataset).numpy()
            val_card = tf_data_exp.cardinality(val_dataset).numpy()

            if train_card == tf_data_exp.INFINITE_CARDINALITY or train_card == tf_data_exp.UNKNOWN_CARDINALITY:
                train_steps = 100
            else:
                train_steps = train_card

            if val_card == tf_data_exp.INFINITE_CARDINALITY or val_card == tf_data_exp.UNKNOWN_CARDINALITY:
                val_steps = 20
            else:
                val_steps = val_card
        
        print(f"Using train steps: {train_steps}, val steps: {val_steps}")

        history = self.model.fit(
            train_dataset,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model.save(f"{save_path}/final_model.h5")
        return history


if __name__ == "__main__":
    print("Testing models.py import")
    p = PPEDataPreprocessor("dummy_path")
    print("PPEDataPreprocessor loaded:", isinstance(p, PPEDataPreprocessor))
    m = PPEDetectionModel()
    print("PPEDetectionModel loaded:", isinstance(m, PPEDetectionModel))
=======
import os
import tensorflow as tf

class PPEDataPreprocessor:
    def __init__(self, dataset_path, image_size=(224, 224)):
        self.dataset_path = dataset_path
        self.image_size = image_size
    
    def create_tensorflow_dataset(self):
        def load_and_preprocess_image(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.image_size)
            image = tf.cast(image, tf.float32) / 255.0
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
            
            path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
            label_ds = tf.data.Dataset.from_tensor_slices(labels)
            dataset = tf.data.Dataset.zip((path_ds, label_ds))
            dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            
            if split == 'train':
                dataset = dataset.shuffle(80)
                dataset = dataset.repeat()
            
            dataset = dataset.batch(32)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            datasets[split] = dataset
        
        return datasets


class PPEDetectionModel:
    def __init__(self, num_classes=9, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
    
    def create_mobilenet_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def train(self, train_dataset, val_dataset, epochs=50, save_path="ppe_model", train_steps=None, val_steps=None):
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
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        tf_data_exp = tf.data.experimental
        
        if train_steps is None or val_steps is None:
            train_card = tf_data_exp.cardinality(train_dataset).numpy()
            val_card = tf_data_exp.cardinality(val_dataset).numpy()

            if train_card == tf_data_exp.INFINITE_CARDINALITY or train_card == tf_data_exp.UNKNOWN_CARDINALITY:
                train_steps = 100
            else:
                train_steps = train_card

            if val_card == tf_data_exp.INFINITE_CARDINALITY or val_card == tf_data_exp.UNKNOWN_CARDINALITY:
                val_steps = 20
            else:
                val_steps = val_card
        
        print(f"Using train steps: {train_steps}, val steps: {val_steps}")

        history = self.model.fit(
            train_dataset,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model.save(f"{save_path}/final_model.h5")
        return history


if __name__ == "__main__":
    print("Testing models.py import")
    p = PPEDataPreprocessor("dummy_path")
    print("PPEDataPreprocessor loaded:", isinstance(p, PPEDataPreprocessor))
    m = PPEDetectionModel()
    print("PPEDetectionModel loaded:", isinstance(m, PPEDetectionModel))
>>>>>>> 63bdbc57ff2dd37cea86ebefae4c1a32377e710c
