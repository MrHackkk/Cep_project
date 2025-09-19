import tensorflow as tf
from models import PPEDataPreprocessor, PPEDetectionModel

def main():
    dataset_path = "construction_ppe_dataset"
    
    print("🔄 Loading dataset")
    preprocessor = PPEDataPreprocessor(dataset_path)
    datasets = preprocessor.create_tensorflow_dataset()
    
    print("⚙️ Creating and compiling model")
    ppe_model = PPEDetectionModel(num_classes=9)
    model = ppe_model.create_mobilenet_model()
    ppe_model.compile_model()
    
    if 'train' in datasets and 'validation' in datasets:
        tf_data_exp = tf.data.experimental
        
        train_card = tf_data_exp.cardinality(datasets['train']).numpy()
        val_card = tf_data_exp.cardinality(datasets['validation']).numpy()
        
        train_steps = 100 if train_card in [tf_data_exp.INFINITE_CARDINALITY, tf_data_exp.UNKNOWN_CARDINALITY] else train_card
        val_steps = 20 if val_card in [tf_data_exp.INFINITE_CARDINALITY, tf_data_exp.UNKNOWN_CARDINALITY] else val_card
        
        print(f"Training steps: {train_steps}, Validation steps: {val_steps}")
        
        print("🏋️ Starting training")
        history = ppe_model.train(
            train_dataset=datasets['train'],
            val_dataset=datasets['validation'],
            epochs=10,
            train_steps=train_steps,
            val_steps=val_steps
        )
        print("✅ Training complete. Model saved.")
    else:
        print("Training or validation dataset not found")

if __name__ == "__main__":
    main()
