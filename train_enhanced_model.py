#!/usr/bin/env python3
"""
Enhanced PPE Detection Model Training Script
Trains a high-accuracy model for construction site safety equipment detection
"""

import os
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from enhanced_models import EnhancedPPEDetector, PPEDataPreprocessor, create_callbacks

def setup_gpu():
    """Configure GPU settings for optimal training"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU available: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
    else:
        print("⚠️ No GPU found, using CPU")

def create_model_summary(model, save_path):
    """Create and save model summary"""
    with open(f"{save_path}/model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    
    # Plot model architecture
    tf.keras.utils.plot_model(
        model, 
        to_file=f"{save_path}/model_architecture.png",
        show_shapes=True,
        show_layer_names=True
    )

def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Top-3 Accuracy
    if 'top_3_accuracy' in history.history:
        axes[1, 0].plot(history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
        axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_history.png", dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_dataset, class_names):
    """Evaluate model and create detailed report"""
    print("\n🔍 Evaluating model...")
    
    # Get predictions
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
        y_pred_proba.extend(predictions)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"enhanced_ppe_model/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

def main():
    """Main training function"""
    print("🏗️ Enhanced PPE Detection Model Training")
    print("=" * 50)
    
    # Setup
    setup_gpu()
    
    # Configuration
    dataset_path = "construction_ppe_dataset"
    model_save_path = "enhanced_ppe_model"
    epochs = 100
    batch_size = 32
    learning_rate = 0.001
    
    # Create save directory
    os.makedirs(model_save_path, exist_ok=True)
    
    print(f"📁 Dataset path: {dataset_path}")
    print(f"💾 Model save path: {model_save_path}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please ensure the dataset is properly organized with train/validation/test splits")
        return
    
    # Create data preprocessor
    print("\n📊 Preparing dataset...")
    preprocessor = PPEDataPreprocessor(dataset_path)
    datasets = preprocessor.create_tensorflow_dataset(batch_size=batch_size)
    
    if 'train' not in datasets or 'validation' not in datasets:
        print("❌ Required train/validation datasets not found")
        return
    
    train_dataset = datasets['train']
    val_dataset = datasets['validation']
    test_dataset = datasets.get('test', None)
    
    print(f"✅ Dataset loaded successfully")
    
    # Create model
    print("\n🤖 Creating enhanced model...")
    detector = EnhancedPPEDetector()
    model = detector.create_enhanced_model()
    model = detector.compile_model(learning_rate=learning_rate)
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Create model summary
    create_model_summary(model, model_save_path)
    
    # Create callbacks
    callbacks = create_callbacks(model_save_path)
    
    # Train model
    print(f"\n🚀 Starting training for {epochs} epochs...")
    start_time = datetime.now()
    
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = datetime.now() - start_time
        print(f"✅ Training completed in {training_time}")
        
        # Save final model
        model.save(f"{model_save_path}/final_model.h5")
        print(f"💾 Final model saved to {model_save_path}/final_model.h5")
        
        # Plot training history
        plot_training_history(history, model_save_path)
        print("📊 Training plots saved")
        
        # Evaluate model
        if test_dataset:
            evaluation_results = evaluate_model(model, test_dataset, detector.class_names)
            
            # Save evaluation results
            with open(f"{model_save_path}/evaluation_results.txt", "w") as f:
                f.write(f"Test Accuracy: {evaluation_results['accuracy']:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(str(evaluation_results['classification_report']))
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 All files saved to: {model_save_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        print("💾 Saving current model...")
        model.save(f"{model_save_path}/interrupted_model.h5")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
