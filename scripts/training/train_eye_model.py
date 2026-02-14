"""
Train Eye State Classifier
Binary classification: Awake vs Sleepy
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint, 
    TensorBoard
)
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.config import config
from scripts.utils.logger import logger
from scripts.utils.visualization import plot_training_history, plot_confusion_matrix
from scripts.utils.model_utils import convert_to_tflite, save_model_info

# ============================================
# CONFIGURATION - UPDATE THESE PATHS! with urs
# ============================================

# Your dataset path
# Expected structure from the dataset in the readme //kaggle:
# C:\Users\ADMIN\Desktop\data\
# ├── train\
# │   ├── awake\    ← Images of awake/open eyes
# │   └── sleepy\   ← Images of sleepy/closed eyes
# ├── val\
# │   ├── awake\
# │   └── sleepy\
# └── test\
#     ├── awake\
#     └── sleepy\

DATA_DIR = r"C:\Users\ADMIN\Desktop\data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Hyperparameters from config
IMG_SIZE = config.get('data.eye_image_size', 32)
BATCH_SIZE = config.get('data.batch_size', 32)
EPOCHS = config.get('training.eye_classifier.epochs', 50)
LEARNING_RATE = config.get('training.eye_classifier.learning_rate', 0.001)

# Output paths
MODELS_DIR = "models/saved"
CHECKPOINTS_DIR = "models/checkpoints"
TFLITE_DIR = "models/tflite"
PLOTS_DIR = "outputs/plots/training"

# Create directories
for directory in [MODELS_DIR, CHECKPOINTS_DIR, TFLITE_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================
# DATA LOADING
# ============================================

def load_data():
    """Load and prepare training data"""
    logger.info("="*70)
    logger.info("LOADING DATASET")
    logger.info("="*70)
    
    # Check if directories exist
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}\n"
            f"Please make sure your data is organized as:\n"
            f"  {DATA_DIR}/train/awake/\n"
            f"  {DATA_DIR}/train/sleepy/"
        )
    if not os.path.exists(VAL_DIR):
        raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")
    
    logger.info(f"Train directory: {TRAIN_DIR}")
    logger.info(f"Val directory: {VAL_DIR}")
    logger.info(f"Test directory: {TEST_DIR}")
    logger.info("")
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True,
        seed=42
    )
    
    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    
    # Load test data if exists
    test_generator = None
    if os.path.exists(TEST_DIR):
        test_generator = val_datagen.flow_from_directory(
            TEST_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=False
        )
    
    # Print dataset info
    logger.info(f"✓ Training samples: {train_generator.samples}")
    logger.info(f"✓ Validation samples: {val_generator.samples}")
    if test_generator:
        logger.info(f"✓ Test samples: {test_generator.samples}")
    
    # Print class mapping
    class_indices = train_generator.class_indices
    logger.info(f"✓ Classes detected: {class_indices}")
    logger.info(f"  - {list(class_indices.keys())[0]}: {list(class_indices.values())[0]}")
    logger.info(f"  - {list(class_indices.keys())[1]}: {list(class_indices.values())[1]}")
    logger.info(f"✓ Batch size: {BATCH_SIZE}")
    logger.info("")
    
    return train_generator, val_generator, test_generator


# ============================================
# MODEL ARCHITECTURE
# ============================================

def build_model():
    """Build Eye State CNN model"""
    logger.info("="*70)
    logger.info("BUILDING MODEL")
    logger.info("="*70)
    logger.info("")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(2, activation='softmax')  # 2 classes: awake, sleepy
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    # Print summary
    model.summary()
    
    # Save model architecture
    save_model_info(model, os.path.join(MODELS_DIR, "eye_classifier_architecture.txt"))
    logger.info(f"✓ Model architecture saved")
    logger.info("")
    
    return model


# ============================================
# CALLBACKS
# ============================================

def get_callbacks():
    """Setup training callbacks"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(CHECKPOINTS_DIR, 'eye_classifier_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=os.path.join('outputs/logs', f'eye_classifier_{timestamp}'),
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks


# ============================================
# TRAINING
# ============================================

def train_model(model, train_gen, val_gen):
    """Train the model"""
    logger.info("="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info("")
    
    callbacks = get_callbacks()
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("")
    logger.info("="*70)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info("")
    
    return history


# ============================================
# EVALUATION
# ============================================

def evaluate_model(model, val_gen, test_gen=None):
    """Evaluate model performance"""
    logger.info("="*70)
    logger.info("EVALUATING MODEL")
    logger.info("="*70)
    logger.info("")
    
    # Evaluate on validation set
    logger.info("Validation Set:")
    val_loss, val_acc, val_precision, val_recall = model.evaluate(val_gen, verbose=0)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
    
    logger.info(f"  Loss:      {val_loss:.4f}")
    logger.info(f"  Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
    logger.info(f"  Precision: {val_precision:.4f}")
    logger.info(f"  Recall:    {val_recall:.4f}")
    logger.info(f"  F1-Score:  {val_f1:.4f}")
    logger.info("")
    
    # Evaluate on test set if available
    if test_gen:
        logger.info("Test Set:")
        test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen, verbose=0)
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        logger.info(f"  Loss:      {test_loss:.4f}")
        logger.info(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        logger.info(f"  Precision: {test_precision:.4f}")
        logger.info(f"  Recall:    {test_recall:.4f}")
        logger.info(f"  F1-Score:  {test_f1:.4f}")
        logger.info("")
        
        # Generate confusion matrix
        logger.info("Generating confusion matrix...")
        predictions = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_gen.classes
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get actual class names (awake, sleepy)
        class_names = list(test_gen.class_indices.keys())
        plot_confusion_matrix(
            cm, 
            class_names, 
            save_path=os.path.join(PLOTS_DIR, "eye_classifier_confusion_matrix.png"),
            show=False
        )
        logger.info(f"✓ Confusion matrix saved")
        logger.info("")


# ============================================
# SAVE MODEL
# ============================================

def save_final_model(model, history):
    """Save trained model and artifacts"""
    logger.info("="*70)
    logger.info("SAVING MODEL")
    logger.info("="*70)
    logger.info("")
    
    # Save Keras model
    model_path = os.path.join(MODELS_DIR, 'eye_classifier.h5')
    model.save(model_path)
    logger.info(f"✓ Keras model saved: {model_path}")
    
    # Convert to TFLite
    logger.info("Converting to TensorFlow Lite...")
    tflite_path = os.path.join(TFLITE_DIR, 'eye_classifier.tflite')
    
    try:
        original_size, tflite_size = convert_to_tflite(
            model_path,
            tflite_path,
            quantize=True,
            quantization_type='float16'
        )
        
        compression = (1 - tflite_size/original_size) * 100
        logger.info(f"✓ TFLite model saved: {tflite_path}")
        logger.info(f"  Original size: {original_size} KB")
        logger.info(f"  TFLite size:   {tflite_size} KB")
        logger.info(f"  Compression:   {compression:.1f}%")
    except Exception as e:
        logger.error(f"Failed to convert to TFLite: {e}")
    
    logger.info("")
    
    # Plot training history
    logger.info("Generating training plots...")
    plot_training_history(
        history.history,
        save_path=os.path.join(PLOTS_DIR, "eye_classifier_training_history.png"),
        show=False
    )
    logger.info(f"✓ Training history plot saved")
    logger.info("")


# ============================================
# MAIN
# ============================================

def main():
    """Main training pipeline"""
    logger.info("\n")
    logger.info("╔" + "="*68 + "╗")
    logger.info("║" + " "*20 + "EYE STATE CLASSIFIER TRAINING" + " "*19 + "║")
    logger.info("║" + " "*25 + "(Awake vs Sleepy)" + " "*27 + "║")
    logger.info("╚" + "="*68 + "╝")
    logger.info("\n")
    
    try:
        # Load data
        train_gen, val_gen, test_gen = load_data()
        
        # Build model
        model = build_model()
        
        # Train model
        history = train_model(model, train_gen, val_gen)
        
        # Evaluate model
        evaluate_model(model, val_gen, test_gen)
        
        # Save model
        save_final_model(model, history)
        
        logger.info("="*70)
        logger.info("🎉 ALL DONE! MODEL TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info("")
        logger.info("Saved files:")
        logger.info(f"  - Keras model: models/saved/eye_classifier.h5")
        logger.info(f"  - TFLite model: models/tflite/eye_classifier.tflite")
        logger.info(f"  - Best checkpoint: models/checkpoints/eye_classifier_best.h5")
        logger.info(f"  - Training plots: outputs/plots/training/")
        logger.info("="*70)
        logger.info("")
        
        return model, history
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run training
    model, history = main()
