"""
Train Yawn Detector
Binary classification: No Yawn vs Yawn
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
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.config import config
from scripts.utils.logger import logger
from scripts.utils.visualization import plot_training_history, plot_confusion_matrix
from scripts.utils.model_utils import convert_to_tflite, save_model_info

# ============================================
# CONFIGURATION - UPDATE THIS PATH!
# ============================================

# Your dataset path
# Expected structure:
# C:\Users\ADMIN\Desktop\data1\
# ├── no_yawn\   ← Images of normal mouth/no yawning
# └── yawn\      ← Images of yawning

SOURCE_DATA_DIR = r"C:\Users\ADMIN\Desktop\data1"
PREPARED_DATA_DIR = r"C:\Users\ADMIN\Desktop\data1_prepared"

# Hyperparameters
IMG_SIZE = config.get('data.mouth_image_size', 48)
BATCH_SIZE = config.get('data.batch_size', 32)
EPOCHS = config.get('training.yawn_detector.epochs', 40)
LEARNING_RATE = config.get('training.yawn_detector.learning_rate', 0.0005)

# Data split ratios
TRAIN_SPLIT = 0.8  # 80% training
VAL_SPLIT = 0.1    # 10% validation
TEST_SPLIT = 0.1   # 10% test

# Output paths
MODELS_DIR = "models/saved"
CHECKPOINTS_DIR = "models/checkpoints"
TFLITE_DIR = "models/tflite"
PLOTS_DIR = "outputs/plots/training"

# Create directories
for directory in [MODELS_DIR, CHECKPOINTS_DIR, TFLITE_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================
# DATA PREPARATION
# ============================================

def prepare_data_splits():
    """
    Split the data into train/val/test if not already done
    """
    logger.info("="*70)
    logger.info("PREPARING DATA SPLITS")
    logger.info("="*70)
    logger.info("")
    
    # Check if already prepared
    train_dir = os.path.join(PREPARED_DATA_DIR, "train")
    if os.path.exists(train_dir):
        logger.info("✓ Data already prepared")
        logger.info(f"Using: {PREPARED_DATA_DIR}")
        logger.info("")
        return
    
    # Verify source data exists
    if not os.path.exists(SOURCE_DATA_DIR):
        raise FileNotFoundError(
            f"Source data directory not found: {SOURCE_DATA_DIR}\n"
            f"Please make sure your data is at this location with folders:\n"
            f"  - no_yawn/\n"
            f"  - yawn/"
        )
    
    logger.info(f"Source: {SOURCE_DATA_DIR}")
    logger.info(f"Target: {PREPARED_DATA_DIR}")
    logger.info("")
    
    # Get all image files for each class
    classes = ['no_yawn', 'yawn']
    
    for class_name in classes:
        class_dir = os.path.join(SOURCE_DATA_DIR, class_name)
        
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        
        # Get all image files
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {class_dir}")
        
        logger.info(f"{class_name}: {len(image_files)} images")
        
        # Split into train/val/test
        train_files, temp_files = train_test_split(
            image_files, 
            test_size=(VAL_SPLIT + TEST_SPLIT),
            random_state=42
        )
        
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)),
            random_state=42
        )
        
        logger.info(f"  Train: {len(train_files)}")
        logger.info(f"  Val:   {len(val_files)}")
        logger.info(f"  Test:  {len(test_files)}")
        logger.info("")
        
        # Copy files to prepared directory
        for split_name, split_files in [
            ('train', train_files),
            ('val', val_files),
            ('test', test_files)
        ]:
            target_dir = os.path.join(PREPARED_DATA_DIR, split_name, class_name)
            os.makedirs(target_dir, exist_ok=True)
            
            for filename in split_files:
                src = os.path.join(class_dir, filename)
                dst = os.path.join(target_dir, filename)
                shutil.copy2(src, dst)
    
    logger.info("✓ Data preparation complete!")
    logger.info("")


# ============================================
# DATA LOADING
# ============================================

def load_data():
    """Load and prepare training data"""
    logger.info("="*70)
    logger.info("LOADING DATASET")
    logger.info("="*70)
    logger.info("")
    
    # First, prepare the data splits
    prepare_data_splits()
    
    # Now load from prepared directory
    train_dir = os.path.join(PREPARED_DATA_DIR, "train")
    val_dir = os.path.join(PREPARED_DATA_DIR, "val")
    test_dir = os.path.join(PREPARED_DATA_DIR, "test")
    
    logger.info(f"Train directory: {train_dir}")
    logger.info(f"Val directory: {val_dir}")
    logger.info(f"Test directory: {test_dir}")
    logger.info("")
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation/Test data (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data (RGB for mouth)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',  # RGB for mouth images
        shuffle=True,
        seed=42
    )
    
    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
    
    # Load test data
    test_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
    
    # Print dataset info
    logger.info(f"✓ Training samples: {train_generator.samples}")
    logger.info(f"✓ Validation samples: {val_generator.samples}")
    logger.info(f"✓ Test samples: {test_generator.samples}")
    
    # Print class mapping
    class_indices = train_generator.class_indices
    logger.info(f"✓ Classes detected: {class_indices}")
    logger.info(f"  - {list(class_indices.keys())[0]}: {list(class_indices.values())[0]}")
    logger.info(f"  - {list(class_indices.keys())[1]}: {list(class_indices.values())[1]}")
    logger.info(f"✓ Batch size: {BATCH_SIZE}")
    logger.info(f"✓ Image size: {IMG_SIZE}x{IMG_SIZE}")
    logger.info("")
    
    return train_generator, val_generator, test_generator


# ============================================
# MODEL ARCHITECTURE
# ============================================

def build_model():
    """Build Yawn Detector CNN model"""
    logger.info("="*70)
    logger.info("BUILDING MODEL")
    logger.info("="*70)
    logger.info("")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),  # RGB images
        
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
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Output layer
        layers.Dense(2, activation='softmax')  # 2 classes: no_yawn, yawn
    ])
    
    # Compile
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
    
    # Save architecture
    save_model_info(model, os.path.join(MODELS_DIR, "yawn_detector_architecture.txt"))
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
        EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        ModelCheckpoint(
            filepath=os.path.join(CHECKPOINTS_DIR, 'yawn_detector_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        TensorBoard(
            log_dir=os.path.join('outputs/logs', f'yawn_detector_{timestamp}'),
            histogram_freq=1
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

def evaluate_model(model, val_gen, test_gen):
    """Evaluate model"""
    logger.info("="*70)
    logger.info("EVALUATING MODEL")
    logger.info("="*70)
    logger.info("")
    
    # Validation
    logger.info("Validation Set:")
    val_loss, val_acc, val_precision, val_recall = model.evaluate(val_gen, verbose=0)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)
    
    logger.info(f"  Loss:      {val_loss:.4f}")
    logger.info(f"  Accuracy:  {val_acc:.4f} ({val_acc*100:.2f}%)")
    logger.info(f"  Precision: {val_precision:.4f}")
    logger.info(f"  Recall:    {val_recall:.4f}")
    logger.info(f"  F1-Score:  {val_f1:.4f}")
    logger.info("")
    
    # Test
    logger.info("Test Set:")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen, verbose=0)
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
    
    logger.info(f"  Loss:      {test_loss:.4f}")
    logger.info(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    logger.info(f"  Precision: {test_precision:.4f}")
    logger.info(f"  Recall:    {test_recall:.4f}")
    logger.info(f"  F1-Score:  {test_f1:.4f}")
    logger.info("")
    
    # Confusion matrix
    logger.info("Generating confusion matrix...")
    predictions = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    class_names = list(test_gen.class_indices.keys())
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=os.path.join(PLOTS_DIR, "yawn_detector_confusion_matrix.png"),
        show=False
    )
    logger.info(f"✓ Confusion matrix saved")
    logger.info("")


# ============================================
# SAVE MODEL
# ============================================

def save_final_model(model, history):
    """Save trained model"""
    logger.info("="*70)
    logger.info("SAVING MODEL")
    logger.info("="*70)
    logger.info("")
    
    # Save Keras model
    model_path = os.path.join(MODELS_DIR, 'yawn_detector.h5')
    model.save(model_path)
    logger.info(f"✓ Keras model saved: {model_path}")
    
    # Convert to TFLite
    logger.info("Converting to TensorFlow Lite...")
    tflite_path = os.path.join(TFLITE_DIR, 'yawn_detector.tflite')
    
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
    
    # Plot history
    plot_training_history(
        history.history,
        save_path=os.path.join(PLOTS_DIR, "yawn_detector_training_history.png"),
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
    logger.info("║" + " "*23 + "YAWN DETECTOR TRAINING" + " "*23 + "║")
    logger.info("║" + " "*23 + "(No Yawn vs Yawn)" + " "*27 + "║")
    logger.info("╚" + "="*68 + "╝")
    logger.info("\n")
    
    try:
        # Load data (will auto-split if needed)
        train_gen, val_gen, test_gen = load_data()
        
        # Build model
        model = build_model()
        
        # Train
        history = train_model(model, train_gen, val_gen)
        
        # Evaluate
        evaluate_model(model, val_gen, test_gen)
        
        # Save
        save_final_model(model, history)
        
        logger.info("="*70)
        logger.info("🎉 ALL DONE! YAWN DETECTOR TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info("")
        logger.info("Saved files:")
        logger.info(f"  - Keras model: models/saved/yawn_detector.h5")
        logger.info(f"  - TFLite model: models/tflite/yawn_detector.tflite")
        logger.info(f"  - Best checkpoint: models/checkpoints/yawn_detector_best.h5")
        logger.info(f"  - Training plots: outputs/plots/training/")
        logger.info("="*70)
        logger.info("")
        
        return model, history
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run training
    model, history = main()