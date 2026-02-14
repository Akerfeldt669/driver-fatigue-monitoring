"""
Model Utilities
Helper functions for model creation and management
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Dict, Tuple
import os
from pathlib import Path


def build_model_from_config(config: dict) -> keras.Model:
    """
    Build Keras model from configuration dictionary
    
    Args:
        config: Model configuration (from model_config.yaml)
    
    Returns:
        Compiled Keras model
    
    Example:
        >>> config = {
        ...     'input_shape': [32, 32, 1],
        ...     'num_classes': 2,
        ...     'layers': [
        ...         {'type': 'Conv2D', 'filters': 32, 'kernel_size': [3, 3]},
        ...         {'type': 'Dense', 'units': 64}
        ...     ]
        ... }
        >>> model = build_model_from_config(config)
    """
    input_shape = config['input_shape']
    layer_configs = config['layers']
    
    # Create sequential model
    model = keras.Sequential()
    
    # Add input layer
    model.add(layers.Input(shape=input_shape))
    
    # Add layers from config
    for layer_config in layer_configs:
        layer = create_layer_from_config(layer_config)
        model.add(layer)
    
    return model


def create_layer_from_config(layer_config: dict) -> layers.Layer:
    """
    Create a Keras layer from configuration
    
    Args:
        layer_config: Layer configuration dictionary
    
    Returns:
        Keras layer
    """
    layer_type = layer_config['type']
    params = {k: v for k, v in layer_config.items() if k != 'type'}
    
    # Map layer types
    layer_map = {
        'Conv2D': layers.Conv2D,
        'MaxPooling2D': layers.MaxPooling2D,
        'AveragePooling2D': layers.AveragePooling2D,
        'Flatten': layers.Flatten,
        'Dense': layers.Dense,
        'Dropout': layers.Dropout,
        'BatchNormalization': layers.BatchNormalization,
        'GlobalAveragePooling2D': layers.GlobalAveragePooling2D,
        'GlobalMaxPooling2D': layers.GlobalMaxPooling2D,
    }
    
    if layer_type not in layer_map:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    return layer_map[layer_type](**params)


def convert_to_tflite(
    model_path: str,
    output_path: str,
    quantize: bool = True,
    quantization_type: str = 'float16'
) -> Tuple[int, int]:
    """
    Convert Keras model to TensorFlow Lite
    
    Args:
        model_path: Path to .h5 model file
        output_path: Path to save .tflite file
        quantize: Enable quantization
        quantization_type: 'float16', 'int8', or 'dynamic'
    
    Returns:
        (original_size_kb, tflite_size_kb)
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quantization_type == 'float16':
            converter.target_spec.supported_types = [tf.float16]
        elif quantization_type == 'int8':
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Calculate sizes
    original_size = os.path.getsize(model_path) / 1024  # KB
    tflite_size = os.path.getsize(output_path) / 1024   # KB
    
    return int(original_size), int(tflite_size)


def get_model_summary(model: keras.Model) -> dict:
    """
    Get model summary as dictionary
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with model info
    """
    total_params = model.count_params()
    
    trainable_params = sum([
        tf.keras.backend.count_params(w) 
        for w in model.trainable_weights
    ])
    
    non_trainable_params = total_params - trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }


def save_model_info(model: keras.Model, save_path: str):
    """
    Save model architecture and summary to text file
    
    Args:
        model: Keras model
        save_path: Path to save info
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        # Model summary
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        f.write("\n\n")
        f.write("="*70 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("="*70 + "\n\n")
        
        # Additional info
        info = get_model_summary(model)
        for key, value in info.items():
            f.write(f"{key}: {value}\n")


def load_model_safe(model_path: str) -> keras.Model:
    """
    Safely load Keras model with error handling
    
    Args:
        model_path: Path to model file
    
    Returns:
        Loaded model
    
    Raises:
        FileNotFoundError: If model doesn't exist
        ValueError: If model is corrupted
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")


if __name__ == "__main__":
    # Test model utilities
    print("="*70)
    print("MODEL UTILITIES TEST")
    print("="*70)
    print()
    
    # Test model building from config
    print("Testing build_model_from_config...")
    
    test_config = {
        'input_shape': [32, 32, 1],
        'num_classes': 2,
        'layers': [
            {'type': 'Conv2D', 'filters': 32, 'kernel_size': [3, 3], 'activation': 'relu', 'padding': 'same'},
            {'type': 'MaxPooling2D', 'pool_size': [2, 2]},
            {'type': 'Flatten'},
            {'type': 'Dense', 'units': 64, 'activation': 'relu'},
            {'type': 'Dense', 'units': 2, 'activation': 'softmax'}
        ]
    }
    
    model = build_model_from_config(test_config)
    print("✓ Model built successfully")
    print()
    
    # Test model summary
    print("Testing get_model_summary...")
    summary = get_model_summary(model)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print()
    print("="*70)