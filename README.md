# Driver Fatigue Monitoring System

## Overview
Real-time detection of driver fatigue through eye closure and yawn detection using deep learning models.

## Features
- **Eye Detection**: Binary classifier (open/closed eyes)
- **Yawn Detection**: Binary classifier (yawning/normal)
- **Real-time Processing**: Uses OpenCV + MediaPipe for face detection
- **TFLite Models**: Optimized models for edge deployment

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

### Usage
```bash
python python app\main.py
```

## Models
- **Eye Classifier**: `models/saved/eye_classifier.h5` (32×32 grayscale)
- **Yawn Detector**: `models/saved/yawn_detector.h5` (48×48 RGB)
- **TFLite versions**: Available in `models/tflite/`

## Training Your Own Models
```bash
python scripts/training/train_eye_model.py
python scripts/training/train_yawn_model.py
```

## Datasets

### Eye Dataset
Download from: https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset

Extract to:
```
data/raw/eyes/
├── train/
│   ├── closed/
│   └── open/
├── val/
│   ├── closed/
│   └── open/
└── test/
    ├── closed/
    └── open/
```

### Yawn Dataset
Download from: https://www.kaggle.com/datasets/davidvazquezcic/yawn-dataset

Extract to:
```
data/raw/yawns/
├── train/
│   ├── yawn/
│   └── no_yawn/
└── val/
    ├── yawn/
    └── no_yawn/
```

## Model Performance
- Eye Model Accuracy: [INSERT %]
- Yawn Model Accuracy: [INSERT %]

## Project Structure
```
├── app/              # Main application
├── config/           # Configuration files
├── data/             # Dataset folder (organize as above)
├── models/           # Saved models
│   ├── saved/        # Full models (.h5)
│   ├── tflite/       # Optimized models
│   └── checkpoints/  # Training checkpoints
├── scripts/          # Training & utility scripts
├── notebooks/        # Jupyter notebooks
├── outputs/          # Training logs & plots
└── testcode.py       # Main testing script
```

## License
MIT License - feel free to use for research and development

## Contact
ayoubaknouche666@gmail.com

## Acknowledgments
- MRL Eye Dataset by Akashshingha
- Yawn Dataset by David Vazquez
