# ASL Sign Language Translation System

A real-time American Sign Language (ASL) translation system using deep learning techniques to classify sign language gestures with high accuracy.

## 🎯 Project Overview

This project implements a machine learning-based system for recognizing American Sign Language gestures using computer vision and recurrent neural networks. The system achieves 98% testing accuracy and supports real-time prediction through webcam integration.

## 📊 Dataset

We utilized the **ASL Citizen dataset** from Kaggle, which contains video samples of ASL glosses. For this project:

- **Selected Glosses**: 12 carefully chosen glosses with balanced representation
- **Original Samples**: ~30 video samples per gloss
- **Final Dataset**: ~3,240 video samples after augmentation (~270 samples per gloss)

### Data Augmentation Pipeline

To enhance model generalization and expand the limited dataset, we implemented comprehensive augmentation techniques:

**Geometric Transformations**
- Random rotation
- Horizontal flipping  
- Random cropping

**Temporal Variations**
- Speed adjustments (faster/slower versions)

**Photometric Augmentations**
- Brightness adjustments
- High/low contrast modifications
- Blur effects

## 🔧 Feature Extraction

### MediaPipe Holistic Integration
- **Tool**: MediaPipe Holistic for landmark detection
- **Focus**: Hand keypoints (most relevant for ASL recognition)
- **Output**: 63 keypoints per hand × 2 hands = 126 keypoints per frame
- **Coordinates**: x, y, z coordinates for each landmark

### Sequence Normalization
Videos varied in length, requiring normalization for LSTM input:

**Motion Scoring Technique**
- Calculated frame-by-frame motion using Euclidean distance
- Identified most dynamic/informative portions
- Trimmed longer sequences to preserve semantic content
- Padded shorter sequences with zeros for uniform length

## 🧠 Model Architecture Comparison

We evaluated three different recurrent neural network architectures:

### 1. LSTM (Selected Model)
- Two LSTM layers (64 units each)
- ReLU activation function
- Batch normalization and dropout (0.2)
- Dense layer (32 units) + dropout
- Output layer with SoftMax activation

### 2. Bidirectional LSTM (BiLSTM)
- Two Bidirectional LSTM layers (64 units each)
- ReLU activation function
- Batch normalization and dropout (0.2)
- Dense layer (32 units) + dropout
- Output layer with SoftMax activation

### 3. GRU
- Two GRU layers (64 units each) with L2 regularization
- LeakyReLU activation
- Batch normalization and dropout (0.3)
- Dense layer (64 units) + dropout
- Output layer with SoftMax activation

### Architecture Selection Rationale

**LSTM was chosen for deployment based on:**
- **Highest accuracy** on validation and test sets
- **Optimal complexity-performance balance** compared to BiLSTM
- **Better accuracy** than GRU while maintaining reasonable speed
- **Real-time suitability** with efficient inference

## 🚀 Training Configuration

- **Optimizer**: Adam (adaptive learning rates)
- **Loss Function**: Categorical Cross-Entropy
- **Evaluation Metric**: Categorical Accuracy
- **Epochs**: 120
- **Batch Size**: 96
- **Train/Test Split**: 95%/5% (stratified sampling)

## 📈 Performance Results

### Model Performance
- **Training Accuracy**: ~99%
- **Testing Accuracy**: 98%

### Evaluation Metrics
- **Confusion Matrix**: Minimal misclassifications across all glosses
- **Classification Report**: High precision, recall, and F1-scores for all classes
- **Real-time Performance**: Near-perfect prediction with minimal latency

## 🎥 Real-Time Testing

The system includes live webcam integration for real-time sign language recognition:

- **Pipeline**: MediaPipe Holistic → Feature Extraction → LSTM Prediction
- **Performance**: Accurate predictions with minimal latency
- **Validation**: Successful deployment confirms practical viability

## 💻 Getting Started

### Prerequisites
```bash
pip install tensorflow
pip install mediapipe
pip install opencv-python
pip install numpy
pip install pandas
pip install scikit-learn
```

### Installation
```bash
git clone [your-repository-url]
cd asl-sign-language-recognition
pip install -r requirements.txt
```

## 📁 Project Structure
```
asl-sign-language-recognition/
├── data/
│   ├── raw/              # Original ASL Citizen dataset
│   ├── processed/        # Augmented and preprocessed data
│   └── features/         # Extracted keypoints
├── models/
│   ├── lstm_model.py     # LSTM architecture
│   ├── bilstm_model.py   # BiLSTM architecture
│   └── gru_model.py      # GRU architecture
├── preprocessing/
│   ├── augmentation.py   # Data augmentation pipeline
│   ├── feature_extraction.py  # MediaPipe integration
│   └── sequence_processing.py # Motion scoring & normalization
├── evaluation/
│   ├── metrics.py        # Performance evaluation
│   └── visualization.py  # Results visualization
├── live_prediction.py    # Real-time webcam integration
├── train_model.py        # Training script
└── README.md
```

## 🔬 Technical Highlights

- **Advanced Data Augmentation**: Comprehensive pipeline addressing geometric, temporal, and photometric variations
- **Intelligent Sequence Processing**: Motion scoring technique for optimal frame selection
- **Architecture Optimization**: Systematic comparison of RNN variants
- **Real-time Deployment**: Successful integration with live video stream

## 🎯 Future Enhancements

- Expand vocabulary to include more ASL glosses
- Implement sentence-level translation
- Add support for continuous sign language recognition
- Optimize model for mobile deployment
- Integrate with speech synthesis for complete translation


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



**Note**: This project demonstrates the practical application of deep learning in accessibility technology, achieving high accuracy in real-time ASL recognition.
