# Raw vs Cooked Food Recognition 🍽️

A deep learning project that uses Convolutional Neural Networks (CNN) to classify food images as either **Raw** or **Cooked**.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Web Application](#web-application)

## ✨ Features

- **Deep Learning Model**: Custom CNN architecture for binary classification
- **Data Augmentation**: Robust training with image augmentation techniques
- **Easy Prediction**: Simple scripts for single and batch predictions
- **Web Interface**: Interactive Streamlit web application
- **Visualization**: Training metrics and prediction confidence visualization
- **Dataset Tools**: Helper scripts for dataset organization

## 📁 Project Structure

```
raw-vs-cooked-food-recognition/
│
├── train_model.py          # Main training script
├── predict.py              # Prediction script for testing
├── app.py                  # Streamlit web application
├── organize_data.py        # Dataset organization helper
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── data/                  # Dataset directory (you create this)
│   ├── train/
│   │   ├── raw/          # Raw food images for training
│   │   └── cooked/       # Cooked food images for training
│   ├── validation/
│   │   ├── raw/          # Raw food images for validation
│   │   └── cooked/       # Cooked food images for validation
│   └── test/
│       ├── raw/          # Raw food images for testing
│       └── cooked/       # Cooked food images for testing
│
├── test_images/           # Sample images for quick testing
│
└── outputs/               # Generated files
    ├── best_model.h5
    ├── food_classifier_model.h5
    └── training_history.png
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this project**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Option 1: Manual Organization

1. Create the directory structure:
```bash
python organize_data.py
# Choose option 1
```

2. Add your images:
   - Place raw food images in `data/train/raw/`, `data/validation/raw/`, `data/test/raw/`
   - Place cooked food images in `data/train/cooked/`, `data/validation/cooked/`, `data/test/cooked/`

### Option 2: Automatic Splitting

If you have all images in two folders, use the automatic splitting:

```bash
python organize_data.py
# Choose option 2
# Provide paths to your raw and cooked image folders
```

### Dataset Recommendations

- **Minimum**: 500-1000 images per category
- **Recommended**: 2000+ images per category
- **Image quality**: Clear, well-lit, focused on the food
- **Variety**: Different types of foods, angles, lighting conditions

### Where to Get Images

1. **Kaggle Datasets**:
   - Food-101: https://www.kaggle.com/datasets/dansbecker/food-101
   - Various fruit and vegetable datasets

2. **Free Image Sites**:
   - Unsplash: https://unsplash.com
   - Pexels: https://www.pexels.com
   - Pixabay: https://pixabay.com

3. **Create Your Own**:
   - Photograph raw ingredients
   - Cook them and photograph again
   - Ensure consistent lighting and quality

## 🎯 Usage

### 1. Verify Your Dataset

```bash
python organize_data.py
# Choose option 3 to verify dataset
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess your images
- Train the CNN model with data augmentation
- Save the best model as `best_model.h5`
- Generate training history plots
- Save the final model as `food_classifier_model.h5`

**Training Parameters**:
- Image size: 224x224
- Batch size: 32
- Epochs: 50 (with early stopping)
- Optimizer: Adam (learning rate: 0.0001)

### 3. Make Predictions

#### Single Image Prediction:
```python
python predict.py
```

Or in Python:
```python
from predict import FoodPredictor

predictor = FoodPredictor('food_classifier_model.h5')
result = predictor.predict('path/to/your/image.jpg')
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}%")
```

#### Batch Prediction:
```python
from predict import FoodPredictor

predictor = FoodPredictor('food_classifier_model.h5')
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = predictor.predict_batch(images)

for result in results:
    print(f"{result['path']}: {result['label']} ({result['confidence']:.2f}%)")
```

#### Visualize Predictions:
```python
predictor = FoodPredictor('food_classifier_model.h5')
predictor.visualize_prediction('path/to/image.jpg', save_path='prediction_result.png')
```

### 4. Launch Web Application

```bash
streamlit run app.py
```

This will open a browser window with an interactive interface where you can:
- Upload and classify individual images
- Process multiple images in batch mode
- View confidence scores and detailed results
- See real-time predictions

## 🏗️ Model Architecture

The model uses a custom CNN architecture:

```
Layer (type)                Output Shape              Params
================================================================
Conv2D (32 filters)         (None, 222, 222, 32)      896
BatchNormalization          (None, 222, 222, 32)      128
MaxPooling2D                (None, 111, 111, 32)      0
Dropout (0.25)              (None, 111, 111, 32)      0

Conv2D (64 filters)         (None, 109, 109, 64)      18,496
BatchNormalization          (None, 109, 109, 64)      256
MaxPooling2D                (None, 54, 54, 64)        0
Dropout (0.25)              (None, 54, 54, 64)        0

Conv2D (128 filters)        (None, 52, 52, 128)       73,856
BatchNormalization          (None, 52, 52, 128)       512
MaxPooling2D                (None, 26, 26, 128)       0
Dropout (0.25)              (None, 26, 26, 128)       0

Conv2D (256 filters)        (None, 24, 24, 256)       295,168
BatchNormalization          (None, 24, 24, 256)       1,024
MaxPooling2D                (None, 12, 12, 256)       0
Dropout (0.25)              (None, 12, 12, 256)       0

Flatten                     (None, 36864)             0
Dense (512)                 (None, 512)               18,874,880
BatchNormalization          (None, 512)               2,048
Dropout (0.5)               (None, 512)               0
Dense (256)                 (None, 256)               131,328
BatchNormalization          (None, 256)               1,024
Dropout (0.5)               (None, 256)               0
Dense (1, sigmoid)          (None, 1)                 257
================================================================
Total params: 19,399,873
Trainable params: 19,397,377
Non-trainable params: 2,496
```

**Key Features**:
- Multiple convolutional blocks for feature extraction
- Batch normalization for stable training
- Dropout layers to prevent overfitting
- Binary classification output with sigmoid activation

## 📈 Results

After training, you'll get:

1. **Training History Plot** (`training_history.png`):
   - Accuracy curves
   - Loss curves
   - Precision and Recall metrics

2. **Model Files**:
   - `best_model.h5`: Best model based on validation accuracy
   - `food_classifier_model.h5`: Final trained model

3. **Performance Metrics**:
   - Training and validation accuracy
   - Precision and recall scores
   - Confusion matrix (optional)

### Expected Performance

With a good dataset:
- Training Accuracy: 90-95%
- Validation Accuracy: 85-92%
- Prediction Confidence: 70-99% depending on image quality

## 🌐 Web Application

The Streamlit web app provides:

- **Single Image Classification**: Upload and classify one image at a time
- **Batch Processing**: Upload multiple images for bulk classification
- **Confidence Visualization**: See detailed prediction scores
- **User-Friendly Interface**: No coding required
- **Real-Time Results**: Instant predictions

### Running the Web App

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found error**:
   - Make sure you've trained the model first with `python train_model.py`
   - Check that `food_classifier_model.h5` exists in the project directory

2. **Out of memory error**:
   - Reduce batch size in `train_model.py` (line with `batch_size=32`)
   - Use a smaller image size (change `img_height` and `img_width`)

3. **Low accuracy**:
   - Add more training images
   - Ensure images are properly labeled
   - Increase training epochs
   - Check for data quality issues

4. **Import errors**:
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Use Python 3.8 or higher

## 📝 Tips for Better Results

1. **Dataset Quality**:
   - Use clear, high-resolution images
   - Ensure good lighting
   - Avoid cluttered backgrounds
   - Balance the number of images per class

2. **Data Augmentation**:
   - The training script includes augmentation (rotation, zoom, flip)
   - This helps the model generalize better

3. **Training**:
   - Let the model train until early stopping triggers
   - Monitor validation metrics to prevent overfitting
   - Save multiple checkpoints

4. **Testing**:
   - Test with images the model hasn't seen
   - Try different food types
   - Check edge cases (partially cooked, grilled, etc.)

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- Streamlit for the web application framework
- The open-source community for inspiration and resources

## 📧 Contact

For questions or suggestions, please open an issue in the project repository.

---

**Happy Classifying! 🍽️🤖**