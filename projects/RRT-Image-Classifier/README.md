# RRT Image Classifier

A deep-learning based binary image classifier for distinguishing **RRT Normal** and **RRT Abnormal** images using **MobileNetV3Small** with transfer learning.

## Overview

This project uses a pretrained MobileNetV3Small convolutional neural network as the feature extractor and adds a lightweight binary classification head for RRT image classification.

### Pipeline

```text
RRT Dataset
    ↓
Train / Validation Split (80 / 20)
    ↓
Resize to 224 × 224
    ↓
MobileNetV3Small (ImageNet pretrained)
    ↓
Global Average Pooling
    ↓
Dense(128, ReLU)
    ↓
Dense(1, Sigmoid)
    ↓
RRT Normal / RRT Abnormal
```

## Technologies

- Python
- TensorFlow / Keras
- MobileNetV3Small
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Model Architecture

- **Backbone:** MobileNetV3Small
- **Pretrained weights:** ImageNet
- **Input:** 224 × 224 × 3
- **Backbone:** frozen during training
- **Pooling:** Global Average Pooling
- **Dense layer:** 128 neurons, ReLU
- **Output:** 1 neuron, Sigmoid
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Training:** 10 epochs

## Evaluation

The notebook evaluates the classifier using:

- Training and validation accuracy
- Training and validation loss
- Sensitivity / True Positive Rate
- ROC curve
- ROC-AUC
- Confusion matrix

## Inference

The trained model can classify an input image as:

- `RRT Normal`
- `RRT Abnormal`

Models are saved in both formats:

```text
RRT_mobilenet_model.h5
RRT_mobilenet_model.keras
```

## Google Colab

[Open the original notebook in Google Colab](https://colab.research.google.com/drive/1DTz1mJEPYk2sf-qSQzlUCEgrWo9r270F?usp=sharing)

## Project Structure

```text
RRT-Image-Classifier/
├── RRT_Image_Classifier.ipynb
├── README.md
└── requirements.txt
```

## Requirements

Install the main dependencies with:

```bash
pip install -r requirements.txt
```

> The dataset ZIP file is not included in this repository. The notebook expects `RRT_Dataset.zip` and extracts it into `RRT_Dataset/`.

## Note on the Current Notebook

The notebook defines a RandomFlip/RandomRotation augmentation pipeline and a Rescaling layer. However, those intermediate tensors are overwritten before the final Model is constructed, so the augmentation/rescaling operations are not currently connected to the model graph. The repository preserves the submitted experiment rather than silently changing its results.

## Author

**Disha Goel**  
B.Tech — Electrical & Electronics Engineering  
Interests: AI/ML, Computer Vision, Robotics & Automation
