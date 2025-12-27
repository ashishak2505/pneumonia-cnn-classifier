🫁 Pneumonia Detection from Chest X-Ray Images using Deep Learning
📌 Project Overview

This project implements an end-to-end Deep Learning solution to detect Pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs) and Transfer Learning (MobileNetV2).
The model is trained on a real-world medical dataset and deployed as an interactive Streamlit web application.

The focus of this project is not only achieving good accuracy, but also handling medical-domain challenges such as class imbalance, noisy images, and evaluation using clinically relevant metrics.

🎯 Problem Statement

Early detection of pneumonia is critical in medical diagnosis. Manual examination of chest X-rays is time-consuming and subject to human error.
This project aims to build an AI-based screening system that can assist in identifying pneumonia cases with high recall, reducing the chances of missing infected patients.

🧠 Approach & Methodology
1️⃣ Data Exploration

Dataset contains Chest X-ray images labeled as:

NORMAL

PNEUMONIA

Observed:

Class imbalance (more pneumonia cases)

Variation in image resolution

Blurry and noisy medical images

2️⃣ Baseline CNN (From Scratch)

Built a simple CNN to understand dataset behavior

Observed overfitting and poor generalization

Used as a baseline for comparison

3️⃣ Regularization Techniques

Data Augmentation

Dropout layers

Reduced model complexity
→ Still limited improvement due to dataset size

4️⃣ Transfer Learning (Final Model)

Used MobileNetV2 pretrained on ImageNet

Frozen base layers for feature extraction

Added custom classification head

Fine-tuned top layers with low learning rate

🏗️ Model Architecture (Final)

Base Model: MobileNetV2 (pretrained)

Custom Head:

Global Average Pooling

Dense Layer (ReLU)

Dropout (0.5)

Output Layer (Sigmoid)

📊 Model Performance
✅ Key Results

Validation Accuracy: ~87%

Test Accuracy: ~81%

Pneumonia Recall: 98%

🧪 Confusion Matrix Insight

Very low false negatives (missed pneumonia cases)

Higher false positives (acceptable in medical screening)

📌 Why recall matters more than accuracy here:
In healthcare, missing a disease is far riskier than additional screening of healthy patients.

📈 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Special emphasis was placed on recall for pneumonia cases.

🖥️ Web Application (Streamlit)

The trained model is deployed using Streamlit, allowing users to:

Upload a chest X-ray image

View probability scores for both classes

Get a clear prediction with confidence

Visualize confidence using progress bars

⚠️ This application is for educational purposes only and not a medical diagnosis tool
🛠️ Tech Stack

Python

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn

Streamlit

Google Colab

Git & GitHub