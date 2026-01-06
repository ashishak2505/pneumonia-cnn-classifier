import os

model_path = "model/pneumonia_mobilenetv2.h5"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

print("Model file exists ✅")
