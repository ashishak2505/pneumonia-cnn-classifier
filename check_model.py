import os

MODEL_PATH = "model/pneumonia_mobilenetv2.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

print("Model file exists ✅")
