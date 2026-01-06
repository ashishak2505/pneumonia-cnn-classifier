import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# App title
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to predict Pneumonia.")

# Load model
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "pneumonia_mobilenetv2.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()

IMG_SIZE = (224, 224)

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", width=300)

    # Preprocess image
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)[0][0]

    pneumonia_prob = float(prediction)
    normal_prob = 1 - pneumonia_prob

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🫁 Pneumonia Probability", f"{pneumonia_prob * 100:.2f}%")

    with col2:
        st.metric("🫀 Normal Probability", f"{normal_prob * 100:.2f}%")
# Progress bars
    st.write("Confidence Visualization")
    st.progress(pneumonia_prob)
    st.progress(normal_prob)

# Final decision
    if pneumonia_prob > 0.5:
        st.error("⚠️ Pneumonia Detected")
    else:
        st.success("✅ Normal")

    st.caption("⚠️ This tool is for educational purposes only and not a medical diagnosis.")
