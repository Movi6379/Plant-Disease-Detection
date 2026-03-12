import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os

st.set_page_config(page_title="Plant Disease AI", page_icon="🌿")

# --- SAFE LOADING ---
@st.cache_resource
def load_my_model():
    try:
        import tensorflow as tf
        model_path = 'plant_model.h5'
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path, compile=False)
        return "Model file not found."
    except ImportError:
        return "TensorFlow is not installed. Please check requirements.txt."

# Initialize
result_of_load = load_my_model()

st.title("🌿 Plant Disease Diagnostic")

# Check if model loaded or if it's an error string
if isinstance(result_of_load, str):
    st.error(result_of_load)
    st.info("System is likely still installing dependencies. Please wait 2-3 minutes and refresh.")
    st.stop()
else:
    model = result_of_load

# --- REST OF YOUR UI CODE ---
uploaded_file = st.file_uploader("Upload leaf...", type=["jpg", "jpeg", "png"])
# ... (rest of your prediction logic)
