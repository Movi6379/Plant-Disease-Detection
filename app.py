import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Plant Disease AI", page_icon="🌿")

# --- CSS FOR UI ---
st.markdown("""
    <style>
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 10px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED MODEL LOADING (FIXED) ---
@st.cache_resource
def get_model():
    import tensorflow as tf
    # Ensure the file name matches your .h5 file exactly
    model_path = 'plant_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    else:
        return None

# Load the model once
with st.spinner("Initializing AI Engine..."):
    model = get_model()

# --- PREDICTION LOGIC ---
def predict_disease(image, model):
    # 1. Resize image to match model input (224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # 2. Convert to array and Normalize (0 to 1 or -1 to 1 based on your training)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # 3. Reshape for model (Batch size, Height, Width, Channels)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # 4. Predict
    prediction = model.predict(data)
    return prediction

# --- MAIN UI ---
st.title("🌿 Plant Disease Diagnostic")

if model is None:
    st.error("Model file 'plant_model.h5' not found in repository!")
    st.stop()

uploaded_file = st.file_uploader("Upload a leaf photo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    if st.button("Run Diagnosis"):
        with st.spinner("Analyzing Leaf Pattern..."):
            try:
                result = predict_disease(image, model)
                
                classes = ['Apple Scab', 'Corn Rust', 'Potato Blight', 'Healthy']
                idx = np.argmax(result)
                confidence = result[0][idx] * 100
                
                # UI Results
                st.divider()
                st.success(f"### Result: {classes[idx]}")
                st.progress(int(confidence))
                st.write(f"**Confidence Level:** {confidence:.2f}%")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

else:
    st.info("Please upload a clear image of a plant leaf to start the diagnosis.")
