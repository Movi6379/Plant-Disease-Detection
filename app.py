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
            # compile=False is important for loading models across different TF versions
            return tf.keras.models.load_model(model_path, compile=False)
        return "Model file 'plant_model.h5' not found in repository."
    except ImportError:
        return "TensorFlow is not installed. Please ensure 'tensorflow-cpu' is in requirements.txt."

# Initialize model
result_of_load = load_my_model()

st.title("🌿 Plant Disease Diagnostic")

if isinstance(result_of_load, str):
    st.error(result_of_load)
    st.stop()
else:
    model = result_of_load

# --- MAIN UI ---
uploaded_file = st.file_uploader("Upload a leaf photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    if st.button("Run Diagnosis"):
        with st.spinner("Analyzing..."):
            # 1. Image Preprocessing (Matches most PlantVillage-style models)
            size = (224, 224) 
            image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_resized)
            
            # Normalize to [-1, 1] (Standard for many Keras models)
            normalized_img = (img_array.astype(np.float32) / 127.5) - 1
            
            # Create batch axis
            batch_data = np.expand_dims(normalized_img, axis=0)
            
            # 2. Prediction
            prediction = model.predict(batch_data)
            idx = np.argmax(prediction)
            confidence = prediction[0][idx] * 100
            
            # 3. Class Labels (UPDATE THESE to match your specific model's classes)
            classes = ['Apple Scab', 'Corn Rust', 'Potato Blight', 'Healthy']
            
            # 4. Display Results
            st.divider()
            if classes[idx] == 'Healthy':
                st.success(f"### Result: {classes[idx]}")
            else:
                st.warning(f"### Detected: {classes[idx]}")
                
            st.write(f"**Confidence Level:** {confidence:.2f}%")
            st.progress(int(confidence))

else:
    st.info("Please upload a leaf image to begin.")
