import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Plant Disease AI", page_icon="🌿")

# --- CSS FOR UI ---
st.markdown("""
    <style>
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- LIGHTWEIGHT PREDICTION ---
def predict_disease(image):
    # 1. Resize image to match model input (224x224)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # 2. Turn image into numpy array
    image_array = np.asarray(image)
    
    # 3. Normalize image (scales pixels to between -1 and 1)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # 4. Load your model only when button is clicked to save memory
    # We use a try/except so the app doesn't show the "Oh No" face
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('plant_model.h5', compile=False)
        
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        prediction = model.predict(data)
        return prediction
    except Exception as e:
        return f"Error: {e}"

# --- MAIN UI ---
st.title("🌿 Plant Disease Diagnostic")

uploaded_file = st.file_uploader("Upload a leaf photo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    if st.button("Run Diagnosis"):
        with st.spinner("Analyzing..."):
            result = predict_disease(image)
            
            if isinstance(result, str):
                st.error("Deep Learning module is still loading on the server. Please wait 1 minute and try again.")
                st.info("Tip: Make sure 'tensorflow-cpu' is in your requirements.txt")
            else:
                classes = ['Apple Scab', 'Corn Rust', 'Potato Blight', 'Healthy']
                idx = np.argmax(result)
                confidence = result[0][idx] * 100
                
                st.success(f"### Result: {classes[idx]}")
                st.write(f"Confidence: {confidence:.2f}%")
