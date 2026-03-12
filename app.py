import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="PlantPatrol AI", page_icon="🌿", layout="centered")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #2e7d32; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_plant_model():
    # Ensure 'plant_model.h5' is uploaded to your GitHub repo
    model = tf.keras.models.load_model('plant_model.h5')
    return model

# --- PREDICTION LOGIC ---
def predict_disease(image, model):
    # Convert image to RGB (important if someone uploads a PNG with transparency)
    img = image.convert("RGB")
    img = img.resize((224, 224)) 
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    classes = ['Apple Scab', 'Corn Rust', 'Potato Blight', 'Healthy']
    
    result = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return result, confidence

# --- APP UI ---
st.title("🌿 Plant Disease Diagnostic")
st.write("Instant AI diagnosis for small-scale farmers.")

with st.sidebar:
    st.header("How to use")
    st.info("1. Upload a clear photo.\n2. Ensure infection is visible.\n3. Get diagnosis.")

tab1, tab2 = st.tabs(["📸 Take Photo", "📁 Upload Image"])

with tab1:
    camera_image = st.camera_input("Scan your plant leaf")

with tab2:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

input_image = camera_image or uploaded_file

if input_image:
    image = Image.open(input_image)
    st.image(image, caption="Current Scan", use_container_width=True)
    
    if st.button("Diagnose Disease"):
        try:
            with st.spinner('Analyzing patterns...'):
                model = load_plant_model()
                label, score = predict_disease(image, model)
                
                st.success(f"### Result: {label}")
                st.write(f"**AI Confidence:** {score:.2f}%")
                
                if label != 'Healthy':
                    st.warning("⚠️ **Treatment:** Isolate the plant and apply organic fungicide.")
                else:
                    st.balloons()
                    st.info("✅ Your plant looks healthy!")
        except Exception as e:
            st.error(f"Error: {e}. Make sure 'plant_model.h5' is in your GitHub repo.")
