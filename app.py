import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps

# --- LOAD ONNX MODEL ---
@st.cache_resource
def load_onnx_model():
    # This replaces tf.keras.models.load_model
    session = ort.InferenceSession("plant_model.onnx")
    return session

def predict_with_onnx(image, session):
    # Preprocess image
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    # Most models expect values between 0-1 or -1 to 1
    img_array = img_array / 255.0 

    # Run Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    
    classes = ['Apple Scab', 'Corn Rust', 'Potato Blight', 'Healthy']
    prediction = outputs[0]
    index = np.argmax(prediction)
    
    return classes[index], float(np.max(prediction) * 100)

# --- UI LOGIC ---
st.title("🌿 Ultra-Light Plant Diagnostic")
image_file = st.file_uploader("Upload Leaf", type=["jpg", "png"])

if image_file:
    img = Image.open(image_file).convert("RGB")
    st.image(img, width=300)
    
    if st.button("Analyze"):
        session = load_onnx_model()
        label, conf = predict_with_onnx(img, session)
        st.success(f"Result: {label} ({conf:.1f}%)")
