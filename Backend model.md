# Plant-Disease-Detection
# Traning a machine in the object dection model, using the plant disease
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Allow your frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model('plant_model.h5')
CLASSES = ['Apple Scab', 'Corn Rust', 'Potato Blight', 'Healthy']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and resize image
    data = await file.read()
    image = Image.open(io.BytesIO(data)).resize((224, 224))
    
    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    result = CLASSES[np.argmax(preds)]
    
    return {"diagnosis": result, "confidence": f"{np.max(preds)*100:.2f}%"}
