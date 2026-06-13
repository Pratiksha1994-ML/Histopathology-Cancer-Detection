import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load Model
model = tf.keras.models.load_model("best_model_v2.keras")

st.title("Histopathology Cancer Detection")
st.write("Upload a histopathology image to predict Benign or Malignant.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        st.error(
            f"Prediction: Malignant\nConfidence: {confidence*100:.2f}%"
        )
    else:
        st.success(
            f"Prediction: Benign\nConfidence: {(1-confidence)*100:.2f}%"
        )