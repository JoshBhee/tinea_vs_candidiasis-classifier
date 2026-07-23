import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(page_title="Tinea vs Candidiasis Classifier", layout="centered")

# ---------------------------
# Constants
# ---------------------------
IMG_SIZE = (160, 160)
CLASS_NAMES = ["Candidiasis", "Tinea"]  # index 0, index 1 - update if your order differs
MODEL_PATH = "skin_model.h5"

# ---------------------------
# Load model (cached so it only loads once)
# ---------------------------
@st.cache_resource
def load_classifier():
    model = load_model(MODEL_PATH)
    return model

model = load_classifier()

# ---------------------------
# Preprocessing function
# ---------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------------------
# App UI
# ---------------------------
st.title("Tinea vs Candidiasis Classifier")
st.write(
    "Upload a skin lesion image to classify it as **Tinea** or **Candidiasis**. "
    "This tool is for educational/demonstration purposes only and is not a medical diagnosis."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        # Binary output: single sigmoid unit
        if prediction.shape[-1] == 1:
            confidence = float(prediction[0][0])
            predicted_class = CLASS_NAMES[1] if confidence > 0.5 else CLASS_NAMES[0]
            confidence_score = confidence if confidence > 0.5 else 1 - confidence
        else:
            # Softmax with 2 units
            predicted_index = int(np.argmax(prediction[0]))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence_score = float(np.max(prediction[0]))

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence_score * 100:.2f}%")

    st.progress(confidence_score)

    st.caption(
        "Disclaimer: This is a student project demo, not a substitute for professional medical advice."
    )
else:
    st.info("Please upload an image file to get a prediction.")