
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🔬",
    layout="centered"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "skin_cancer_cnn.keras"
LABELS_PATH = BASE_DIR / "class_names.json"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return [labels[str(i)] for i in range(len(labels))]

st.title("🔬 Skin Cancer Image Classification")
st.write(
    "Upload a skin lesion image to see the model's predicted class."
)

st.warning(
    "Educational project only. This model is not reliable enough "
    "for medical diagnosis or treatment decisions. Please consult "
    "a qualified healthcare professional about any concerning lesion."
)

uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            model = load_model()
            class_names = load_class_names()

            resized_image = image.resize(IMG_SIZE)
            image_array = np.array(resized_image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            probabilities = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            predicted_class = class_names[predicted_index]
            confidence = float(probabilities[predicted_index])

        st.subheader("Model output")
        st.write("**Predicted class:**", predicted_class)
        st.write(f"**Model confidence:** {confidence:.1%}")

        st.caption(
            "Confidence is the model's score, not the probability "
            "that you have cancer. This model may misclassify images."
        )

        st.subheader("Scores for all classes")
        for name, score in sorted(
            zip(class_names, probabilities),
            key=lambda item: item[1],
            reverse=True
        ):
            st.write(f"{name}: {score:.1%}")
            st.progress(float(score))
