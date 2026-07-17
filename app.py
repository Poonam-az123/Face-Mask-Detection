# pip install streamlit tensorflow pillow

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="😷 AI Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# ---------------- SESSION STATE ---------------- #

if "open_camera" not in st.session_state:
    st.session_state.open_camera = False

# ---------------- LOAD MODEL ---------------- #

try:
    


# Keras model load करा
    model = tf.keras.models.load_model("face_mask_model.keras")

# Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

# Save
    with open("mask_model.tflite", "wb") as f:
        f.write(tflite_model)
except:
    st.error("❌ Model file 'mask_final.keras' not found.")
    st.stop()

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.stApp{
    background:linear-gradient(135deg,#E3F2FD,#FFFFFF);
}

.title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#1565C0;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:20px;
}

div.stButton > button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

[data-testid="stFileUploader"]{
    border:2px dashed #1976D2;
    border-radius:12px;
    padding:15px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("😷 Face Mask Detection")

st.sidebar.success("""
### Features

✅ Upload Image

📸 Live Camera

🤖 Deep Learning

⚡ Instant Prediction

🎯 High Accuracy
""")

# ---------------- TITLE ---------------- #

st.markdown("""
<div class='title'>
😷 AI Face Mask Detection
</div>

<div class='subtitle'>
Upload an image or use your webcam to detect whether a face is wearing a mask.
</div>
""", unsafe_allow_html=True)

# ---------------- PREDICTION FUNCTION ---------------- #

def predict_image(img):

    img = img.convert("RGB")
    img = img.resize((128,128))

    img_array = image.img_to_array(img)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array,axis=0)

    with st.spinner("🤖 AI is analyzing image..."):
        prediction = model.predict(img_array,verbose=0)

    prob = float(prediction[0][0])

    if prob > 0.5:

        confidence = prob

        st.error("## ❌ WITHOUT MASK")

        st.progress(confidence)

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

    else:

        confidence = 1-prob

        st.success("## ✅ WITH MASK")

        st.progress(confidence)

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

# ---------------- FILE UPLOADER ---------------- #

uploaded_file = st.file_uploader(
    "📂 Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    predict_image(img)

st.markdown("---")

# ---------------- CAMERA ---------------- #

col1,col2 = st.columns(2)

with col1:

    if st.button("📸 Open Camera"):
        st.session_state.open_camera=True

with col2:

    if st.button("❌ Close Camera"):
        st.session_state.open_camera=False

if st.session_state.open_camera:

    camera_image = st.camera_input("Capture Image")

    if camera_image:

        img = Image.open(camera_image)

        st.image(
            img,
            caption="Captured Image",
            use_container_width=True
        )

        predict_image(img)

        st.session_state.open_camera=False

st.markdown("---")

# ---------------- FOOTER ---------------- #

st.markdown("""
<div class='footer'>

Made with ❤️ using <b>TensorFlow</b> & <b>Streamlit</b>

</div>
""",unsafe_allow_html=True)