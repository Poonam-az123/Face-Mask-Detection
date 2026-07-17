# pip install streamlit tensorflow pillow

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="😷 AI Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# ---------------- SESSION ---------------- #

if "open_camera" not in st.session_state:
    st.session_state.open_camera = False

# ---------------- LOAD TFLITE MODEL ---------------- #

interpreter = tf.lite.Interpreter(model_path="mask_final_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#E3F2FD,#FFFFFF);
}

.title{
text-align:center;
font-size:48px;
font-weight:bold;
color:#1565C0;
}

.subtitle{
text-align:center;
font-size:18px;
color:gray;
margin-bottom:25px;
}

div.stButton>button{
width:100%;
height:55px;
font-size:18px;
font-weight:bold;
border-radius:12px;
background:#1976D2;
color:white;
border:none;
}

div.stButton>button:hover{
background:#0D47A1;
}

[data-testid="stFileUploader"]{
border:2px dashed #1976D2;
border-radius:15px;
padding:15px;
background:white;
}

.footer{
text-align:center;
color:gray;
margin-top:30px;
font-size:16px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("😷 Face Mask Detection")

st.sidebar.success("""
### Features

✅ Upload Image

📸 Live Camera

🤖 TensorFlow Lite

⚡ Fast Prediction

🎯 High Accuracy
""")

# ---------------- TITLE ---------------- #

st.markdown("""
<div class='title'>
😷 AI Face Mask Detection
</div>

<div class='subtitle'>
Upload an image or use your camera to detect whether a person is wearing a face mask.
</div>
""", unsafe_allow_html=True)

# ---------------- PREDICTION FUNCTION ---------------- #

def predict(img):

    img = img.convert("RGB")
    img = img.resize((128,128))

    img_array = np.array(img,dtype=np.float32)/255.0
    img_array = np.expand_dims(img_array,axis=0)

    with st.spinner("🤖 AI is analyzing image..."):

        interpreter.set_tensor(
            input_details[0]["index"],
            img_array
        )

        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        )

    prob = float(prediction[0][0])

    if prob>0.5:

        st.error("## ❌ WITHOUT MASK")

        st.progress(prob)

        st.metric(
            "Confidence",
            f"{prob*100:.2f}%"
        )

    else:

        confidence=1-prob

        st.success("## ✅ WITH MASK")

        st.progress(confidence)

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

# ---------------- FILE UPLOAD ---------------- #

uploaded_file=st.file_uploader(
    "📂 Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    img=Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    predict(img)

st.markdown("---")

# ---------------- CAMERA ---------------- #

col1,col2=st.columns(2)

with col1:

    if st.button("📸 Open Camera"):
        st.session_state.open_camera=True

with col2:

    if st.button("❌ Close Camera"):
        st.session_state.open_camera=False

if st.session_state.open_camera:

    camera_image=st.camera_input("Take Photo")

    if camera_image:

        img=Image.open(camera_image)

        st.image(
            img,
            caption="Captured Image",
            use_container_width=True
        )

        predict(img)

        st.session_state.open_camera=False

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.markdown("""
<div class='footer'>

Made with ❤️ using <b>TensorFlow Lite</b> & <b>Streamlit</b>

</div>
""",unsafe_allow_html=True)