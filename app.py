import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import cv2

# -------- PAGE CONFIGURATION --------
st.set_page_config(page_title="Plant Disease Detector 🌿", layout="wide")

# -------- LOAD MODEL --------
# -------- LOAD MODEL (FINAL FIX) --------
# -------- LOAD MODEL (FINAL FIX) --------
@st.cache_resource
def load_model():
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    # Build same architecture
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Load weights only (fixes ALL compatibility errors)
    model.load_weights("model/plant_disease_model.h5")

    return model


model = load_model()

classes = ["Early Blight","Healthy","Late Blight"]
# -------- HISTORY FILE --------
history_file = "history.json"

if not os.path.exists(history_file):
    with open(history_file,"w") as f:
        json.dump([], f)


def save_history(result):
    with open(history_file,"r") as f:
        data = json.load(f)

    data.append(result)

    with open(history_file,"w") as f:
        json.dump(data,f)


def load_history():
    with open(history_file,"r") as f:
        return json.load(f)


# -------- BLUR DETECTION --------
def is_blurry(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 100


# -------- DISEASE INFO --------
disease_info = {
    "Early Blight":{
        "causes":[
            "Fungal infection by Alternaria solani",
            "Warm humid weather",
            "Overwatering",
            "Poor air circulation"
        ],
        "treatment":[
            "Remove infected leaves",
            "Apply fungicide",
            "Improve plant spacing",
            "Avoid wet leaves"
        ]
    },

    "Late Blight":{
        "causes":[
            "Phytophthora infestans fungus",
            "Cool wet climate",
            "High humidity",
            "Infected nearby plants"
        ],
        "treatment":[
            "Remove infected plants",
            "Use fungicide spray",
            "Improve drainage",
            "Water at soil level"
        ]
    },

    "Healthy":{
        "causes":[
            "No disease symptoms",
            "Proper watering",
            "Good sunlight",
            "Healthy soil nutrients"
        ],
        "treatment":[
            "Continue normal care",
            "Monitor plant regularly",
            "Maintain watering schedule",
            "Keep garden clean"
        ]
    }
}

# -------- STYLE (YOUR ORIGINAL BACKGROUND KEPT) --------
st.markdown("""
<style>

.stApp{
background:linear-gradient(
rgba(0,0,0,0.6),
rgba(0,0,0,0.6)
),
url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6");
background-size:cover;
background-position:center;
}

.upload-box{
background:white;
padding:40px;
border-radius:15px;
text-align:center;
}

.result-box{
background:white;
padding:25px;
border-radius:15px;
text-align:center;
margin-top:20px;
}

.info-box{
background:white;
padding:15px;
border-radius:10px;
text-align:center;
margin-top:20px;
}

.box-heading{
color:black;
font-weight:bold;
font-size:22px;
}

.disease-text{
color:black;
font-size:28px;
font-weight:bold;
}

.points{
color:white;
font-size:18px;
margin-left:20px;
}

.history-card{
background:white;
padding:15px;
border-radius:10px;
margin-bottom:10px;
color:black;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)


# -------- PAGE STATE --------
if "page" not in st.session_state:
    st.session_state.page = "home"


# -------- NAVIGATION --------
col1,col2,col3 = st.columns([8,1,1])

with col1:
    st.markdown("<h2 style='color:white'>🌿 Plant Disease Detector</h2>", unsafe_allow_html=True)

with col2:
    if st.button("🏠 Home"):
        st.session_state.page="home"

with col3:
    if st.button("📜 History"):
        st.session_state.page="history"


# -------- PREDICTION --------
def predict(image):

    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image,axis=0)

    prediction = model.predict(image)
    confidence = np.max(prediction)

    if confidence < 0.7:
        return "Invalid Image"

    return classes[np.argmax(prediction)]


# -------- HOME PAGE --------
if st.session_state.page=="home":

    st.markdown("<h1 style='color:white'>Detect Plant Diseases Instantly 🌱</h1>", unsafe_allow_html=True)

    center = st.columns([2,4,2])[1]

    with center:

        st.markdown('<div class="upload-box">', unsafe_allow_html=True)

        file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

        if file:

            image = Image.open(file)
            st.image(image, width=350)

            # -------- BLUR CHECK --------
            if is_blurry(image):
                st.warning("⚠️ Image is too blurry. Please upload a clearer leaf image.")

            else:

                if st.button("Analyze 🌿"):

                    result = predict(image)

                    if result == "Invalid Image":
                        st.warning("⚠️ Please upload plant leaf image.")
                    else:
                        save_history(result)
                        st.session_state.result = result
                        st.session_state.page = "result"
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# -------- RESULT PAGE --------
elif st.session_state.page=="result":

    result = st.session_state.result

    st.markdown(f"""
    <div class="result-box">
    <div class="box-heading">Disease Detected</div>
    <div class="disease-text">{result}</div>
    </div>
    """, unsafe_allow_html=True)

    col1,col2 = st.columns(2)

    with col1:
        st.markdown("<div class='info-box'><div class='box-heading'>Causes</div></div>", unsafe_allow_html=True)
        for c in disease_info[result]["causes"]:
            st.markdown(f"<div class='points'>• {c}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='info-box'><div class='box-heading'>Treatment</div></div>", unsafe_allow_html=True)
        for t in disease_info[result]["treatment"]:
            st.markdown(f"<div class='points'>• {t}</div>", unsafe_allow_html=True)


# -------- HISTORY PAGE --------
elif st.session_state.page=="history":

    st.markdown("<h1 style='color:white'>Prediction History</h1>", unsafe_allow_html=True)

    history = load_history()

    if len(history)==0:
        st.info("No predictions yet")
    else:
        for item in reversed(history):
            st.markdown(f"""
            <div class="history-card">
            🌿 {item}
            </div>
            """, unsafe_allow_html=True)
