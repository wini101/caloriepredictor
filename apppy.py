import streamlit as st
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False

import numpy as np
try:
    import h5py
    H5PY_AVAILABLE = True
except:
    h5py = None
    H5PY_AVAILABLE = False

from PIL import Image
import os, io, csv

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    SKLEARN_AVAILABLE = True
except:
    confusion_matrix = None
    ConfusionMatrixDisplay = None
    SKLEARN_AVAILABLE = False


# ------------------------------------------------------------
# APP BASE
# ------------------------------------------------------------
st.set_page_config(page_title="Food Predictor", page_icon="🍏", layout="wide")
st.title("AI Food Calorie Predictor")


# ------------------------------------------------------------
# THEME (UNCHANGED)
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    :root{--accent:#10b981;--accent-dark:#057a55;--card-bg:#ffffff;--muted:#6b7280}
    .stApp { background: linear-gradient(180deg,#f6fffa,#ffffff); }
    .control-card { background: linear-gradient(180deg, #ecfdf5, #ffffff); border-radius:12px; padding:16px; }
    .result-card { background: linear-gradient(180deg,#ffffff,#f0fff7); border-radius:12px; padding:18px; box-shadow:0 8px 28px rgba(6,95,70,0.06); }
    .big-number { font-size:36px; font-weight:800; color:var(--accent-dark); }
    .muted { color:var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        st.warning("TensorFlow not available: Demo mode.")
        return None
    for fname in ("best_food_model.h5", "food_model.h5"):
        if os.path.exists(fname):
            try:
                return tf.keras.models.load_model(fname)
            except:
                pass
    st.error("No model found")
    return None

model = load_model()


# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------
@st.cache_resource
def get_labels():
    return ["apple_pie","baby_back_ribs","baklava","beef_carpaccio","beef_tartare","beet_salad",
    "beignets","bibimbap","bread_pudding","breakfast_burrito","bruschetta","caesar_salad",
    "cannoli","caprese_salad","carrot_cake","ceviche","cheese_plate","cheesecake","chicken_curry",
    "chicken_quesadilla","chicken_wings","chocolate_cake","chocolate_mousse","churros",
    "clam_chowder","club_sandwich","crab_cakes","creme_brulee","croque_madame","cup_cakes",
    "deviled_eggs","donuts","dumplings","edamame","eggs_benedict","escargots","falafel",
    "filet_mignon","fish_and_chips","foie_gras","french_fries","french_onion_soup","french_toast",
    "fried_calamari","fried_rice","frozen_yogurt","garlic_bread","gnocchi","greek_salad",
    "grilled_cheese_sandwich","grilled_salmon","guacamole","gyoza","hamburger",
    "hot_and_sour_soup","hot_dog","huevos_rancheros","hummus","ice_cream","lasagna",
    "lobster_bisque","lobster_roll_sandwich","macaroni_and_cheese","macarons","miso_soup",
    "mussels","nachos","omelette","onion_rings","oysters","pad_thai","paella","pancakes",
    "panna_cotta","peking_duck","pho","pizza","pork_chop","poutine","prime_rib",
    "pulled_pork_sandwich","ramen","ravioli","red_velvet_cake","risotto","samosa","sashimi",
    "scallops","seaweed_salad","shrimp_and_grits","spaghetti_bolognese","spaghetti_carbonara",
    "spring_rolls","steak","strawberry_shortcake","sushi","tacos","takoyaki","tiramisu",
    "tuna_tartare","waffles"]

labels = get_labels()


# ------------------------------------------------------------
# PREPROCESS
# ------------------------------------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    w, h = image.size
    m = min(w, h)
    image = image.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))
    image = image.resize((384, 384))
    arr = np.array(image).astype("float32") / 255.0
    return np.expand_dims(arr, 0)


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------
def predict_array(arr):
    if model is None:
        n = len(labels)
        out = np.zeros((1, n))
        out[0, np.random.randint(0,n)] = 1
        return out
    return model.predict(arr, verbose=0)


# ------------------------------------------------------------
# CALORIES
# ------------------------------------------------------------
cal_dict={"pizza":266,"burger":295,"samosa":310,"salad":120}


def get_cal(label):
    return cal_dict.get(label, 250)


# ------------------------------------------------------------
# UI LAYOUT
# ------------------------------------------------------------
col1, col2 = st.columns([0.45, 0.55], gap="medium")

with col1:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    st.subheader("Body Parameters")
    gender = st.radio("Gender", ["Male","Female"])
    age = st.number_input("Age", 10, 100, 25)
    weight = st.number_input("Weight (kg)", 20, 200, 55)
    activity = st.select_slider("Activity", ["Low","Moderate","High"])
    goal = st.selectbox("Goal", ["Lose","Maintain","Gain"])

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload food image", ["jpg","jpeg","png"])
    calc = st.button("Calculate", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


with col2:
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Your Result</h3>", unsafe_allow_html=True)

    if uploaded_file:
        img = Image.open(uploaded_file)
        arr = preprocess_image(img)

        with st.spinner("Analyzing..."):
            preds = predict_array(arr)
            scores = preds[0]
            idx = int(np.argmax(scores))
            conf = float(scores[idx]) * 100
            pred_label = labels[idx]

        colA, colB = st.columns(2)

        with colA:
            st.image(img, use_column_width=True)

        with colB:
            cal100 = get_cal(pred_label)
            portion = st.slider("Portion (g)", 50, 1000, 250)
            total = int(cal100 * portion / 100)

            st.markdown(f"<div class='big-number'>{total} kcal</div>", unsafe_allow_html=True)
            st.markdown(f"### {pred_label.replace('_',' ').title()}")
            st.metric("Confidence", f"{conf:.1f}%")

    st.markdown("</div>", unsafe_allow_html=True)





