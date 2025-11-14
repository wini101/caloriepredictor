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


# -------------------------------------------------------------------
# APP CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="Food Predictor", page_icon="🍏", layout="wide")
st.title("AI Food Calorie Predictor")


# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        st.warning("TensorFlow not available: demo mode.")
        return None

    for fname in ("best_food_model.h5", "food_model.h5"):
        if os.path.exists(fname):
            try:
                return tf.keras.models.load_model(fname)
            except Exception as e:
                st.error(f"Error loading model: {e}")

    st.error("Model not found.")
    raise FileNotFoundError("No .h5 model found")

model = load_model()


# -------------------------------------------------------------------
# LOAD LABELS
# -------------------------------------------------------------------
def get_labels():
    # Food-101 labels from your fallback list
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


# -------------------------------------------------------------------
# FIXED PREPROCESSING (IMPORTANT!)
# YOUR MODEL IS 384×384 — ALWAYS USE THIS
# -------------------------------------------------------------------
TARGET_SIZE = (384, 384)

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize(TARGET_SIZE)

    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1,384,384,3)

    return arr


# -------------------------------------------------------------------
# SAFE PREDICT
# -------------------------------------------------------------------
def predict_array(arr):
    if model is None:
        # fallback demo mode
        n = len(labels)
        scores = np.zeros((1,n), dtype=np.float32)
        scores[0, np.random.randint(0,n)] = 1.0
        return scores

    return model.predict(arr, verbose=0)


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload food image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    # FIXED preprocessing
    arr = preprocess_image(img)

    # Predict
    preds = predict_array(arr)[0]
    idx_sorted = np.argsort(preds)[::-1]

    top1 = idx_sorted[0]
    conf_top1 = preds[top1] * 100
    pred_label = labels[top1]

    # show image
    st.image(img, width=300)

    st.subheader(f"Prediction: {pred_label.replace('_',' ').title()}")
    st.metric("Confidence", f"{conf_top1:.1f}%")

    # top 10 ranking
    rows = []
    for rank, j in enumerate(idx_sorted[:10], 1):
        rows.append({
            "rank": rank,
            "label": labels[j],
            "conf": f"{preds[j]*100:.1f}%"
        })

    st.write("### Top Predictions")
    st.table(rows)

else:
    st.info("Upload an image to begin.")
