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
    .app-title { font-size:28px; font-weight:700; color:var(--accent-dark); }
    .panel { background: linear-gradient(180deg,#ffffff,#f8fffb); padding:18px; border-radius:12px; box-shadow:0 6px 20px rgba(6,95,70,0.06); }
    .control-card { background: linear-gradient(180deg, #ecfdf5, #ffffff); border-radius:12px; padding:16px; }
    .result-card { background: linear-gradient(180deg,#ffffff,#f0fff7); border-radius:12px; padding:18px; box-shadow:0 8px 28px rgba(6,95,70,0.06); }
    .big-number { font-size:36px; font-weight:800; color:var(--accent-dark); }
    .muted { color:var(--muted); }
    .stButton>button { background-color: var(--accent); border:none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# MODEL LOADING (UNCHANGED)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        st.warning("TensorFlow not available: demo mode.")
        return None
    for fname in ("best_food_model.h5", "food_model.h5"):
        if os.path.exists(fname):
            try:
                return tf.keras.models.load_model(fname)
            except:
                pass
    st.error("No model found")
    raise FileNotFoundError("Model not found")

model = load_model()


# ------------------------------------------------------------
# LABELS (UNCHANGED)
# ------------------------------------------------------------
@st.cache_resource
def get_labels():
    if H5PY_AVAILABLE and os.path.exists("food_c101_n1000_r384x384x3.h5"):
        try:
            with h5py.File("food_c101_n1000_r384x384x3.h5","r") as f:
                if "category_names" in f:
                    return [l.decode('utf-8') for l in f["category_names"][:]]
                if "labels" in f:
                    return [l.decode('utf-8') for l in f["labels"][:]]
        except:
            pass

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
# FIXED PREDICT (ONLY FIX: 384×384 PREPROCESSING)
# ------------------------------------------------------------
def predict_array(arr):
    if model is None:
        n = len(labels)
        scores = np.zeros((1,n), dtype=np.float32)
        scores[0, np.random.randint(0,n)] = 1
        return scores
    return model.predict(arr, verbose=0)


# ------------------------------------------------------------
# FIXED IMAGE PREPROCESSING (THIS WAS THE ONLY BUG)
# ------------------------------------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((384, 384))  # FIXED SIZE FOR YOUR MODEL
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ------------------------------------------------------------
# CALORIE DICTIONARY (UNCHANGED)
# ------------------------------------------------------------
cal_dict={"apple_pie":237,"baby_back_ribs":320,"baklava":430,"beef_carpaccio":180,"beignets":290,"bibimbap":112,
"bread_pudding":150,"breakfast_burrito":305,"bruschetta":120,"caesar_salad":190,"cheeseburger":295,"chicken_curry":240,
"chicken_wings":290,"chocolate_cake":370,"cup_cakes":305,"donuts":450,"dumplings":180,"fried_rice":333,
"grilled_cheese_sandwich":350,"hamburger":295,"pizza":266,"sushi":150,"tacos":226,"ramen":180,"pad_thai":350,
"steak":250,"spaghetti_carbonara":380,"french_fries":312,"ice_cream":207,"lasagna":132,"macaroni_and_cheese":370,
"risotto":175,"spring_rolls":154,"tiramisu":290,"waffles":291,"pancakes":227,"omelette":154,"fish_and_chips":290}

def get_cal(label):
    if label in cal_dict:
        return cal_dict[label]
    return 250


# ------------------------------------------------------------
# DEBUG SWITCH (UNCHANGED)
# ------------------------------------------------------------
debug = st.sidebar.checkbox("Debug", key="debug")


# ------------------------------------------------------------
# MAIN UI (UNCHANGED)
# ------------------------------------------------------------
col1, col2 = st.columns([0.45, 0.55], gap="medium")

with col1:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    st.subheader("Body Parameters")
    gender = st.radio("Gender", options=["Male", "Female"], index=0, horizontal=True)
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    activity = st.select_slider("Activity Level", options=["Low", "Moderate", "High", "Very High"], value="Moderate")
    goal = st.selectbox("Goal", options=["Lose", "Maintain", "Gain"], index=1)

    st.markdown("---")
    st.caption("Upload a food image to get a prediction and calorie estimate")

    uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])
    calc = st.button("Calculate", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


with col2:
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center'>", unsafe_allow_html=True)
    st.markdown("<div><h3 style='margin:0'>Your Result</h3><div class='muted'>Estimated calories and macros</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        
        # -----------------------
        # FIXED: correct preprocess
        # -----------------------
        img_arr = preprocess_image(img)

        with st.spinner("Analyzing..."):
            preds = predict_array(img_arr)
            scores = preds[0]

            top3_idx = np.argsort(scores)[-3:][::-1]
            idx = int(top3_idx[0])
            conf = float(scores[idx]) * 100

            pred_label = labels[idx]

        # UI BELOW IS EXACTLY SAME AS YOUR ORIGINAL
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.image(img, use_column_width=True)

        with col_b:
            cal_per100 = get_cal(pred_label)
            portion = st.slider("Portion size (g)", 50, 1000, 250, 25)
            total_kcal = int(cal_per100 * portion / 100)

            st.markdown(f"<div class='big-number'>{total_kcal} kcal</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='muted'>Estimated for {portion} g — {cal_per100} kcal / 100g</div>", unsafe_allow_html=True)
            st.markdown("---")

            st.markdown(f"### {pred_label.replace('_',' ').title()}")
            st.metric("Confidence", f"{conf:.1f}%")

            carbs_pct = 0.45
            protein_pct = 0.25
            fat_pct = 0.30

            carbs = int(total_kcal * carbs_pct / 4)
            protein = int(total_kcal * protein_pct / 4)
            fat = int(total_kcal * fat_pct / 9)

            st.markdown(f"**Macros:** {carbs} g carbs · {protein} g protein · {fat} g fat")

            st.markdown("**Macro breakdown**")
            mc1, mc2, mc3 = st.columns(3)

            with mc1:
                st.write("Carbs")
                st.progress(int(carbs_pct * 100))
                st.caption(f"{int(carbs_pct * 100)}%")

            with mc2:
                st.write("Protein")
                st.progress(int(protein_pct * 100))
                st.caption(f"{int(protein_pct * 100)}%")

            with mc3:
                st.write("Fat")
                st.progress(int(fat_pct * 100))
                st.caption(f"{int(fat_pct * 100)}%")

            st.markdown("---")
            st.markdown("**Suggested Products**")

            p1, p2 = st.columns(2)

            with p1:
                st.markdown("<div style='border-radius:8px;padding:12px;background:#ffffff;box-shadow:0 4px 12px rgba(0,0,0,0.04)'>", unsafe_allow_html=True)
                st.markdown("<strong>Shape Shifter</strong>")
                st.markdown("<div class='muted'>Protein powder for lean muscle & recovery</div>", unsafe_allow_html=True)
                if st.button("View product 1"):
                    st.write("Product 1 clicked")
                st.markdown("</div>", unsafe_allow_html=True)

            with p2:
                st.markdown("<div style='border-radius:8px;padding:12px;background:#ffffff;box-shadow:0 4px 12px rgba(0,0,0,0.04)'>", unsafe_allow_html=True)
                st.markdown("<strong>Energy Boost</strong>")
                st.markdown("<div class='muted'>Pre-workout for sustained energy</div>", unsafe_allow_html=True)
                if st.button("View product 2"):
                    st.write("Product 2 clicked")
                st.markdown("</div>", unsafe_allow_html=True)

        if conf > 70:
            st.balloons()

        if debug:
            st.markdown("---")
            st.write({"input": model.input_shape, "output": model.output_shape, "labels": len(labels)})
            idx_sorted = np.argsort(scores)[::-1]
            rows = [{"rank": i, "label": labels[j], "conf": f"{scores[j]*100:.1f}%"} for i, j in enumerate(idx_sorted[:20], 1)]
            st.table(rows)

    else:
        st.markdown("<div style='display:flex; gap:20px'>", unsafe_allow_html=True)
        st.markdown("<div><div class='big-number'>1890 kcal</div><div class='muted'>Suggested daily calories</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Adjust Protein")
        st.slider("Protein target", 0, 200, 80)
        st.markdown("</div>", unsafe_allow_html=True)
