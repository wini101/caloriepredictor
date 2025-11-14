
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

st.set_page_config(page_title="Food Predictor", page_icon="🍏", layout="wide")
st.title("AI Food Calorie Predictor")

# --- App theme (green) ---
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

@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        # TensorFlow not available in this environment — run in demo mode
        st.warning("TensorFlow not available: running in demo mode (predictions will be simulated).")
        return None
    for fname in ("best_food_model.h5", "food_model.h5"):
        if os.path.exists(fname):
            try:
                return tf.keras.models.load_model(fname)
            except Exception:
                pass
    st.error("No model found")
    raise FileNotFoundError("Model not found")

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
    if os.path.exists("food_dataset"):
        try:
            return sorted([d for d in os.listdir("food_dataset") if os.path.isdir(os.path.join("food_dataset",d))])
        except:
            pass
    return ["apple_pie","baby_back_ribs","baklava","beef_carpaccio","beef_tartare","beet_salad","beignets","bibimbap","bread_pudding","breakfast_burrito","bruschetta","caesar_salad","cannoli","caprese_salad","carrot_cake","ceviche","cheese_plate","cheesecake","chicken_curry","chicken_quesadilla","chicken_wings","chocolate_cake","chocolate_mousse","churros","clam_chowder","club_sandwich","crab_cakes","creme_brulee","croque_madame","cup_cakes","deviled_eggs","donuts","dumplings","edamame","eggs_benedict","escargots","falafel","filet_mignon","fish_and_chips","foie_gras","french_fries","french_onion_soup","french_toast","fried_calamari","fried_rice","frozen_yogurt","garlic_bread","gnocchi","greek_salad","grilled_cheese_sandwich","grilled_salmon","guacamole","gyoza","hamburger","hot_and_sour_soup","hot_dog","huevos_rancheros","hummus","ice_cream","lasagna","lobster_bisque","lobster_roll_sandwich","macaroni_and_cheese","macarons","miso_soup","mussels","nachos","omelette","onion_rings","oysters","pad_thai","paella","pancakes","panna_cotta","peking_duck","pho","pizza","pork_chop","poutine","prime_rib","pulled_pork_sandwich","ramen","ravioli","red_velvet_cake","risotto","samosa","sashimi","scallops","seaweed_salad","shrimp_and_grits","spaghetti_bolognese","spaghetti_carbonara","spring_rolls","steak","strawberry_shortcake","sushi","tacos","takoyaki","tiramisu","tuna_tartare","waffles"]

model = load_model()
labels = get_labels()

def predict_array(arr: np.ndarray) -> np.ndarray:
    """Wrapper to predict on an array. If TensorFlow/model not available,
    return a simulated prediction vector so the UI and debug tools still work.
    """
    if model is None or not TF_AVAILABLE:
        # Simulate a prediction vector with length equal to labels (or 101)
        n = len(labels) if labels else 101
        scores = np.zeros((1, n), dtype=np.float32)
        # pick a pseudo-random index based on image mean so results vary
        try:
            mean = float(np.mean(arr))
            idx = int((mean * 1000) % n)
        except Exception:
            idx = 0
        scores[0, idx] = 1.0
        return scores
    else:
        return model.predict(arr, verbose=0)

# ---------------------------
# Sidebar: preprocessing & batch test
# ---------------------------
preproc = st.sidebar.selectbox("Preprocessing", ["Resize only", "Center crop + resize", "Imagenet normalize"], index=0)
use_imagenet = st.sidebar.checkbox("Use ImageNet mean/std normalization", value=False)
batch_test = st.sidebar.button("Run batch test (test_images folder)")

def preprocess_pil(img: Image.Image, target_size, method="Resize only", imagenet=False):
    # img: PIL Image
    if method == "Center crop + resize":
        # center crop to square then resize
        w, h = img.size
        min_edge = min(w, h)
        left = (w - min_edge)//2
        top = (h - min_edge)//2
        img = img.crop((left, top, left+min_edge, top+min_edge))
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    if imagenet:
        # ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std
    return np.expand_dims(arr, axis=0)

def run_batch_test(folder="test_images"):
    results = []
    if not os.path.exists(folder):
        st.error(f"Batch folder '{folder}' not found in repo root.")
        return
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if not files:
        st.info("No images found in test_images folder.")
        return
    for fname in files:
        path = os.path.join(folder, fname)
        img = Image.open(path).convert("RGB")
        arr = preprocess_pil(img, target_size, method=preproc, imagenet=use_imagenet)
        preds = predict_array(arr)
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))*100
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        # try to infer ground truth from filename prefix 'label_...' or 'label-'
        gt = None
        base = os.path.splitext(fname)[0]
        if "_" in base:
            maybe = base.split("_")[0]
            if maybe in labels:
                gt = maybe
        if gt is None and "-" in base:
            maybe = base.split("-")[0]
            if maybe in labels:
                gt = maybe
        results.append({"file": fname, "pred": label, "conf": conf, "gt": gt})
    # save CSV
    out_csv = os.path.join(folder, "batch_results.csv")
    with open(out_csv, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["file","pred","conf","gt"]) 
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    st.success(f"Batch test complete — results saved to {out_csv}")
    # show simple accuracy if GT present
    gts = [r['gt'] for r in results if r['gt']]
    if gts:
        y_true = [r['gt'] for r in results if r['gt']]
        y_pred = [r['pred'] for r in results if r['gt']]
        acc = sum(1 for a,b in zip(y_true,y_pred) if a==b) / len(y_true)
        st.metric("Batch accuracy (inferred GT)", f"{acc*100:.2f}%")
        # confusion matrix for top labels
        if SKLEARN_AVAILABLE and MATPLOTLIB_AVAILABLE:
            labels_unique = sorted(list(set(y_true+y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=labels_unique)
            fig, ax = plt.subplots(figsize=(6,6))
            disp = ConfusionMatrixDisplay(cm, display_labels=labels_unique)
            disp.plot(ax=ax, xticks_rotation=90)
            st.pyplot(fig)
        else:
            st.info("Confusion matrix requires sklearn and matplotlib")
    st.download_button("Download batch CSV", data=open(out_csv,'rb').read(), file_name="batch_results.csv", mime="text/csv")

if batch_test:
    run_batch_test()
try:
    if model is not None:
        mi = model.input_shape
        if isinstance(mi,list): mi=mi[0]
        _,H,W,C=mi
        target_size = (int(W),int(H))
    else:
        target_size = (384,384)
except:
    target_size = (384,384)

cal_dict={"apple_pie":237,"baby_back_ribs":320,"baklava":430,"beef_carpaccio":180,"beignets":290,"bibimbap":112,"bread_pudding":150,"breakfast_burrito":305,"bruschetta":120,"caesar_salad":190,"cheeseburger":295,"chicken_curry":240,"chicken_wings":290,"chocolate_cake":370,"cup_cakes":305,"donuts":450,"dumplings":180,"fried_rice":333,"grilled_cheese_sandwich":350,"hamburger":295,"pizza":266,"sushi":150,"tacos":226,"ramen":180,"pad_thai":350,"steak":250,"spaghetti_carbonara":380,"french_fries":312,"ice_cream":207,"lasagna":132,"macaroni_and_cheese":370,"risotto":175,"spring_rolls":154,"tiramisu":290,"waffles":291,"pancakes":227,"omelette":154,"fish_and_chips":290}

def get_cal(label):
    if label in cal_dict: return cal_dict[label]
    l=label.lower()
    if "salad" in l: return 120
    if "ice" in l or "gelato" in l: return 210
    if "cake" in l or "brownie" in l or "cheesecake" in l: return 350
    if "pizza" in l: return 270
    if "burger" in l or "hamburger" in l: return 300
    if "fries" in l or "potato" in l: return 312
    if "rice" in l or "fried_rice" in l: return 330
    if "sushi" in l or "fish" in l: return 170
    if "pasta" in l or "spaghetti" in l or "lasagna" in l: return 360
    if "chicken" in l: return 230
    if "beef" in l or "steak" in l: return 250
    if "egg" in l: return 155
    if "donut" in l: return 452
    return 250

debug = st.sidebar.checkbox("Debug", key="debug")

# Main layout: left controls, right results (like the screenshot style, green)
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
        img_resized = img.resize(target_size)
        img_arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)

        with st.spinner("Analyzing..."):
            preds = predict_array(img_arr)
            scores = preds[0]
            top3_idx = np.argsort(scores)[-3:][::-1]
            idx = int(top3_idx[0])
            conf = float(scores[idx])*100

            if idx >= len(labels):
                st.error(f"Index {idx} out of range ({len(labels)} labels)")
            else:
                pred_label = labels[idx]
                top3 = [(labels[i], float(scores[i])*100) for i in top3_idx]

        # result layout: big calories number + breakdown
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

            # simple macro breakdown heuristics
            carbs_pct = 0.45
            protein_pct = 0.25
            fat_pct = 0.30
            carbs = int(total_kcal * carbs_pct / 4)
            protein = int(total_kcal * protein_pct / 4)
            fat = int(total_kcal * fat_pct / 9)
            st.markdown(f"**Macros:** {carbs} g carbs · {protein} g protein · {fat} g fat")

            # Visual macro percentages
            st.markdown("**Macro breakdown**")
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.write("Carbs")
                st.progress(min(100, int(carbs_pct*100)))
                st.caption(f"{int(carbs_pct*100)}% of kcal")
            with mcol2:
                st.write("Protein")
                st.progress(min(100, int(protein_pct*100)))
                st.caption(f"{int(protein_pct*100)}% of kcal")
            with mcol3:
                st.write("Fat")
                st.progress(min(100, int(fat_pct*100)))
                st.caption(f"{int(fat_pct*100)}% of kcal")

            # Suggested product cards (placeholders)
            st.markdown("---")
            st.markdown("**Suggested Products**")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown("<div style='border-radius:8px;padding:12px;background:#ffffff;box-shadow:0 4px 12px rgba(0,0,0,0.04)'>", unsafe_allow_html=True)
                st.markdown("<strong>Shape Shifter</strong>")
                st.markdown("<div class='muted'>Protein powder for lean muscle & recovery</div>", unsafe_allow_html=True)
                if st.button("View product 1"):
                    st.write("Product 1 clicked")
                st.markdown("</div>", unsafe_allow_html=True)
            with pcol2:
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
            if model is not None:
                st.write({"input": model.input_shape, "output": model.output_shape, "labels": len(labels)})
            else:
                st.write({"input": "N/A (model not available)", "output": "N/A (model not available)", "labels": len(labels)})
            idx_sorted = np.argsort(scores)[::-1]
            rows = [{"rank": i, "label": labels[j] if j < len(labels) else f"class_{j}", "conf": f"{scores[j]*100:.1f}%"} for i, j in enumerate(idx_sorted[:20], 1)]
            st.table(rows)
    else:
        # placeholder result card when no image
        st.markdown("<div style='display:flex; gap:20px'>", unsafe_allow_html=True)
        st.markdown("<div><div class='big-number'>1890 kcal</div><div class='muted'>Suggested amount of calories per day</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Adjust Protein")
        st.slider("Protein target", 0, 200, 80)
    st.markdown("</div>", unsafe_allow_html=True)



