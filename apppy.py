# 🍽️Healthy Plate - Food Calorie Predictor
import streamlit as st
from PIL import Image
import numpy as np
import random
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

# --- Page Setup ---
st.set_page_config(page_title="Healthy Plate 🍽️", layout="centered")
st.title("🌍 Healthy Plate -  Food Calorie Predictor")
st.write("Upload any food image — Indian, Western, or international — and get estimated calories with health rating.")

# --- Upload Section ---
uploaded_file = st.file_uploader("📸 Upload a Food Image", type=["jpg", "jpeg", "png"])

# Load model only once
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# --- Food Calorie Data (Indian + Global) ---
food_calories = {
    # Indian foods
    "Roti": 120, "Dal": 180, "Paneer Curry": 300, "Biryani": 450, "Dosa": 200, "Idli": 100,
    "Samosa": 250, "Poha": 180, "Chole": 350, "Rajma": 320, "Pulao": 300, "Aloo Paratha": 350,
    "Curd Rice": 280, "Upma": 220, "Pav Bhaji": 400,

    # Western foods
    "Pizza": 350, "Burger": 450, "Pasta": 300, "Sandwich": 250, "Salad": 150, "Soup": 180,
    "Fries": 400, "Steak": 500, "Ice Cream": 380, "Sushi": 220, "Taco": 280, "Donut": 310
}

# Mapping model predictions to Indian or known foods
name_mapping = {
    "bread": "Roti", "curry": "Dal", "rice": "Biryani", "pancake": "Dosa",
    "sandwich": "Sandwich", "cream": "Paneer Curry", "omelet": "Egg Curry",
    "ice_cream": "Ice Cream", "soup": "Soup", "pizza": "Pizza", "burger": "Burger",
    "spaghetti": "Pasta", "fries": "Fries", "salad": "Salad", "burrito": "Taco"
}

def health_category(cal):
    if cal < 150:
        return "Healthy 🌿"
    elif cal < 300:
        return "Moderate 🍲"
    else:
        return "High Calorie 🍔"

# --- Prediction Logic ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🍱 Uploaded Food Image", use_column_width=True)

    img = keras_image.load_img(uploaded_file, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]
    raw_food = decoded[0][1].lower()

    # Map to known food name
    food_item = name_mapping.get(raw_food, raw_food.title())

    # Get calorie value
    predicted_calories = food_calories.get(food_item, random.randint(150, 500))

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🍽️ Food", food_item)
    with col2:
        st.metric("🔥 Calories", f"{predicted_calories} kcal")
    with col3:
        st.metric("💪 Category", health_category(predicted_calories))
    st.markdown("---")

    st.caption("⚠️ AI-based food recognition — calorie estimates may vary depending on portion size and recipe.")
else:
    st.info("Please upload a food image to get calorie prediction.")





