# 🍛 Healthy Plate - Global + Indian Food Calorie Predictor
import streamlit as st
from PIL import Image
import numpy as np
import random

st.set_page_config(page_title="Healthy Plate 🍽️", layout="centered")
st.title("🍽️ Healthy Plate - Food Image Calorie Predictor")

st.write("Upload a food image (JPG/PNG) and get estimated calories + health category. Works for Indian & Global foods!")

uploaded_file = st.file_uploader("📸 Upload a Food Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Food Image", use_column_width=True)

    # --- Expanded food list ---
    possible_foods = {
        # Indian foods
        "Chole Bhature": 500,
        "Dal Chawal": 320,
        "Rajma Chawal": 380,
        "Chole Bhature": 520,
        "Paneer Butter Masala": 430,
        "Masala Dosa": 350,
        "Idli Sambar": 210,
        "Biryani": 480,
        "Roti & Sabzi": 280,
        "Samosa": 260,
        "Poha": 180,
        "Upma": 190,

        # Global foods
        "Salad": 120,
        "Pasta": 250,
        "Pizza": 350,
        "Burger": 450,
        "Soup": 180,
        "Rice Bowl": 300,
        "Ice Cream": 380,
        "Sandwich": 230,
        "Steak": 500,
        "Noodles": 320
    }

    food_item = random.choice(list(possible_foods.keys()))
    predicted_calories = possible_foods[food_item] + random.randint(-20, 20)

    def health_category(cal):
        if cal < 150:
            return "🥦 Very Healthy"
        elif cal < 300:
            return "🍲 Moderate"
        else:
            return "🍔 High Calorie"

    st.success(f"🍱 **Predicted Food:** {food_item}")
    st.write(f"🔥 **Estimated Calories:** {predicted_calories} kcal")
    st.info(f"💪 **Health Category:** {health_category(predicted_calories)}")

    st.caption("*(Note: This is a simulated AI estimate. For real nutrition data, use a verified calorie tracker.)*")
else:
    st.warning("Please upload a food image to get calorie prediction.")









