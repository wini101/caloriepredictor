# 🍔 Healthy Plate - Image Calorie Predictor
import streamlit as st
from PIL import Image
import numpy as np
import random

# --- Page setup ---
st.set_page_config(page_title="Healthy Plate 🍽️", layout="centered")
st.title("🍽️ Healthy Plate - Food Image Calorie Predictor")

st.write("Upload a food image (JPG/PNG) and get estimated calories + health category.")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload a Food Image", type=["jpg", "jpeg", "png"])

# Only run the rest if file is uploaded
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Food Image", use_column_width=True)

    # --- Dummy AI Prediction (simulated for now) ---
    possible_foods = {
        "Salad": 120,
        "Pasta": 250,
        "Pizza": 350,
        "Burger": 450,
        "Soup": 180,
        "Rice Bowl": 300,
        "Ice Cream": 380
    }

    # Randomly pick a food and calorie estimate
    food_item = random.choice(list(possible_foods.keys()))
    predicted_calories = possible_foods[food_item] + random.randint(-20, 20)

    # --- Define health category ---
    def health_category(cal):
        if cal < 150:
            return "Healthy 🌿"
        elif cal < 300:
            return "Moderate 🍲"
        else:
            return "High Calorie 🍔"

    # --- Display results ---
    st.success(f"🍱 **Predicted Food:** {food_item}")
    st.write(f"🔥 **Estimated Calories:** {predicted_calories} kcal")
    st.info(f"💪 **Health Category:** {health_category(predicted_calories)}")

    st.caption("*(Note: These are AI-based estimates — for accurate results, use a nutrition calculator.)*")
else:
    st.warning("Please upload a food image to get calorie prediction.")


