from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("food_c101_n1000_r384x384x3.h5")
print("✅ Model loaded successfully!")

img_path = "test.jpg"  # put any food image in this folder
img = image.load_img(img_path, target_size=(384, 384))
x = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

pred = model.predict(x)
print("Predicted class index:", np.argmax(pred))
