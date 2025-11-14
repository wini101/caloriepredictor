# AI Food Calorie Predictor

This is a Streamlit app that predicts food dishes and estimates calories per serving using a Keras model trained on Food-101.

## Quick start

1. Create a Python environment and install dependencies:

```powershell
pip install -r requirements.txt
```

2. Make sure one of the model files is present in the project root:
- `best_food_model.h5` or `food_model.h5`

3. Start the app:

```powershell
streamlit run apppy.py
```

4. Open the URL shown by Streamlit (usually http://localhost:8501).

## Features
- Upload single images to get a top-3 prediction and calorie estimate.
- Debug mode shows raw scores and model shapes.
- Preprocessing options: resize-only, center-crop+resize, ImageNet normalization.
- Batch test: put images into `test_images/` and click `Run batch test` in the sidebar to run predictions and save `batch_results.csv`.

## Troubleshooting
- If TensorFlow or other packages fail to install on Windows, consider using a Conda environment or use a compatible TensorFlow wheel for your Python version.
- If model and labels mismatch, run `check_model_labels.py` to inspect shapes and label counts.

