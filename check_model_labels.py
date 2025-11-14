import os
import h5py
import tensorflow as tf
import numpy as np

print("\n🔍 Checking dataset and model alignment...")


h5path = "food_c101_n1000_r384x384x3.h5"
print(f"\n📊 Dataset ({h5path}):")
if os.path.exists(h5path):
    with h5py.File(h5path, "r") as f:
        print("Available keys:", list(f.keys()))
        if "category_names" in f:
            names = [n.decode("utf-8") for n in f["category_names"][:]]
            print(f"Found {len(names)} labels in category_names")
            print("First 10 labels:", names[:10])
        elif "labels" in f:
            names = [n.decode("utf-8") for n in f["labels"][:]]
            print(f"Found {len(names)} labels in labels dataset")
            print("First 10 labels:", names[:10])
        else:
            print("❌ No labels found in dataset!")
            names = []
else:
    print(f"❌ Dataset file not found!")
    names = []


print("\n🤖 Model check:")
for model_fname in ["best_food_model.h5", "food_model.h5"]:
    if os.path.exists(model_fname):
        try:
            print(f"\nLoading {model_fname}...")
            model = tf.keras.models.load_model(model_fname)
            
            
            output_shape = model.output_shape
            print(f"Model output shape: {output_shape}")
            
            if output_shape:
                num_classes = output_shape[-1]  # Last dimension is number of classes
                print(f"Number of output classes: {num_classes}")
                
                if len(names) > 0:
                    if num_classes == len(names):
                        print("✅ Model output matches number of labels!")
                    else:
                        print(f"❌ Mismatch: Model expects {num_classes} classes but found {len(names)} labels")
            
                        print("\nModel Summary:")
            model.summary()
            
        except Exception as e:
            print(f"❌ Error loading {model_fname}: {e}")
    else:
        print(f"\n{model_fname} not found")