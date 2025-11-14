import h5py
import numpy as np

with h5py.File("food_c101_n1000_r384x384x3.h5", "r") as f:
    print("Dataset structure:")
    for key in f.keys():
        print(f"{key}: {f[key].shape}")
        
    if "labels" in f:
        labels = [label.decode('utf-8') for label in f["labels"][:]]
        print("\nLabels found:", len(labels))
        print("\nFirst 20 labels:")
        for i, label in enumerate(labels[:20]):
            print(f"{i}: {label}")