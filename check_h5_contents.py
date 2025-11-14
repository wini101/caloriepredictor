import h5py

with h5py.File("food_c101_n1000_r384x384x3.h5", "r") as f:
    print("📂 Keys inside the file:")
    for key in f.keys():
        print("-", key)
