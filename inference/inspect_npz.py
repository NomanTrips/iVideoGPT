import numpy as np

data = np.load("/home/brian/Desktop/frame_test_9/my_sample.npz", allow_pickle=True)

# Check if it's a dict-like npz or a plain array
if hasattr(data, 'files'):
    print("Keys found:", data.files)
    for key in data.files:
        arr = data[key]
        print(f"Key: {key}, Shape: {arr.shape}, Dtype: {arr.dtype}")
else:
    print(f"Loaded array directly. Shape: {data.shape}, Dtype: {data.dtype}")
