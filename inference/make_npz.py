import glob, json, os, cv2, numpy as np

# -------- paths --------
clip_dir = "/home/brian/Desktop/frame_test_11"
out_npz = "/home/brian/Desktop/frame_test_11/my_sample.npz"

# -------- collect frames --------
frame_paths = sorted(glob.glob(os.path.join(clip_dir, "frame_*.jpg")))
frames = []
actions = []

seg_len = 20                                    # matches --segment_length

for i, fp in enumerate(sorted(frame_paths)[:seg_len]):
    # ------------- image ---------------------
    img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 64 / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    canvas = np.zeros((64, 64, 3), np.uint8)
    y0, x0 = (64 - nh) // 2, (64 - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = cv2.resize(img, (nw, nh), cv2.INTER_AREA)
    frames.append(canvas)

    # ------------- action --------------------
    if i < seg_len - 1:                         # skip the last frame
        action_file = os.path.join(
            os.path.dirname(fp),
            os.path.basename(fp).replace("frame_", "action_").replace(".jpg", ".json"))
        with open(action_file) as f:
            act = json.load(f)
        key_code = (act["virtual_key_codes"] or [0])[0]
        actions.append([act["dx"], act["dy"], act["wheel"], key_code, 0.0])

np.savez_compressed(
    out_npz,
    image=np.stack(frames, 0),
    action=np.stack(actions, 0).astype(np.float32)  # shape (2,5)
)
print("wrote", out_npz)

