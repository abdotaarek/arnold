import numpy as np
from PIL import Image
import os

# Load the npz file
npz_file = np.load('/media/local/atarek/arnold/data_root/open_drawer/test/Steven-open_drawer-0-0-0.0-0.5-2-Mon_Jan_30_06:06:37_2023.npz', allow_pickle=True)
gt_frames = npz_file['gt']

# Directory to save frames
save_directory = "/media/local/atarek/npzFrames/opendrawer/test/1"

# Ensure the directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Iterate over frames and save images
for i, step in enumerate(gt_frames):
    obs_all = step['images']

    obs_view = obs_all[0]  # Change the number according to the view, 0 being front, 1 being base, 2 being left, 3 being wrist-bottom, 4 being wrist
    rgb_view = obs_view['rgb']

    # Convert to PIL Image and save
    img = Image.fromarray(rgb_view)
    img.save(os.path.join(save_directory, f"frame_{i}.png"))
