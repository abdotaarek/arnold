import cv2
import os
from pathlib import Path
from PIL import Image

# Define the path to the directory containing images
image_folder = Path('/media/local/atarek/3Dscene/5') 
video_path = Path('/media/local/atarek/3Dscene/videos/5_video.mp4')

# Ensure the output directory exists
video_path.parent.mkdir(parents=True, exist_ok=True)

# Fetch all image files from the directory
images = list(image_folder.glob('camera_4_frame_*.png')) # edit according to the images name
# Sort the images by frame number
images.sort(key=lambda x: int(x.stem.split('_')[-1]))

# Retrieve the dimensions of the first image
first_image = Image.open(str(images[0]))
width, height = first_image.size

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
video = cv2.VideoWriter(str(video_path), fourcc, 30, (width, height))

# Loop through all the image files
for img_path in images:
    img = cv2.imread(str(img_path))
    if img is not None:
        video.write(img)  # Write the image to the video
    else:
        print(f"Could not read image {img_path}")

# Release the video after finishing
video.release()
# Output the path to the saved video file
print(f"Video saved to {video_path}")
