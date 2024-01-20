import torch
import clip
from PIL import Image
import numpy as np
import os
import glob

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_clip_embeddings(image_paths):
    # Process and stack images
    images = [preprocess(Image.open(path)).unsqueeze(0) for path in image_paths if os.path.isfile(path)]
    images = torch.cat(images, dim=0).to(device)

    # Generate embeddings
    with torch.no_grad():
        embeddings = model.encode_image(images)
    return embeddings

def find_closest_match(test_embedding, train_embeddings):
    distances = torch.nn.functional.pairwise_distance(test_embedding, train_embeddings)
    closest_idx = torch.argmin(distances).item()
    closest_distance = distances[closest_idx].item()
    return closest_idx, closest_distance

# Paths for three training episodes
train_episode_dirs = [
    '/media/local/atarek/npzFrames/opendrawer/train/1',
    '/media/local/atarek/npzFrames/opendrawer/train/2',
    '/media/local/atarek/npzFrames/opendrawer/train/3'
]

# Path for test episode
test_episode_dir = '/media/local/atarek/npzFrames/opendrawer/test/1'

# Collect image paths
train_episode_paths = [img for dir in train_episode_dirs for img in glob.glob(os.path.join(dir, "*.png"))]
test_episode_paths = glob.glob(os.path.join(test_episode_dir, "*.png"))

# Get embeddings
train_embeddings = get_clip_embeddings(train_episode_paths)
test_embeddings = get_clip_embeddings(test_episode_paths)

# Find closest match for each test frame
frequency = 1 # Change this to your desired frequency

# Iterate over test embeddings and find closest match at specified frequency
for i, test_embedding in enumerate(test_embeddings):
    if i % frequency == 0:
        closest_match_idx, closest_distance = find_closest_match(test_embedding.unsqueeze(0), train_embeddings)
        # Print the embeddings and closest match details
        print(f"Test Embedding for Image {i}: {test_embedding}")
        print(f"Closest Match Embedding (Train Image {closest_match_idx}): {train_embeddings[closest_match_idx]}")
        print(f"Euclidean Distance to Closest Match: {closest_distance}")
