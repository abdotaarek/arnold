import torch
import clip
from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_clip_embeddings_batched(image_paths, batch_size=32):
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [preprocess(Image.open(path)).unsqueeze(0) for path in batch_paths if os.path.isfile(path)]
        batch_images = torch.cat(batch_images, dim=0).to(device)
        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_images)
        embeddings.extend(batch_embeddings)
    return torch.stack(embeddings)

def find_closest_match(test_embedding, train_embeddings):
    distances = torch.nn.functional.pairwise_distance(test_embedding, train_embeddings)
    closest_idx = torch.argmin(distances).item()
    closest_distance = distances[closest_idx].item()
    return closest_idx, closest_distance


# Paths for train and test episodes
train_episode_dirs = [
    '/media/local/atarek/npzFrames/opendrawer/train/1',
    '/media/local/atarek/npzFrames/opendrawer/train/2',
    '/media/local/atarek/npzFrames/opendrawer/train/3'
]
test_episode_dir = '/media/local/atarek/npzFrames/opendrawer/test/1'

# Collect image paths
train_episode_paths = [img for dir in train_episode_dirs for img in glob.glob(os.path.join(dir, "*.png"))]
test_episode_paths = glob.glob(os.path.join(test_episode_dir, "*.png"))

# Get embeddings
train_embeddings = get_clip_embeddings_batched(train_episode_paths)
test_embeddings = get_clip_embeddings_batched(test_episode_paths)

# Iterate over test embeddings
for i, test_embedding in enumerate(test_embeddings):
    closest_match_idx, closest_distance = find_closest_match(test_embedding.unsqueeze(0), train_embeddings)

    # Print the embeddings and closest match details
    print(f"Test Image {i}: {os.path.basename(test_episode_paths[i])}")
    print(f"Closest Train Image: {os.path.basename(train_episode_paths[closest_match_idx])}")
    print(f"Euclidean Distance: {closest_distance}\n")

    # Load images
    test_image = Image.open(test_episode_paths[i])
    train_image = Image.open(train_episode_paths[closest_match_idx])

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(test_image)
    axs[0].set_title("Test Image")
    axs[0].axis('off')

    axs[1].imshow(train_image)
    axs[1].set_title("Closest Match in Train")
    axs[1].axis('off')

    plt.show()
