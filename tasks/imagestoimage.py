import os
from PIL import Image

def create_composite_image(directory, output_filename):
    # Retrieve all image files in the specified directory
    image_files = [file for file in os.listdir(directory) if file.endswith(('.png', '.jpg', '.jpeg'))]
    # Sort images: 'initial' images first, then 'end' images
    initial_images = sorted([file for file in image_files if 'initial' in file])
    end_images = sorted([file for file in image_files if 'end' in file])
    sorted_images = initial_images + end_images

    # Load images
    images = [Image.open(os.path.join(directory, file)) for file in sorted_images]

    # Assuming all images have the same size
    image_width, image_height = images[0].size

    # Create a new blank composite image
    composite_image = Image.new('RGB', (image_width * 5, image_height * 2))

    # Paste images into the composite image
    for i, image in enumerate(images):
        x_offset = (i % 5) * image_width
        y_offset = (i // 5) * image_height
        composite_image.paste(image, (x_offset, y_offset))

    # Save the composite image
    composite_image.save(output_filename)

# Example usage
directory = '/media/local/atarek/demonstrations/episodes/1'  # Change to your directory path
output_filename = '/media/local/atarek/demonstrations/episodesimages/composite_image1.png'  # Change to your desired output file path but leave composite_image1.png
create_composite_image(directory, output_filename)
