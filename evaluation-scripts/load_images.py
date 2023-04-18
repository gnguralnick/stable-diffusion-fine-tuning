from PIL import Image
import os


def load_images_from_directory(directory):
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = []
    for file in image_files:
        image = Image.open(os.path.join(directory, file))
        images.append(image)
    return images
