import os
from PIL import Image
from utils import whiteout_areas

output_path = "C:/Users/thanapob/Downloads/khontonsurmong"
image_path = os.path.join(output_path, "images")
whiteout_bboxes = [(620, 90, 668, 175)]

def load_images_from_directory(directory_path):
    images = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(file_path) as img:
                images[filename] = img.copy()  # Use img.copy() to ensure PIL Image is not closed
    return images
# Load images from the directory
images = load_images_from_directory(image_path)

# Apply whiteout and save each image back to the same directory
for image_name, image in images.items():
    # Apply whiteout to the image
    whiteout_image = whiteout_areas(image, whiteout_bboxes)
    
    # Save the modified image back to the same directory
    whiteout_image.save(os.path.join(image_path, image_name))

print("All images have been processed and saved.")