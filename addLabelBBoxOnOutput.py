import os
from PIL import Image
from utils import load_images_from_directory

output_path = "C:/Users/thanapob/Downloads/khontonsurmong"
image_path = os.path.join(output_path, "images")
labels_path = os.path.join(output_path, "labels")
bounding_box = (617, 87, 670, 177)
box_id = 1  # Example ID for the bounding box

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

# Function to append bounding box to label file in YOLO format
def append_bounding_box_to_label(label_file_path, box, box_id, image_width, image_height):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    box_str = f"{box_id} {x_center} {y_center} {width} {height}\n"
    with open(label_file_path, 'a') as label_file:
        label_file.write(box_str)

# Process each image
for image_name, image in images.items():
    # Derive corresponding label file name
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_file_path = os.path.join(labels_path, label_name)
    
    # Get image dimensions
    image_width, image_height = image.size
    
    # Append bounding box to the label file
    append_bounding_box_to_label(label_file_path, bounding_box, box_id, image_width, image_height)

print("All label files have been processed and updated.")
