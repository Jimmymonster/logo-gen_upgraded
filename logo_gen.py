from augmenter import Augmenter
from utils import load_images_from_directory, extract_random_frames, insert_augmented_images, save_yolo_format
import os,shutil
import config

"""
Augmenter allow augment method

flip_horizontal(self, img)
flip_vertical(self, img)
resize(self, image, scale_range=(0.9, 1.5))
set_resolution(self, image, max_resolution=(80, 80))
noise(self, image_pil: Image.Image, min_noise_level: float = 25.0, max_noise_level: float = 50.0)
occlusions(self, image_pil: Image.Image, occlusion_images: list, num_occlusions: int = 3)
blur(self, pil_image, scale_factor)
perspective(self, image, max_warp=0.2)
rotation(self, image, angle_range=(-60, 60))
stretch(self, image_pil: Image.Image, scale_range= (0.5,1.5), min_strech = 0.0)
color(self, img: Image.Image,brightness_range = (1.0,1.0), contrast_range = (1.0,1.0), hue_range = (0.0,0.0), saturation_range=(1.0,1.0), gamma_range=(1.0,1.0))
"""

video_path = config.video_path
logo_folder = config.logo_folder
num_images = config.num_images
num_frames = config.num_frames
output_folder = config.output_folder
classes = config.classes

#clear output path
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

augmenter = config.augmenter
augmented_images = []
class_indices = []
for class_name in classes:
    path = os.path.join(logo_folder, class_name)
    images = load_images_from_directory(path)
    augmenter.add_dict(class_name, images)
for idx, class_name in enumerate(classes):
    num_class_images = num_images // len(classes)
    augmented_images.extend(augmenter.augment(class_name, num_class_images))
    class_indices.extend([idx] * num_class_images)

frames = extract_random_frames(video_path, num_frames)
frames_with_augmentations, bounding_boxes = insert_augmented_images(frames, augmented_images, class_indices)
save_yolo_format(frames_with_augmentations, bounding_boxes, output_folder, classes)

