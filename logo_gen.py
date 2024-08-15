from augmenter import Augmenter
from utils import load_images_from_directory, extract_random_frames, insert_augmented_images, save_yolo_format
import os,shutil
import config

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
    augmented_images.extend(augmenter.augment(class_name, num_class_images, random=config.random_logo))
    class_indices.extend([idx] * num_class_images)

frames = extract_random_frames(video_path, num_frames)
frames_with_augmentations, bounding_boxes = insert_augmented_images(frames, augmented_images, class_indices,padding_crop=config.padding_crop)
save_yolo_format(frames_with_augmentations, bounding_boxes, output_folder, classes)

