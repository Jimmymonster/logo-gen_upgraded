from augmenter import Augmenter
from utils import load_images_from_directory, extract_random_images, extract_random_frames, insert_augmented_images, save_yolo_format, save_yolo_obbox_format, ranbow_frames,rgb_frames
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
oriented_bboxs = []
class_indices = []
for class_name in classes:
    path = os.path.join(logo_folder, class_name)
    images = load_images_from_directory(path)
    augmenter.add_dict(class_name, images)
for idx, class_name in enumerate(classes):
    num_class_images = num_images // len(classes)
    augmented_images_, oriented_bboxs_ = augmenter.augment(class_name, num_class_images, random=config.random_logo)
    augmented_images.extend(augmented_images_)
    oriented_bboxs.extend(oriented_bboxs_)
    class_indices.extend([idx] * num_class_images)

if video_path == 'rainbow':
    frames = ranbow_frames(num_frames)
elif video_path == 'white':
    frames = rgb_frames(num_frames,rgb=(255,255,255))
elif video_path == 'black':
    frames = rgb_frames(num_frames,rgb=(0,0,0))
elif os.path.isdir(video_path):
    frames = extract_random_images(video_path, num_frames)
else:
    frames = extract_random_frames(video_path, num_frames)

frames_with_augmentations, bounding_boxes, oriented_bounding_boxs = insert_augmented_images(frames, augmented_images, oriented_bboxs, class_indices,whiteout_bboxes=config.whiteout_bboxes,padding_crop=config.padding_crop)
if not config.obbox_format:
    save_yolo_format(frames_with_augmentations, bounding_boxes, output_folder, classes)
else:
    save_yolo_obbox_format(frames_with_augmentations, oriented_bounding_boxs, output_folder, classes)