from backgrounder import Blackgrounder
from utils import load_images_from_directory, insert_augmented_images, save_yolo_format, save_yolo_obbox_format
import os,shutil
import config
from ultralytics import YOLO
import time

start_time = time.time()

logo_folder = config.logo_folder
num_images = config.num_images
num_frames = config.num_frames
output_folder2 = config.output_folder
classes = config.classes

#clear output path
if os.path.exists(output_folder2):
    shutil.rmtree(output_folder2)
if not os.path.exists(output_folder2):
    os.makedirs(output_folder2)

augmenter = config.augmenter

dict_name = "dict1"
video_list = ["video_TNN.mp4","video_TNN2.mp4","video_ch3.mp4"]
image_list = ["ch3","ch5","ch7","ch8","pptv","workpoint"]
backgrounder = Blackgrounder()

for video in video_list:
    backgrounder.add_dict(dict_name,"video/" + video , "video")
for image in image_list:
    backgrounder.add_dict(dict_name,"video/" + image , "image")

for class_name in classes:
    output_folder = os.path.join(output_folder2,class_name)
    os.makedirs(output_folder)
    augmented_images = []
    oriented_bboxs = []
    path = os.path.join(logo_folder, class_name)
    images = load_images_from_directory(path)
    augmenter.add_dict(class_name, images)
    num_class_images = num_images // len(classes)
    num_frames_images = num_class_images
    augmented_images_, oriented_bboxs_ = augmenter.augment(class_name, num_class_images, random=config.random_logo)
    augmented_images.extend(augmented_images_)
    oriented_bboxs.extend(oriented_bboxs_)

    frames = backgrounder.get_background([dict_name],[num_frames_images],[None])

    frames_with_augmentations, bounding_boxes, oriented_bounding_boxs = insert_augmented_images(frames, augmented_images, oriented_bboxs, [0],padding_crop=config.padding_crop)
    if not config.obbox_format:
        save_yolo_format(frames_with_augmentations, bounding_boxes, output_folder, [class_name])
    else:
        save_yolo_obbox_format(frames_with_augmentations, oriented_bounding_boxs, output_folder, [class_name])

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")