from augmenter import Augmenter
from backgrounder import Blackgrounder
from PIL import Image
from utils import load_images_from_directory, extract_random_images, extract_random_frames, insert_augmented_images, save_yolo_format, save_yolo_obbox_format, ranbow_frames,rgb_frames
import os,shutil
import config
from ultralytics import YOLO
import time

start_time = time.time()

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

# if video_path == 'rainbow':
#     frames = ranbow_frames(num_frames)
# elif video_path == 'white':
#     frames = rgb_frames(num_frames,rgb=(255,255,255))
# elif video_path == 'black':
#     frames = rgb_frames(num_frames,rgb=(0,0,0))
# elif os.path.isdir(video_path):
#     frames = extract_random_images(video_path, num_frames)
# else:
#     frames = extract_random_frames(video_path, num_frames)

# need to load model first if want to use it
model = YOLO("model/model.pt")

backgrounder = Blackgrounder()
backgrounder.add_dict("dict1","video/video_TNN.mp4","video")
backgrounder.add_dict("dict1","video/video_TNN2.mp4","video")
backgrounder.add_dict("dict1","video/video_ch3.mp4","video")
# backgrounder.add_dict("dict1","video/video_TNN3.mp4","video")
backgrounder.add_dict("dict1","video/ch3","image")
backgrounder.add_dict("dict1","video/ch5","image")
backgrounder.add_dict("dict1","video/ch7","image")
backgrounder.add_dict("dict1","video/ch8","image")
backgrounder.add_dict("dict1","video/gmm25","image")
backgrounder.add_dict("dict1","video/pptv","image")
backgrounder.add_dict("dict1","video/workpoint","image")
# backgrounder.add_dict("dict1","video/video_mono29.mp4","video")
# backgrounder.add_rgb_bg_dict("dict1",width=704,height=576,rgb=(255,255,255))
# backgrounder.add_settings("shift","rgb_shift",position=(0 ,0 , 0.5 ,0.5),red_shift=-500)
# backgrounder.add_settings("shift","rgb_shift",position=(0.5 ,0 ,1 ,0.5),green_shift=-500)
# backgrounder.add_settings("shift","rgb_shift",position=(0 ,0.5 ,0.5 ,1),blue_shift=-500)
object_img = [Image.open("temp/object/mrdiy1.png"),Image.open("temp/object/mrdiy2.png"),Image.open("temp/object/mrdiy3.png")]
backgrounder.add_settings("add","remove_object_with_yolo_model",model, confident_level=0.4, target_class_name = None, rgb=(255,255,255), opacity = 1)
backgrounder.add_settings("add","add_object",object_image=object_img[0] ,num_object=1, position_range=(0, 0, 1, 1), scale_range=(0.5,1.5), image_range=(0,9))
backgrounder.add_settings("add","add_object",object_image=object_img[1] ,num_object=1, position_range=(0, 0, 1, 1), scale_range=(0.5,1.5), image_range=(10,19))
backgrounder.add_settings("add","add_object",object_image=object_img[2] ,num_object=1, position_range=(0, 0, 1, 1), scale_range=(0.5,1.5), image_range=(20,29))
frames = backgrounder.get_background(["dict1"],[num_frames],["add"])

frames_with_augmentations, bounding_boxes, oriented_bounding_boxs = insert_augmented_images(frames, augmented_images, oriented_bboxs, class_indices,whiteout_bboxes=config.whiteout_bboxes,padding_crop=config.padding_crop)
if not config.obbox_format:
    save_yolo_format(frames_with_augmentations, bounding_boxes, output_folder, classes)
else:
    save_yolo_obbox_format(frames_with_augmentations, oriented_bounding_boxs, output_folder, classes)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")