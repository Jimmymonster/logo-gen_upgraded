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

# need to load model first if want to use it
# model = YOLO("model/model.pt")
dict_name = "dict1"
video_list = ["video_TNN.mp4","video_TNN2.mp4","video_ch3.mp4"]
image_list = ["ch3","ch5","ch7","ch8","pptv","workpoint"]
backgrounder = Blackgrounder()

for video in video_list:
    backgrounder.add_dict(dict_name,"video/" + video , "video")
for image in image_list:
    backgrounder.add_dict(dict_name,"video/" + image , "image")

# backgrounder.add_rgb_bg_dict("dict1",width=704,height=576,rgb=(255,255,255))
# backgrounder.add_settings("shift","rgb_shift",position=(0 ,0 , 0.5 ,0.5),red_shift=-500)
# backgrounder.add_settings("shift","rgb_shift",position=(0.5 ,0 ,1 ,0.5),green_shift=-500)
# backgrounder.add_settings("shift","rgb_shift",position=(0 ,0.5 ,0.5 ,1),blue_shift=-500)

# backgrounder.add_settings("add","remove_object_with_yolo_model",model, confident_level=0.4, target_class_name = None, rgb=(255,255,255), opacity = 1)

# object_img = [Image.open("temp/object/mrdiy1.png"),Image.open("temp/object/mrdiy2.png"),Image.open("temp/object/mrdiy3.png")]
# backgrounder.add_settings("add","add_object",object_image=object_img[0] ,num_object=1, position_range=(0, 0, 1, 1), scale_range=(0.5,1.5), image_range=(0,9))
# backgrounder.add_settings("add","add_object",object_image=object_img[1] ,num_object=1, position_range=(0, 0, 1, 1), scale_range=(0.5,1.5), image_range=(10,19))
# backgrounder.add_settings("add","add_object",object_image=object_img[2] ,num_object=1, position_range=(0, 0, 1, 1), scale_range=(0.5,1.5), image_range=(20,29))

frames = backgrounder.get_background([dict_name],[num_frames],[None])

frames_with_augmentations, bounding_boxes, oriented_bounding_boxs = insert_augmented_images(frames, augmented_images, oriented_bboxs, class_indices,padding_crop=config.padding_crop)
if not config.obbox_format:
    save_yolo_format(frames_with_augmentations, bounding_boxes, output_folder, classes)
else:
    save_yolo_obbox_format(frames_with_augmentations, oriented_bounding_boxs, output_folder, classes)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")