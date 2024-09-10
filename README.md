# About this project
This project is about doing augmentation and save it to yolo and yolo-obb format for train yolo model. The concept is augment the target object and place it into background. Augmenter class handle the augmentation of target object and Backgrounder class handle the background generation.
# How to use

### Augmenter
The Augmenter class contain target object dict which you can add dict during init class or just add it later.<br> Here's the command to init class and handle the dict.
```
# init augmenter
augmenter = Augmenter()

# use load pil images function from utils
images = load_images_from_directory("path_to_image_directory")

# after get list of pil images add it to augmenter dict
augmenter.add_dict("target_object_name",images)
```
After add dict of all target objects now you can add augmentation function.

```
# add augmentation
augmenter.add_augmentation('set_resolution', max_resolution=(200,200))
augmenter.add_augmentation('rotation', angle_range=(-30,30), image_range=(0,10)) # <-- specific image range from 0-10 (10 included)
```

As you can see augmentation function can target specific image range that you want to augment, if not specified image range the augmentation will apply to all images.<br>
Here's the list off all augmentation function with parameters
| Function  | Description |
| ------------- | ------------- |
| flip_horizontal()  | flip target object horizontal  |
| flip_vertical()  | flip target object vertical |
| perspective(max_warp=0.2) | randomly apply perspective to image with max warp default is 0.2 |
| set_perspective(angle=10, direction=(0,0)) | apply perspective into specific 8 directions with angle (example if want southeast input as (1,-1)) |
| cylindrical(focal_len_x=100, focal_len_y=100, rotation_angle=0, perspective_angle=0, lighting_angle=0, lighting_strokeWidth=0, lighting_luminance=0, lighting_opacity=0)| Make image wrap around the cylindrical shape like can or bottle. Note that cylindrical shape shoundn't use with any 2d transformation. |
| resize(scale_range=(0.9, 1.5)) | Random resize in scale range |
| set_resolution(max_resolution=(80, 80)) | Resize image with largest width or height that not exceed max resolution |
| set_area(max_area=40000) | Resize image with largest area that not exceed max area |
| noise(min_noise_level: float = 25.0, max_noise_level: float = 50.0) | Add random noise to the image. |
| occlusions( occlusion_images: list, num_occlusions: int = 3) | Add object to image in random position. Input is pil image. |
| blur(scale_factor = 1.5) | Add blur to image by scale down and scale up image which divided by scale factor |
| stretch(scale_range= (0.5,1.5)) | Random stretch on x axis in scale range. |
| hue_color(brightness_range = (1.0,1.0), contrast_range = (1.0,1.0), hue_range = (0.0,0.0), saturation_range=(1.0,1.0), gamma_range=(1.0,1.0)) | Change hue color of image by percent in range. |
| rgb_color(target_color_range=((0, 255), (0, 255), (0, 255)), random_color_range=((0, 255), (0, 255), (0, 255))) | Change color of target image to random color image range in rgb color system. |
| adjust_opacity(opacity: float) | Adjust opacity of image. |
| adjust_background_opacity(rgb_color: tuple, background_opacity: float) | Replace transparent background with rgb color and opacity. |
<br>
After you add augmentation to target image now you can get augment image by using this command.
<br>

```
# get augmentation
# augmenter.augment('Dict Name', 'Num Images', 'Random') <-- if random is true the image will be spread evenly in the entire list if not the image will be in order of dict list

augmented_images, oriented_bboxs = augmenter.augment("target_object_name", 50, random=True)
```

### Backgrounder
The backgrounder class contain dict of image sources and dict of settings to apply in background. <br>
The images source has 3 types images, video and rgb. <br>
Here's the command to init backgrouder and add dict of image sources and settings.

```
backgrouder = Backgrounder()

# backgrounder.add_dict('dict_name','path','type')

backgrounder.add_dict("source1","path_to_images_directory","image")
backgrounder.add_dict("source1","path_to_video.mp4","video" , max_frames=5)
backgrounder.add_rgb_bg_dict("source1",width=400,height=400,rgb=(255,255,255), max_frames=10)
backgrounder.add_settings("setting1","rgb_shift", position(0,0,0.5,0.5) , red_shift = 255)
backgrounder.add_settings("setting1","rgb_shift", position(0,0,0.5,0.5) , blue_shift = 255, image_range=(0,5))
```
Here's the list off all settings function with parameters
| Function  | Description |
| ------------- | ------------- |
| add_color_rectangle(position=(0.25,0.25,0.75,0.75), rgb = (255,255,255)) | add colored rectangle in position x1,y1,x2,y2 in percent |
| hsv_shift(position=(0.25,0.25,0.75,0.75), brightness_shift=0, contrast_shift=0, hue_shift=0, saturation_shift=0, gamma_shift=0) | shift hsv color system in position x1,y1,x2,y2 in percent |
| rgb_shift(position=(0.25,0.25,0.75,0.75), red_shift=0, green_shift=0, blue_shift=0) | shift rgb color system in position x1,y1,x2,y2 in percent |
| remove_object_with_yolo_model(model, confident_level=0.5, target_class_name:list = None, rgb=(255,255,255), opacity = 1.0) | remove object in background using yolo model. Note that you need to init model before and pass model into this function. |
| add_object(object_image, num_object, position_range=(0.25, 0.25, 0.75, 0.75), scale_range = (1,1)) | add object image in background using for add some negative object that similar to target object. input is pil image. |
<br>

After add image sources and settings now you can get background using this command.

```
# backgrounder.get_background(["source1","source2",....],["num_images1","num_images2",...],["setting1","setting2",....])
frames = backgrounder.get_background(["source1"],[50],["setting1"])
```
### Utils Function
Utils function is a function that make your life easier. <br>
Here's all utils function.

| Function  | Description |
| ------------- | ------------- |
| load_images_from_directory(path) | load image from directory path and return as list of pil images |
| insert_augmented_images(frames, augmented_images, oriented_bboxs, class_indices, padding_crop=False) | insert augmented object into background frame and return frames, bbox, obbox |
| save_yolo_format(frames_with_augmentations, bounding_boxes, output_folder, classes) | save project in yolo format (class_index, x_mid, y_mid, width, height) |
| save_yolo_obbox_format(frames_with_augmentations, oriented_bounding_boxs, output_folder, classes) | save project in yolo-obb format (class_index, x1, y1, x2, y2, x3, y3, x4, y4) |

### Others
| Python Code  | Description |
| ------------- | ------------- |
| createMapping.py | create json file for upload into label studio from yolo project and label studio project for map the changed names that upload into label studio. Using api key and ip from .env |
| createMapping_obbox.py | create json for upload into label studio from yolo-obb project and label studio project for map the changed names that upload into label studio. Using api key and ip from .env |

### Example Usage

```
# init augmenter and parameter
augmenter = Augmenter()
logo_folder = 'logos'
output_folder = 'output'
random_logo = True
padding_crop = False
obbox_format = False
areas = [10000,40000]
perspec_directions = [(0,0),(0,1),(0,-1),(1,0),(-1,0)]
perspec_angles = [25]
rotation_angles = [-30,0,30]
stretchs = [0.7,1,1.3]
copy = 1
classes = ['object1','object2']
num_images = len(perspec_directions)*len(areas)*len(perspec_angles)*len(rotation_angles)*len(stretchs)*len(classes)*copy
num_frames = num_images
i=0

# apply augmentation combination
augmenter.add_augmentation('noise',min_noise_level=0,max_noise_level=75)
augmenter.add_augmentation('hue_color',brightness_range = (0.8,1.2))
augmenter.add_augmentation('blur',scale_factor=3)
for _ in range(len(classes)*copy):
    for area in areas:
        for rotation_angle in rotation_angles:
            for perspec_angle in perspec_angles:
                for stretch in stretchs:
                    for (x,y) in perspec_directions:
                        augmenter.add_augmentation('set_area',max_area=area,image_range=(i,i))
                        if stretch != 1:
                            augmenter.add_augmentation('stretch',scale_range=(stretch,stretch) ,image_range=(i,i))
                        augmenter.add_augmentation('rotation',angle_range=(rotation_angle,rotation_angle),image_range=(i,i))
                        augmenter.add_augmentation('set_perspective',angle=perspec_angle,direction=(x,y),image_range=(i,i))
                        augmenter.add_augmentation('adjust_background_opacity',rgb_color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)) , background_opacity=0.3,image_range=(i,i))
                        i+=1

#get augmentation images
augmented_images = []
oriented_bboxs = []
class_indices = []
#loop to add dict of each class into augmenter
for class_name in classes:
    path = os.path.join(logo_folder, class_name)
    images = load_images_from_directory(path)
    augmenter.add_dict(class_name, images)
#loop to get augment images and append into augmented_images list
for idx, class_name in enumerate(classes):
    num_class_images = num_images // len(classes)
    augmented_images_, oriented_bboxs_ = augmenter.augment(class_name, num_class_images, random=config.random_logo)
    augmented_images.extend(augmented_images_)
    oriented_bboxs.extend(oriented_bboxs_)
    class_indices.extend([idx] * num_class_images)

# init backgrounder
backgrounder = Blackgrounder()
model = YOLO("model/remove_object.pt")
# add source dict background
backgrounder.add_dict("dict1","video/video_TNN.mp4","video")
backgrounder.add_dict("dict1","video/ch3","image")
backgrounder.add_rgb_bg_dict("dict1",width=704,height=576,rgb=(255,255,255))
# add background setting
backgrounder.add_settings("add","remove_object_with_yolo_model",model, confident_level=0.4, target_class_name = None, rgb=(255,255,255), opacity = 1)

# get background
frames = backgrounder.get_background(["dict1"],[num_frames],["add"])

# place augmented images into background
frames_with_augmentations, bounding_boxes, oriented_bounding_boxs = insert_augmented_images(frames, augmented_images, oriented_bboxs, class_indices,padding_crop=config.padding_crop)

# save as yolo project
if not config.obbox_format:
    save_yolo_format(frames_with_augmentations, bounding_boxes, output_folder, classes)
else:
    save_yolo_obbox_format(frames_with_augmentations, oriented_bounding_boxs, output_folder, classes)
```


