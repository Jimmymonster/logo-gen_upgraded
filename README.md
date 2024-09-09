# About this project
This project is about doing augmentation and save it to yolo and yolo-obb format for train yolo model. The concept is augment the target object and place it into background. Augmenter class handle the augmentation of target object and Backgrounder class handle the background generation.
# How to use
### Augmenter
The Augmenter class contain target object dict which you can add dict during init class or just add it later.<br> Here the command to init class and handle the dict.
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

### Got a work to do. I will finish this docs later!!!
