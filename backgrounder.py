import numpy as np
import os
import cv2
import random

class Blackgrounder:
    # Note that every image is np array not PIL like augmenter
    def __init__(self, background_dict = None):
        # background dict contain list of folder path or video path
        self.background_dict = background_dict if background_dict is not None else {} 
        self.background_edit_settings_dict = {}
    def get_background(self, background_dict_list, num_bg_list, background_edit_setting_list):
        # iterate through background dict list to extract video to image and image according to numbg list, if num bg is higher than all possible extract image raise error
        all_backgrounds = []
        # check before do anything
        for i, dict_name in enumerate(background_dict_list):
            if dict_name not in self.background_dict:
                raise ValueError(f"Dictionary '{dict_name}' not found in background_dict.")
            if background_edit_setting_list[i] is not None and background_edit_setting_list[i] not in self.background_edit_settings_dict:
                raise ValueError(f"Dictionary '{dict_name}' not found in background_edit_settings_dict.")
            total_images=0
            for j,(type,path) in enumerate(self.background_dict[dict_name]):
                if(type == "rgb"):
                    self.background_dict[dict_name][j].append(-1) # -1 mean infinity
                    total_images+=num_bg_list[i]
                elif(type == "image"):
                    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
                    # Count the number of image files
                    images = len([file for file in os.listdir(path) if file.lower().endswith(image_extensions)])
                    self.background_dict[dict_name][j].append(images)
                    total_images += images
                elif(type == "video"):
                    video = cv2.VideoCapture(path)
                    images = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.background_dict[dict_name][j].append(images)
                    total_images += images
            if total_images<num_bg_list[i]:
                raise ValueError(f"Requested more backgrounds ({num_bg_list[i]}) than available in '{dict_name}' ({total_images})")
        # extract image split evenly from every source
        for i, dict_name in enumerate(background_dict_list):
            backgrounds=[]
            total_images_needed = num_bg_list[i]
            image_per_source = [0 for _ in range(len(self.background_dict[dict_name]))]
            total_image_per_source = []
            leftover_images = total_images_needed

            for j, (_, _, num_images) in enumerate(self.background_dict[dict_name]):
                if(num_images<0):
                    total_image_per_source.append(total_images_needed)
                else:
                    total_image_per_source.append(num_images)

            # Distribute remaining images to sources with available images
            while(leftover_images>0):
                min_positive = 1e9
                for num in total_image_per_source:
                    if 0 < num < min_positive:
                        min_positive = num
                for j in range(len(image_per_source)):
                    if total_image_per_source[j] > 0:
                        if total_image_per_source[j] >= min_positive:
                            # Allocate images from this source
                            image_per_source[j] += min_positive
                            total_image_per_source[j] -= min_positive
                            leftover_images -= min_positive
                            
                        else:
                            # Allocate remaining images from this source
                            image_per_source[j] += total_image_per_source[j]
                            leftover_images -= total_image_per_source[j]
                            total_image_per_source[j] = 0
                
            # Extract images from sources
            for j, (type, path, _) in enumerate(self.background_dict[dict_name]):
                if type == "rgb":
                    for _ in range(image_per_source[j]):
                        # Here you would add a blank image with the RGB color
                        backgrounds.append(path)
                elif type == "image":
                    images = [file for file in os.listdir(path) if file.lower().endswith(image_extensions)]
                    selected_images = random.sample(images, image_per_source[j])
                    for image_name in selected_images:
                        image = cv2.imread(os.path.join(path, image_name))
                        backgrounds.append(image)
                elif type == "video":
                    video = cv2.VideoCapture(path)
                    frame_indices = random.sample(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))), image_per_source[j])
                    for idx in frame_indices:
                        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = video.read()
                        if ret:
                            backgrounds.append(frame)
            # do apply setting to image
            if(background_edit_setting_list[i] is not None):
                all_backgrounds.extend(self._apply_settings(backgrounds,background_edit_setting_list[i]))
            else:
                all_backgrounds.extend(backgrounds)
        
        # do shuffle every images and save in list (optional)
        random.shuffle(backgrounds)
        # return list of background the total background eqaul sum of num bg list
        return all_backgrounds
    
    def add_rgb_bg_dict(self, name, rgb, width, height):
        rgb_image = np.full((width, height, 3), rgb, dtype=np.uint8)
        if name in self.background_dict:
            self.background_dict[name].append(["rgb",rgb_image])
        else:
            self.background_dict[name] = ["rgb",rgb_image]
    def add_dict(self, name, path, type):
        #there are two types first is image this type path will be directory, second is video this type path will be video file
        if name not in self.background_dict:
            self.background_dict[name] = []
        if type == 'image':
            if not os.path.isdir(path):
                raise ValueError(f"Path '{path}' is not a directory.")
            self.background_dict[name].append(["image",path])
        elif type == 'video':
            if not os.path.isfile(path) or not path.lower().endswith(('.mp4', '.avi', '.mov')):
                raise ValueError(f"Path '{path}' is not a valid video file.")
            self.background_dict[name].append(["video",path])
    def remove_dict(self, name):
        if name in self.background_dict:
            del self.background_dict[name]
        else:
            raise ValueError(f"Background name '{name}' not found in background_dict.")
    def add_settings(self, setting_dict_name, setting_method_name, *args, **kwargs):
        if not hasattr(self, setting_method_name):
            raise ValueError(f"Augmentation method '{setting_method_name}' does not exist.")
        if setting_dict_name not in self.background_edit_settings_dict:
            self.background_edit_settings_dict[setting_dict_name] = []
        self.background_edit_settings_dict[setting_dict_name].append((getattr(self, setting_method_name), args, kwargs))

    def clear_settings(self, setting_name):
        if setting_name in self.background_edit_settings_dict:
            del self.background_edit_settings_dict[setting_name]
        else:
            raise ValueError(f"Setting name '{setting_name}' not found in background_edit_settings_dict.")
    def _apply_settings(self, images, setting_name):
        for setting,args,kwargs in self.background_edit_settings_dict[setting_name]:
            for i in range(len(images)):
                images[i] = setting(images[i], *args, **kwargs)
        return images
            

    #============================== Settings function ===============================
    def add_color_rectangle(self, image, position=(0.25,0.25,0.75,0.75), rgb = (255,255,255)):
        height, width = image.shape[:2]
        x1, y1 = int(position[0] * width), int(position[1] * height)
        x2, y2 = int(position[2] * width), int(position[3] * height)

        # Add the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color=rgb, thickness=-1)
        return image
    def hsv_shift(self, image, position=(0.25,0.25,0.75,0.75),brightness_shift=0,contrast_shift=0,hue_shift=0,saturation_shift=0,gamma_shift=0):
        # Convert image to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Calculate the region of interest (ROI)
        height, width = image.shape[:2]
        x1, y1 = int(position[0] * width), int(position[1] * height)
        x2, y2 = int(position[2] * width), int(position[3] * height)
        # Extract ROI
        roi = hsv_image[y1:y2, x1:x2]
        # Apply shifts
        h, s, v = cv2.split(roi)
        # Shift Hue
        h = cv2.add(h, hue_shift)
        # Shift Saturation
        s = cv2.add(s, saturation_shift)
        # Shift Brightness (Value)
        v = cv2.add(v, brightness_shift)
        # Merge and apply ROI back to the image
        shifted_roi = cv2.merge([h, s, v])
        hsv_image[y1:y2, x1:x2] = shifted_roi
        # Convert back to BGR
        shifted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return shifted_image
    
    def rgb_shift(self, image, position=(0.25,0.25,0.75,0.75),red_shift=0,green_shift=0,blue_shift=0):
        # Calculate the region of interest (ROI)
        height, width = image.shape[:2]
        x1, y1 = int(position[0] * width), int(position[1] * height)
        x2, y2 = int(position[2] * width), int(position[3] * height)
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        # Apply shifts
        b, g, r = cv2.split(roi)
        # Shift Blue
        b = cv2.add(b, blue_shift)
        # Shift Green
        g = cv2.add(g, green_shift)
        # Shift Red
        r = cv2.add(r, red_shift)
        # Merge and apply ROI back to the image
        shifted_roi = cv2.merge([b, g, r])
        image[y1:y2, x1:x2] = shifted_roi
        return image
    def add_object(self, image, object_image, num_object, position_range=(0.25,0.25,0.75,0.75)):
        return image