import numpy as np
import os
import random

class Blackgrounder:
    # Note that every image is np array not PIL like augmenter
    def __init__(self, background_dict = None):
        # background dict contain list of folder path or video path
        self.background_dict = background_dict if background_dict is not None else {} 
        self.background_edit_settings_dict = {}
    def get_background(self, background_dict_list, num_bg_list, background_edit_setting_list):
        # iterate through background dict list to extract video to image and image according to numbg list, if num bg is higher than all possible extract image raise error
        # do apply setting to images
        # do shuffle every images and save in list
        # return list of background the total background eqaul sum of num bg list
        pass
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
        self.background_edit_settings_dict[setting_dict_name].append((getattr))

    def clear_settings(self, setting_name):
        pass
    def _apply_settings(self, images, setting_name):
        pass

    #============================== Settings function ===============================
    def video_interval(self, dict_name, seconds:float):
        if dict_name not in self.background_dict:
            raise ValueError(f"Dictionary '{dict_name}' not found in background_dict.")
        # if dict contain video path make it not select adjacent frame less than the seconds parameter
        return seconds
    def add_color_rectangle(self, image, position=(0.25,0.25,0.75,0.75), rgb = (255,255,255)):
        pass
    def hsv_shift(self, image, position=(0.25,0.25,0.75,0.75),brightness_shift=0,contrast_shift=0,hue_shift=0,saturation_shift=0,gamma_shift=0):
        pass
    def rgb_shift(self, image, position=(0.25,0.25,0.75,0.75),red_shift=0,green_shift=0,blue_shift=0):
        pass
    def add_object(self, image, object_image, num_object, position_range=(0.25,0.25,0.75,0.75)):
        pass