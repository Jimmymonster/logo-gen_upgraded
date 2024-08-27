class Blackgrounder:
    def __init__(self, background_dict = None):
        self.background_dict = background_dict if background_dict is not None else {} 
        self.background_edit_settings_dict = None
    def get_background(self, background_dict_list, num_bg_list, background_edit_setting_list):
        # do extract images
        # do apply setting to images
        # return list of background
        pass
    def add_rgb_bg_dict(self, name, rgb):
        pass
    def add_dict(self, name, path):
        pass
    def remove_dict(self, name):
        pass
    def update_dict(self, name, path):
        pass
    def add_settings(self, setting_name, *args, **kwargs):
        pass
    def clear_settings(self, setting_name):
        pass
    def _apply_settings(self, images, setting_name):
        pass

    #============================== Settings function ===============================
    def add_color_rectangle(self, image, position=(0.25,0.25,0.75,0.75), rgb = (255,255,255)):
        pass
    def hsv_shift(self, image, position=(0.25,0.25,0.75,0.75),brightness_shift=0,contrast_shift=0,hue_shift=0,saturation_shift=0,gamma_shift=0):
        pass
    def rgb_shift(self, image, position=(0.25,0.25,0.75,0.75),red_shift=0,green_shift=0,blue_shift=0):
        pass
    def add_object(self, image, object_image, num_object, position_range=(0.25,0.25,0.75,0.75)):
        pass