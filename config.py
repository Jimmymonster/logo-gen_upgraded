from augmenter import Augmenter

"""
Augmenter allow augment method

flip_horizontal(self, img)
flip_vertical(self, img)
resize(self, image, scale_range=(0.9, 1.5))
set_resolution(self, image, max_resolution=(80, 80))
noise(self, image_pil: Image.Image, min_noise_level: float = 25.0, max_noise_level: float = 50.0)
occlusions(self, image_pil: Image.Image, occlusion_images: list, num_occlusions: int = 3)
blur(self, pil_image, scale_factor)
perspective(self, image, max_warp=0.2)
rotation(self, image, angle_range=(-60, 60))
stretch(self, image_pil: Image.Image, scale_range= (0.5,1.5), min_strech = 0.0)
color(self, img: Image.Image,brightness_range = (1.0,1.0), contrast_range = (1.0,1.0), hue_range = (0.0,0.0), saturation_range=(1.0,1.0), gamma_range=(1.0,1.0))
"""

#============================== Config 1 ==============================================
video_path = 'video/video_TNN.mp4'
logo_folder = 'logos'
output_folder = 'output'

num_images = 53
num_frames = num_images
# logos
classes = ['sevenEleven1']

augmenter = Augmenter()
augmenter.add_augmentation('set_resolution',max_resolution=(570,570),image_range=(0,2))
augmenter.add_augmentation('set_resolution',max_resolution=(200,200),image_range=(3,12))
augmenter.add_augmentation('set_resolution',max_resolution=(125,125),image_range=(13,27))
augmenter.add_augmentation('set_resolution',max_resolution=(75,75),image_range=(28,52))
augmenter.add_augmentation('perspective',max_warp=0.2,image_range=(3,53))
augmenter.add_augmentation('noise',min_noise_level=25,max_noise_level=50,image_range=(3,52))
augmenter.add_augmentation('stretch',scale_range= (0.5,1.5), min_strech = 0.1,image_range=(3,52))
augmenter.add_augmentation('color',brightness_range = (0.95,1.05), contrast_range = (0.95,1.05),image_range=(3,52))
augmenter.add_augmentation('blur',scale_factor=1.5)
#======================================================================================