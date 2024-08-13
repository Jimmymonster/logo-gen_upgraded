from augmenter import Augmenter
import random

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
set_perspective(self, image, angle=10, direction=(0,0)):  direction in (x,y) example if want southeast input as (1,-1)
rotation(self, image, angle_range=(-60, 60))
stretch(self, image_pil: Image.Image, scale_range= (0.5,1.5), min_strech = 0.0)
color(self, img: Image.Image,brightness_range = (1.0,1.0), contrast_range = (1.0,1.0), hue_range = (0.0,0.0), saturation_range=(1.0,1.0), gamma_range=(1.0,1.0))
"""
augmenter = Augmenter()

#============================== Config test ==============================================

# video_path = 'video/video_TNN.mp4'
# logo_folder = 'logos'
# output_folder = 'output'
# num_images = 7
# num_frames = num_images
# classes = ['TNN']

# # augmenter.add_augmentation('set_perspective',angle=30,direction=(-1,-1),image_range=(0,0))
# # augmenter.add_augmentation('set_perspective',angle=30,direction=(-1,0),image_range=(1,1))
# # augmenter.add_augmentation('set_perspective',angle=30,direction=(-1,1),image_range=(2,2))
# # augmenter.add_augmentation('set_perspective',angle=30,direction=(0,-1),image_range=(3,3))
# # augmenter.add_augmentation('set_perspective',angle=30,direction=(0,0),image_range=(4,4))
# # augmenter.add_augmentation('set_perspective',angle=30,direction=(0,1),image_range=(5,5))
# # augmenter.add_augmentation('set_perspective',angle=30,direction=(1,-1),image_range=(6,6))
# # augmenter.add_augmentation('set_perspective',angle=30,direction=(1,0),image_range=(7,7))
# # augmenter.add_augmentation('set_perspective',angle=30,direction=(1,1),image_range=(8,8))
# augmenter.add_augmentation('cylindrical',radius=64,image_range=(0,0))
# augmenter.add_augmentation('cylindrical',radius=128,image_range=(1,1))
# augmenter.add_augmentation('cylindrical',radius=256,image_range=(2,2))
# augmenter.add_augmentation('cylindrical',radius=512,image_range=(3,3))
# augmenter.add_augmentation('cylindrical',radius=1024,image_range=(4,4))
# augmenter.add_augmentation('cylindrical',radius=2048,image_range=(5,5))
# augmenter.add_augmentation('cylindrical',radius=4096,image_range=(6,6))

#============================== Config 3 no random perspective =============================================
video_path = 'video/video_TNN.mp4'
logo_folder = 'logos'
output_folder = 'output'
num_images = 50
num_frames = num_images
classes = ['7up2']
augmenter.add_augmentation('set_resolution',max_resolution=(200,200),image_range=(0,9))
augmenter.add_augmentation('set_resolution',max_resolution=(125,125),image_range=(10,24))
augmenter.add_augmentation('set_resolution',max_resolution=(75,75),image_range=(25,49))

augmenter.add_augmentation('set_perspective',angle=30,direction=(0,-1),image_range=(0,0))
augmenter.add_augmentation('set_perspective',angle=30,direction=(0,1),image_range=(1,1))
augmenter.add_augmentation('set_perspective',angle=30,direction=(-1,0),image_range=(2,2))
augmenter.add_augmentation('set_perspective',angle=30,direction=(1,0),image_range=(3,3))
augmenter.add_augmentation('set_perspective',angle=50,direction=(0,-1),image_range=(4,4))
augmenter.add_augmentation('set_perspective',angle=50,direction=(0,1),image_range=(5,5))
augmenter.add_augmentation('set_perspective',angle=50,direction=(-1,0),image_range=(6,6))
augmenter.add_augmentation('set_perspective',angle=50,direction=(1,0),image_range=(7,7))

augmenter.add_augmentation('set_perspective',angle=30,direction=(0,-1),image_range=(8,9))
augmenter.add_augmentation('set_perspective',angle=30,direction=(0,1),image_range=(10,11))
augmenter.add_augmentation('set_perspective',angle=30,direction=(-1,0),image_range=(12,13))
augmenter.add_augmentation('set_perspective',angle=30,direction=(1,0),image_range=(14,15))
augmenter.add_augmentation('set_perspective',angle=50,direction=(0,-1),image_range=(16,17))
augmenter.add_augmentation('set_perspective',angle=50,direction=(0,1),image_range=(18,19))
augmenter.add_augmentation('set_perspective',angle=50,direction=(-1,0),image_range=(20,21))
augmenter.add_augmentation('set_perspective',angle=50,direction=(1,0),image_range=(22,23))

augmenter.add_augmentation('set_perspective',angle=30,direction=(0,-1),image_range=(25,27))
augmenter.add_augmentation('set_perspective',angle=30,direction=(0,1),image_range=(28,30))
augmenter.add_augmentation('set_perspective',angle=30,direction=(-1,0),image_range=(31,33))
augmenter.add_augmentation('set_perspective',angle=30,direction=(1,0),image_range=(34,36))
augmenter.add_augmentation('set_perspective',angle=50,direction=(0,-1),image_range=(37,39))
augmenter.add_augmentation('set_perspective',angle=50,direction=(0,1),image_range=(40,42))
augmenter.add_augmentation('set_perspective',angle=50,direction=(-1,0),image_range=(43,45))
augmenter.add_augmentation('set_perspective',angle=50,direction=(1,0),image_range=(46,48))

augmenter.add_augmentation('noise',min_noise_level=25,max_noise_level=50)
# augmenter.add_augmentation('stretch',scale_range= (0.5,1.5), min_strech = 0.1)
augmenter.add_augmentation('color',brightness_range = (0.95,1.05), contrast_range = (0.95,1.05))
augmenter.add_augmentation('blur',scale_factor=random.uniform(1.5,2.5))

#============================== Config 1 no huge image==============================================

# video_path = 'video/video_TNN.mp4'
# logo_folder = 'logos'
# output_folder = 'output'
# num_images = 50
# num_frames = num_images
# classes = ['sevenEleven1']
# augmenter.add_augmentation('set_resolution',max_resolution=(200,200),image_range=(0,9))
# augmenter.add_augmentation('set_resolution',max_resolution=(125,125),image_range=(10,24))
# augmenter.add_augmentation('set_resolution',max_resolution=(75,75),image_range=(25,49))
# augmenter.add_augmentation('perspective',max_warp=0.2)
# augmenter.add_augmentation('noise',min_noise_level=25,max_noise_level=50)
# augmenter.add_augmentation('stretch',scale_range= (0.5,1.5), min_strech = 0.1)
# augmenter.add_augmentation('color',brightness_range = (0.95,1.05), contrast_range = (0.95,1.05))
# augmenter.add_augmentation('blur',scale_factor=random.uniform(1.5,2.5))

#============================== Config 2 huge image + 3 ==============================================

# video_path = 'video/video_TNN.mp4'
# logo_folder = 'logos'
# output_folder = 'output'
# num_images = 53
# num_frames = num_images
# classes = ['sevenEleven1']
# augmenter = Augmenter()
# augmenter.add_augmentation('set_resolution',max_resolution=(570,570),image_range=(0,2))
# augmenter.add_augmentation('set_resolution',max_resolution=(200,200),image_range=(3,12))
# augmenter.add_augmentation('set_resolution',max_resolution=(125,125),image_range=(13,27))
# augmenter.add_augmentation('set_resolution',max_resolution=(75,75),image_range=(28,52))
# augmenter.add_augmentation('perspective',max_warp=0.2,image_range=(3,53))
# augmenter.add_augmentation('noise',min_noise_level=25,max_noise_level=50,image_range=(3,52))
# augmenter.add_augmentation('stretch',scale_range= (0.5,1.5), min_strech = 0.1,image_range=(3,52))
# augmenter.add_augmentation('color',brightness_range = (0.95,1.05), contrast_range = (0.95,1.05),image_range=(3,52))
# augmenter.add_augmentation('blur',scale_factor=1.5)
#======================================================================================