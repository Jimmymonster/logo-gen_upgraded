from augmenter import Augmenter
import random

"""
Augmenter allow augment method

=== perspective operation ===
flip_horizontal()
flip_vertical()
perspective(max_warp=0.2)
set_perspective(angle=10, direction=(0,0)):  direction in (x,y) example if want southeast input as (1,-1)
cylindrical(focal_len_x=100, focal_len_y=100, rotation_angle=0, perspective_angle=0, lighting_angle=0, lighting_strokeWidth=0, lighting_luminance=0, lighting_opacity=0) <--! Note that cylindrical shape shoundn't use with perspective function, perspective function only work for 2d image.
rotation(angle_range=(-60, 60))

=== sizing operation ===
resize(scale_range=(0.9, 1.5))
set_resolution(max_resolution=(80, 80))
set_area(max_area=40000)

== distortion operation ===
noise(min_noise_level: float = 25.0, max_noise_level: float = 50.0)
occlusions( occlusion_images: list, num_occlusions: int = 3)
blur(scale_factor)
stretch(scale_range= (0.5,1.5))
hue_color(brightness_range = (1.0,1.0), contrast_range = (1.0,1.0), hue_range = (0.0,0.0), saturation_range=(1.0,1.0), gamma_range=(1.0,1.0))
rgb_color(target_color_range=((0, 255), (0, 255), (0, 255)), random_color_range=((0, 255), (0, 255), (0, 255)))
adjust_opacity(opacity: float)
adjust_background_opacity(rgb_color: tuple, background_opacity: float) 
"""
augmenter = Augmenter()

#============================== Config gen 3 fix augment all ==============================================

video_path = 'video/ch3'
# video_path = 'white' # white, black, rainbow
logo_folder = 'logos'
output_folder = 'output'
random_logo = True
padding_crop = False
obbox_format = False
# whiteout_bboxes = [(620,90,668,175)] TNN
whiteout_bboxes = []

# rectangle areas
# areas = [3500,7500,15000,35000,60000]
# areas = [3500,15000,35000]

# square areas
areas = [25000,50000]

perspec_directions = [(0,0),(0,1),(0,-1),(1,0),(-1,0)]

perspec_angles = [25]
rotation_angles = [-30,0,30]
stretchs = [0.8,1.2]
# perspec_angles = [15,30]
# rotation_angles = [-45,-30,0,30,45]
copy = 1

classes = ['aia2']
num_images = len(perspec_directions)*len(areas)*len(perspec_angles)*len(rotation_angles)*len(classes)*copy
num_frames = num_images
i=0
# for _ in range(len(classes)):
for _ in range(len(classes)*copy):
    for area in areas:
        for rotation_angle in rotation_angles:
            for perspec_angle in perspec_angles:
                for stretch in stretchs:
                    for (x,y) in perspec_directions:
                        augmenter.add_augmentation('set_area',max_area=area,image_range=(i,i))
                        augmenter.add_augmentation('stretch',scale_range=(stretch,stretch) ,image_range=(i,i))
                        augmenter.add_augmentation('rotation',angle_range=(rotation_angle,rotation_angle),image_range=(i,i))
                        augmenter.add_augmentation('set_perspective',angle=perspec_angle,direction=(x,y),image_range=(i,i))
                        # augmenter.add_augmentation('adjust_background_opacity',rgb_color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)) , background_opacity=0.5,image_range=(i,i))
                        i+=1
augmenter.add_augmentation('blur',scale_factor=1.5)
augmenter.add_augmentation('noise',min_noise_level=0,max_noise_level=75)
augmenter.add_augmentation('hue_color',brightness_range = (0.8,1.2))
# augmenter.add_augmentation('hue_color',brightness_range = (0.7,1.3), contrast_range = (0.7,1.3))


num_images+=5
num_frames=num_images
areas = [50+i*50 for i in range(0,5)]
for area in areas:
    augmenter.add_augmentation('set_resolution',max_resolution=(area,area),image_range=(i,i))
    augmenter.add_augmentation('adjust_background_opacity',rgb_color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)) , background_opacity=0.5,image_range=(i,i))
    i+=1
# ===================================== Cylinder Config ================================================
# video_path = 'video/ch3'
# # video_path = 'white' # white, black, rainbow
# logo_folder = 'logos'
# output_folder = 'output'
# random_logo = True
# padding_crop = False
# obbox_format = False
# # whiteout_bboxes = [(620,90,668,175)]
# whiteout_bboxes = []

# # areas = [3500,15000,35000]

# # square areas
# areas = [10000,25000,50000]
# perspec_angles = [-30,-15,0,15,30]
# rotation_angles = [-30,0,30]
# rotation_angles_normal = [0]
# # perspec_angles = [-30,-15,0,15,30]
# # rotation_angles = [-30,-15,0,15,30]
# # rotation_angles_normal = [0,15]
# copy = 1

# classes = ['sponsor']
# num_images = len(areas)*len(perspec_angles)*len(rotation_angles)*len(rotation_angles_normal)*len(classes)*copy
# num_frames = num_images
# # i=0
# for _ in range(len(classes)*copy):
#     for area in areas:
#         for rotation_angle in rotation_angles:
#             for perspec_angle in perspec_angles:
#                 for rotation_angle_normal in rotation_angles_normal:
#                 # for (x,y) in perspec_directions:
#                     augmenter.add_augmentation('set_area',max_area=area,image_range=(i,i))
#                     # augmenter.add_augmentation('cylindrical',focal_len_x=80, focal_len_y=80, rotation_angle = rotation_angle, perspective_angle=perspec_angle, image_range=(i,i), lighting_angle=45, lighting_strokeWidth=3, lighting_luminance=5, lighting_opacity=1)
#                     augmenter.add_augmentation('cylindrical',focal_len_x=80, focal_len_y=80, rotation_angle = rotation_angle, perspective_angle=perspec_angle, image_range=(i,i))
#                     augmenter.add_augmentation('rotation',angle_range=(rotation_angle_normal,rotation_angle_normal),image_range=(i,i))
#                     if(area == areas[0] or i%4==0):
#                         augmenter.add_augmentation('blur',scale_factor=1.5,image_range=(i,i))
#                     elif(area == areas[1]):
#                         augmenter.add_augmentation('blur',scale_factor=2.5,image_range=(i,i))
#                     elif(area == areas[2]):
#                         augmenter.add_augmentation('blur',scale_factor=3.5,image_range=(i,i))
#                     i+=1
# # augmenter.add_augmentation('adjust_background_opacity',rgb_color=(255,255,255) , background_opacity=0.5)

# ================================================ Config Test ===============================================
# video_path = 'video/ch3'
# # video_path = 'white' # white, black, rainbow
# logo_folder = 'logos'
# output_folder = 'output'
# random_logo = True
# padding_crop = False
# obbox_format = True
# # whiteout_bboxes = [(620,90,668,175)] TNN
# whiteout_bboxes = []

# # rectangle areas
# # areas = [3500,15000,35000]

# # square areas
# areas = [50000]
# perspec_directions = [(0,0),(0,1),(0,-1),(1,0),(-1,0)]
# flip_hor = [True,False]
# flip_ver = [True,False]
# perspec_angles = [50]
# rotation_angles = [-45,0,45]
# copy = 1

# classes = ['mrdiy1']
# num_images = len(perspec_directions)*len(areas)*len(perspec_angles)*len(rotation_angles)*len(classes)*copy
# num_frames = num_images
# i=0
# # for _ in range(len(classes)):
# for _ in range(len(classes)*copy):
#     for area in areas:
#         for rotation_angle in rotation_angles:
#             for perspec_angle in perspec_angles:
#                 for (x,y) in perspec_directions:
#                     for fliph in flip_hor:
#                         for flipv in flip_ver:
#                             augmenter.add_augmentation('set_area',max_area=area,image_range=(i,i))
#                             augmenter.add_augmentation('rotation',angle_range=(rotation_angle,rotation_angle),image_range=(i,i))
#                             augmenter.add_augmentation('set_perspective',angle=perspec_angle,direction=(x,y),image_range=(i,i))
#                             if fliph:
#                                 augmenter.add_augmentation('flip_horizontal')
#                             if flipv:
#                                 augmenter.add_augmentation('flip_vertical')
#                             if(area == areas[0] or i%4==0):
#                                 augmenter.add_augmentation('blur',scale_factor=1.5,image_range=(i,i))
#                             elif(area == areas[1]):
#                                 augmenter.add_augmentation('blur',scale_factor=2.5,image_range=(i,i))
#                             elif(area == areas[2]):
#                                 augmenter.add_augmentation('blur',scale_factor=3.5,image_range=(i,i))
#                             i+=1

# augmenter.add_augmentation('noise',min_noise_level=0,max_noise_level=100)
# augmenter.add_augmentation('hue_color',brightness_range = (0.7,1.3), contrast_range = (0.7,1.3))
# # augmenter.add_augmentation('adjust_background_opacity',rgb_color=(255,255,255) , background_opacity=0.25)