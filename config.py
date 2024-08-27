from augmenter import Augmenter
import random

"""
Augmenter allow augment method

=== perspective operation ===
flip_horizontal()
flip_vertical()
perspective(max_warp=0.2)
set_perspective(angle=10, direction=(0,0)):  direction in (x,y) example if want southeast input as (1,-1)
cylindrical(focal_len_x=100, focal_len_y=100, rotation_angle=0, perspective_angle=0) <--! Note that cylindrical shape shoundn't use with perspective function, perspective function only work for 2d image.
rotation(angle_range=(-60, 60))

=== sizing operation ===
resize(scale_range=(0.9, 1.5))
set_resolution(max_resolution=(80, 80))
set_area(max_area=40000)

== distortion operation ===
noise(min_noise_level: float = 25.0, max_noise_level: float = 50.0)
occlusions( occlusion_images: list, num_occlusions: int = 3)
blur(scale_factor)
stretch(scale_range= (0.5,1.5), min_strech = 0.0)
hue_color(brightness_range = (1.0,1.0), contrast_range = (1.0,1.0), hue_range = (0.0,0.0), saturation_range=(1.0,1.0), gamma_range=(1.0,1.0))
rgb_color(target_color_range=((0, 255), (0, 255), (0, 255)), random_color_range=((0, 255), (0, 255), (0, 255)))
adjust_opacity(opacity: float)
adjust_background_opacity(rgb_color: tuple, background_opacity: float) 
"""
augmenter = Augmenter()

#============================== Config gen 3 fix augment all ==============================================

# video_path = 'video/video.mp4'
# # video_path = 'white' # white, black, rainbow
# logo_folder = 'logos'
# output_folder = 'output'
# random_logo = True
# padding_crop = True
# # whiteout_bboxes = [(620,90,668,175)]
# whiteout_bboxes = []

# areas = [3500,15000,35000]
# perspec_directions = [(0,0),(0,1),(0,-1),(1,0),(-1,0)]
# perspec_angles = [25]
# rotation_angles = [-30,0,30]
# rotation_angles_cylinder = [-30,-15,0,15,30]

# classes = ['aia2']
# num_images = len(perspec_directions)*len(areas)*len(perspec_angles)*len(rotation_angles)*len(classes)
# num_frames = num_images
# i=0
# for _ in range(len(classes)):
#     for area in areas:
#         for rotation_angle in rotation_angles:
#             for perspec_angle in perspec_angles:
#                 for (x,y) in perspec_directions:
#                     augmenter.add_augmentation('set_area',max_area=area,image_range=(i,i))
#                     augmenter.add_augmentation('rotation',angle_range=(rotation_angle,rotation_angle),image_range=(i,i))
#                     augmenter.add_augmentation('set_perspective',angle=perspec_angle,direction=(x,y),image_range=(i,i))
#                     if(area == areas[0] or i%4==0):
#                         augmenter.add_augmentation('blur',scale_factor=1.5,image_range=(i,i))
#                     elif(area == areas[1]):
#                         augmenter.add_augmentation('blur',scale_factor=2.5,image_range=(i,i))
#                     elif(area == areas[2]):
#                         augmenter.add_augmentation('blur',scale_factor=3.5,image_range=(i,i))
#                     i+=1
# # for _ in range(len(classes)):
# #     for x in rotation_angles_cylinder:
# #         augmenter.add_augmentation('set_area',max_area=15000,image_range=(i,i))
# #         augmenter.add_augmentation('cylindrical',focal_len_x=50, focal_len_y=50, rotation_angle = x, perspective_angle=0, image_range=(i,i))
# #         augmenter.add_augmentation('blur',scale_factor=1.5,image_range=(i,i))
# #         i+=1

# augmenter.add_augmentation('noise',min_noise_level=0,max_noise_level=25)
# augmenter.add_augmentation('hue_color',brightness_range = (0.7,1.3), contrast_range = (0.7,1.3))
# # augmenter.add_augmentation('adjust_background_opacity',rgb_color=(255,255,255) , background_opacity=0.25)

# ===================================== Cylinder Config ================================================
video_path = 'video/video.mp4'
# video_path = 'white' # white, black, rainbow
logo_folder = 'logos'
output_folder = 'output'
random_logo = True
padding_crop = False
# whiteout_bboxes = [(620,90,668,175)]
whiteout_bboxes = []

areas = [3500,15000,35000]
perspec_angles = [-30,-15,0,15,30]
rotation_angles = [-30,0,30]

classes = ['RedBull']
num_images = len(areas)*len(perspec_angles)*len(rotation_angles)*len(classes)
num_frames = num_images
i=0
for _ in range(len(classes)):
    for area in areas:
        for rotation_angle in rotation_angles:
            for perspec_angle in perspec_angles:
                # for (x,y) in perspec_directions:
                    augmenter.add_augmentation('set_area',max_area=area,image_range=(i,i))
                    # augmenter.add_augmentation('rotation',angle_range=(rotation_angle,rotation_angle),image_range=(i,i))
                    augmenter.add_augmentation('cylindrical',focal_len_x=80, focal_len_y=80, rotation_angle = rotation_angle, perspective_angle=perspec_angle, image_range=(i,i))
                    if(area == areas[0] or i%4==0):
                        augmenter.add_augmentation('blur',scale_factor=1.5,image_range=(i,i))
                    elif(area == areas[1]):
                        augmenter.add_augmentation('blur',scale_factor=2.5,image_range=(i,i))
                    elif(area == areas[2]):
                        augmenter.add_augmentation('blur',scale_factor=3.5,image_range=(i,i))
                    i+=1
# augmenter.add_augmentation('adjust_background_opacity',rgb_color=(255,255,255) , background_opacity=0.5)
# ===================================== Test Config =====================================================

# # video_path = 'video/video.mp4'
# video_path = 'black' # black, white, rainbow
# logo_folder = 'logos'
# output_folder = 'output'
# random_logo = True
# padding_crop = True
# # whiteout_bboxes = [(620,90,668,175)]
# whiteout_bboxes = []
# classes = ['pepsi2']
# # classes = ['sevenEleven1']
# perspective_angles = [-45,-30,-15,0,15,30,45]
# num_images = len(perspective_angles)
# num_frames = num_images
# augmenter.add_augmentation('set_area',max_area=30000)
# i=0
# for perspective_angle in perspective_angles:
#     augmenter.add_augmentation('cylindrical',focal_len_x=100, focal_len_y=100, rotation_angle = 0, perspective_angle= perspective_angle,image_range=(i,i))
#     i+=1
# augmenter.add_augmentation('adjust_background_opacity',rgb_color=(255,255,255) , background_opacity=0.5)

# ================================== TNN ================================================================
# video_path = 'video/video.mp4'
# logo_folder = 'logos'
# output_folder = 'output'
# random_logo = True
# padding_crop = True
# whiteout_bboxes = [(50, 50, 150, 150)]
# num_images=30
# num_frames = num_images
# classes = ['TNN']
# augmenter.add_augmentation('hue_color', hue_range = (-0.9,0.9), image_range=(0,14))
# augmenter.add_augmentation('rgb_color', target_color_range=((80, 255), (80, 255), (80, 255)), image_range=(15,29))

#============================== Config 3 no random perspective =============================================
# video_path = 'video/video_TNN.mp4'
# logo_folder = 'logos'
# output_folder = 'output'
# num_images = 50
# num_frames = num_images
# classes = ['7up2']
# augmenter.add_augmentation('set_resolution',max_resolution=(200,200),image_range=(0,9))
# augmenter.add_augmentation('set_resolution',max_resolution=(125,125),image_range=(10,24))
# augmenter.add_augmentation('set_resolution',max_resolution=(75,75),image_range=(25,49))

# augmenter.add_augmentation('set_perspective',angle=30,direction=(0,-1),image_range=(0,0))
# augmenter.add_augmentation('set_perspective',angle=30,direction=(0,1),image_range=(1,1))
# augmenter.add_augmentation('set_perspective',angle=30,direction=(-1,0),image_range=(2,2))
# augmenter.add_augmentation('set_perspective',angle=30,direction=(1,0),image_range=(3,3))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(0,-1),image_range=(4,4))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(0,1),image_range=(5,5))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(-1,0),image_range=(6,6))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(1,0),image_range=(7,7))

# augmenter.add_augmentation('set_perspective',angle=30,direction=(0,-1),image_range=(8,9))
# augmenter.add_augmentation('set_perspective',angle=30,direction=(0,1),image_range=(10,11))
# augmenter.add_augmentation('set_perspective',angle=30,direction=(-1,0),image_range=(12,13))
# augmenter.add_augmentation('set_perspective',angle=30,direction=(1,0),image_range=(14,15))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(0,-1),image_range=(16,17))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(0,1),image_range=(18,19))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(-1,0),image_range=(20,21))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(1,0),image_range=(22,23))

# augmenter.add_augmentation('set_perspective',angle=30,direction=(0,-1),image_range=(25,27))
# augmenter.add_augmentation('set_perspective',angle=30,direction=(0,1),image_range=(28,30))
# augmenter.add_augmentation('set_perspective',angle=30,direction=(-1,0),image_range=(31,33))
# augmenter.add_augmentation('set_perspective',angle=30,direction=(1,0),image_range=(34,36))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(0,-1),image_range=(37,39))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(0,1),image_range=(40,42))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(-1,0),image_range=(43,45))
# augmenter.add_augmentation('set_perspective',angle=50,direction=(1,0),image_range=(46,48))

# augmenter.add_augmentation('noise',min_noise_level=25,max_noise_level=50)
# # augmenter.add_augmentation('stretch',scale_range= (0.5,1.5), min_strech = 0.1)
# augmenter.add_augmentation('color',brightness_range = (0.95,1.05), contrast_range = (0.95,1.05))
# augmenter.add_augmentation('blur',scale_factor=random.uniform(1.5,2.5))

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