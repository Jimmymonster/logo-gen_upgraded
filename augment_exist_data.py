from augmenter import Augmenter
from utils import load_images_from_directory
from PIL import Image
import os

path = "C:/Users/thanapob/My File/logo/l_dataset/train"
folder_list = os.listdir(path)

augmenter = Augmenter()

perspec_directions = [(0,0),(0,1),(0,-1)]
perspec_angles = [20]
rotation_angles = [-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90]
augmenter.add_augmentation('hue_color',brightness_range = (0.5,1.5), contrast_range = (0.5,1.5), hue_range = (-0.05,0.05), saturation_range=(0.95,1.05))
augmenter.add_augmentation('noise',min_noise_level=0,max_noise_level=25)
i=0

for rotation_angle in rotation_angles:
    for perspec_angle in perspec_angles:
            for (x,y) in perspec_directions:
                augmenter.add_augmentation('rotation',angle_range=(rotation_angle,rotation_angle),image_range=(i,i))
                augmenter.add_augmentation('set_perspective',angle=perspec_angle,direction=(x,y),image_range=(i,i))
                i+=1

for folder in folder_list:
    path_to_images = os.path.join(path,folder)
    images = load_images_from_directory(path_to_images)
    idx = 0
    for image in images:
        augmenter.add_dict(folder,[image])
        augmented_images, _ = augmenter.augment(folder, i, random=True)

        for img in augmented_images:
             name = folder + "augmented" + str(idx) + ".png"
             img.save(os.path.join(path_to_images,name))
             idx+=1

        augmenter.clear_dict()
        