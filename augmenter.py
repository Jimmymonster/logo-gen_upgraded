import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random
import os

class Augmenter:
    def __init__(self, image_dict = None):
        self.image_dict = image_dict if image_dict is not None else {}   # images should be a list of PIL Image objects
        self.augmentations = []  # List to store the sequence of augmentations

    def add_dict(self, category, images):
        """Add or update a category with a list of PIL images."""
        if not isinstance(images, list) or not all(isinstance(img, Image.Image) for img in images):
            raise ValueError("Images should be provided as a list of PIL Image objects.")
        self.image_dict[category] = images

    def remove_dict(self, category):
        """Remove a category from the dictionary."""
        if category in self.image_dict:
            del self.image_dict[category]
        else:
            raise ValueError(f"Category '{category}' does not exist.")

    def update_dict(self, category, images):
        """Update the images in an existing category."""
        if category in self.image_dict:
            if not isinstance(images, list) or not all(isinstance(img, Image.Image) for img in images):
                raise ValueError("Images should be provided as a list of PIL Image objects.")
            self.image_dict[category] = images
        else:
            raise ValueError(f"Category '{category}' does not exist.")    
    
    def add_augmentation(self, aug_method_name, *args, image_range=None, **kwargs):
        """Add an augmentation method to the sequence with arguments and optional image range."""
        if hasattr(self, aug_method_name):
            self.augmentations.append((getattr(self, aug_method_name), args, kwargs, image_range))
        else:
            raise ValueError(f"Augmentation method '{aug_method_name}' does not exist.")
    
    def clear_augmentations(self):
        self.add_augmentation = []

    def set_augmentations(self, aug_method_data):
        """Set the sequence of augmentations with methods and their arguments."""
        self.augmentations = []
        for method_name, args, kwargs in aug_method_data:
            self.add_augmentation(method_name, *args, **kwargs)
        
    def _apply_augmentations(self, images):
        """Apply all augmentations in the specified order."""
        for aug, args, kwargs, image_range in self.augmentations:
            if image_range is None:
                # Apply to all images
                images = [aug(img, *args, **kwargs) for img in images]
            else:
                start, end = image_range
                # Apply to a specific range of images
                augmented_images = []
                for i, img in enumerate(images):
                    if start <= i <= end:
                        augmented_images.append(aug(img, *args, **kwargs))
                    else:
                        augmented_images.append(img)
                images = augmented_images
        return images

    def augment(self, category, num_images):
        """Select and augment a number of images from the specified category."""
        if category not in self.image_dict:
            raise ValueError(f"Category '{category}' does not exist in the image dictionary.")
        images = self.image_dict[category]
        if not images:
            raise ValueError(f"No images available in category '{category}'.")
        # Ensure num_images is a positive integer
        if num_images <= 0:
            raise ValueError("The number of images to augment must be a positive integer.")
        # Select images evenly, looping over if necessary
        selected_images = [images[i % len(images)] for i in range(num_images)]
        # Apply augmentations to the selected images
        augmented_images =  self._apply_augmentations(selected_images)
        return augmented_images
        
    def flip_horizontal(self, img):
        return ImageOps.mirror(img)

    def flip_vertical(self, img):
        return ImageOps.flip(img)

    def resize(self, image, scale_range=(0.9, 1.5)):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        new_size = (int(image.width * scale), int(image.height * scale))
        return image.resize(new_size, Image.LANCZOS)

    def set_resolution(self, image, max_resolution=(80, 80)):
        original_width, original_height = image.size
        scale = min(max_resolution[0] / original_width, max_resolution[1] / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)

    def noise(self, image_pil: Image.Image, min_noise_level: float = 25.0, max_noise_level: float = 50.0) -> Image.Image:
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')
        image_np = np.array(image_pil)
        rgb = image_np[:, :, :3]
        alpha = image_np[:, :, 3]
        noise_level = random.uniform(min_noise_level, max_noise_level)
        noise = np.random.normal(0, noise_level, rgb.shape)
        noisy_rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)
        noisy_image_np = np.dstack((noisy_rgb, alpha))
        noisy_image_pil = Image.fromarray(noisy_image_np, 'RGBA')
        return noisy_image_pil

    def occlusions(self, image_pil: Image.Image, occlusion_images: list, num_occlusions: int = 3) -> Image.Image:
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')
        result_image = image_pil.copy()
        max_width, max_height = result_image.size
        for _ in range(num_occlusions):
            occlusion_pil = random.choice(occlusion_images)
            scale_factor = random.uniform(0.5, 1.0)
            occlusion_width, occlusion_height = occlusion_pil.size
            new_width = int(max_width * scale_factor)
            new_height = int(occlusion_height * new_width / occlusion_width)
            occlusion_pil = occlusion_pil.resize((new_width, new_height), Image.LANCZOS)
            center_x = max_width / 2
            center_y = max_height / 2
            margin = 20
            min_x = -new_width // 2
            max_x = max_width - new_width // 2
            min_y = -new_height // 2
            max_y = max_height - new_height // 2
            while True:
                x = random.randint(min_x, max_x)
                y = random.randint(min_y, max_y)
                occlusion_center_x = x + new_width / 2
                occlusion_center_y = y + new_height / 2
                if not (center_x - margin < occlusion_center_x < center_x + margin and
                        center_y - margin < occlusion_center_y < center_y + margin):
                    break
            result_image.paste(occlusion_pil, (x, y), occlusion_pil)
        return result_image

    def blur(self, pil_image, scale_factor):
        original_width, original_height = pil_image.size
        new_width = int(original_width / scale_factor)
        new_height = int(original_height / scale_factor)
        downscaled_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        upscaled_image = downscaled_image.resize((original_width, original_height), Image.LANCZOS)
        return upscaled_image

    def perspective(self, image, max_warp=0.2):
        width, height = image.size
        pts1 = np.float32([[0,0], [width-1,0], [0,height-1], [width-1,height-1]])
        pts2 = np.float32([
            [np.random.uniform(0, max_warp * width), np.random.uniform(0, max_warp * height)],
            [width-1 - np.random.uniform(0, max_warp * width), np.random.uniform(0, max_warp * height)],
            [np.random.uniform(0, max_warp * width), height-1 - np.random.uniform(0, max_warp * height)],
            [width-1 - np.random.uniform(0, max_warp * width), height-1 - np.random.uniform(0, max_warp * height)]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # Apply perspective transform to the image
        image_np = np.array(image)
        warped_image = cv2.warpPerspective(image_np, matrix, (width, height))
        # Convert to grayscale and find non-zero points
        gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        non_zero_points = cv2.findNonZero(gray_warped)
        if non_zero_points is None:
            return image  # No valid transformation, return original image
        # Compute the bounding box
        x_min, y_min = np.min(non_zero_points, axis=0).flatten()
        x_max, y_max = np.max(non_zero_points, axis=0).flatten()
        # Ensure coordinates are within bounds
        x_min, y_min = int(max(0, x_min)), int(max(0, y_min))
        x_max, y_max = int(min(width - 1, x_max)), int(min(height - 1, y_max))
        # Crop the warped image to the bounding box
        cropped_warped_image = warped_image[y_min:y_max+1, x_min:x_max+1]
        result_image = Image.fromarray(cropped_warped_image)
        return result_image

    def rotation(self, image, angle_range=(-60, 60)):
        angle = np.random.uniform(angle_range[0], angle_range[1])
        width, height = image.size
        angle_rad = np.radians(angle)
        cos_a = np.abs(np.cos(angle_rad))
        sin_a = np.abs(np.sin(angle_rad))
        new_width = int(width * cos_a + height * sin_a)
        new_height = int(width * sin_a + height * cos_a)
        rotated_image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        resized_image = rotated_image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
        return resized_image
        
    def stretch(self, image_pil: Image.Image, scale_range= (0.5,1.5), min_strech = 0.0) -> Image.Image:
        original_width, original_height = image_pil.size
        scale_x=1.0
        scale_y=1.0
        if(random.randint(0, 1) != 0):
            if(random.randint(0, 1) != 0):
                scale_x = random.uniform(scale_range[0],1-min_strech)
            else:
                scale_x = random.uniform(scale_range[1],1+min_strech)
        else:
            if(random.randint(0, 1) != 0):
                scale_y = random.uniform(scale_range[0],1-min_strech)
            else:
                scale_y = random.uniform(scale_range[1],1+min_strech)
        new_width = int(original_width * scale_x)
        new_height = int(original_height * scale_y)
        stretched_image = image_pil.resize((new_width, new_height), Image.Resampling.BILINEAR)
        return stretched_image

    def color(self, img: Image.Image,brightness_range = (1.0,1.0), contrast_range = (1.0,1.0), hue_range = (0.0,0.0), saturation_range=(1.0,1.0), gamma_range=(1.0,1.0)) -> Image.Image:
        if img.mode not in ('RGB', 'RGBA'):
            raise ValueError("Image mode must be RGB or RGBA")
        alpha = None
        if img.mode == 'RGBA':
            img, alpha = img.convert('RGB'), img.getchannel('A')
        brightness_factor = random.uniform(brightness_range[0],brightness_range[1])
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        contrast_factor = random.uniform(contrast_range[0],contrast_range[1])
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        hue_factor = random.uniform(hue_range[0],hue_range[1])
        saturation_factor = random.uniform(saturation_range[0],saturation_range[1])

        def adjust_hue_saturation(image, hue_factor, saturation_factor):
            hsv_image = image.convert('HSV')
            h, s, v = hsv_image.split()
            if h.mode != 'L' or s.mode != 'L' or v.mode != 'L':
                raise ValueError("HSV channels are not in the correct mode")
            enhancer = ImageEnhance.Color(hsv_image)
            s = enhancer.enhance(saturation_factor).split()[1]
            h = h.point(lambda p: (p + hue_factor * 255) % 255)
            hsv_image = Image.merge('HSV', (h, s, v))
            return hsv_image.convert('RGB')
        img = adjust_hue_saturation(img, hue_factor, saturation_factor)
        # Randomly adjust gamma
        # gamma_factor = random.uniform(0.5, 2.0)
        gamma_factor = random.uniform(gamma_range[0],gamma_range[1])
        img = ImageEnhance.Brightness(img).enhance(gamma_factor)
        # Reattach alpha channel if it was separated
        if alpha:
            img = Image.merge('RGBA', (img.convert('RGB').split() + (alpha,)))
        return img