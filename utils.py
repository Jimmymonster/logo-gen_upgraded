import random
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw

def load_images_from_directory(directory_path: str) -> list:
    images = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                image = Image.open(file_path).convert('RGBA')
                images.append(image)
            except Exception as e:
                print(f"Failed to load image {file_path}: {e}")
    return images

def extract_random_frames(video_path, num_frames):
    """Randomly extract a specific number of frames from the video."""
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames > total_frames:
        raise ValueError("Number of frames requested exceeds the total number of frames in the video.")

    # Randomly select frame indices
    selected_indices = sorted(random.sample(range(total_frames), num_frames))
    
    frames = []
    for index in selected_indices:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = video_capture.read()
        if not ret:
            break
        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    video_capture.release()
    return frames

def place_augmented_image(frame, augmented_image, padding_crop = False):
    """Randomly place an augmented image on a frame and return the bounding box."""
    frame_height, frame_width, _ = frame.shape
    aug_width, aug_height = augmented_image.size

    # Convert augmented image to RGBA if it has an alpha channel
    if augmented_image.mode == 'RGBA':
        augmented_image = augmented_image.convert('RGBA')  # Keep alpha for blending

    # Random position for placing the augmented image
    x = random.randint(0, frame_width - aug_width)
    y = random.randint(0, frame_height - aug_height)

    # Convert augmented image to numpy array
    augmented_image_np = np.array(augmented_image)

    # Handle image with alpha channel
    if augmented_image_np.shape[2] == 4:
        alpha = augmented_image_np[:, :, 3] / 255.0
        for c in range(3):  # RGB channels
            frame[y:y+aug_height, x:x+aug_width, c] = (
                alpha * augmented_image_np[:, :, c] +
                (1 - alpha) * frame[y:y+aug_height, x:x+aug_width, c]
            )
    else:
        # For images without alpha channel, directly overlay
        frame[y:y+aug_height, x:x+aug_width] = augmented_image_np

    # Compute bounding box in YOLO format
    if(padding_crop):
        x_center = (x + aug_width / 2) / frame_width + random.uniform(-3,3)/frame_width
        y_center = (y + aug_height / 2) / frame_height + random.uniform(-3,3)/frame_height
        width = aug_width / frame_width + random.uniform(6,10)/frame_width
        height = aug_height / frame_height + random.uniform(6,10)/frame_height
    else:
        x_center = (x + aug_width / 2) / frame_width 
        y_center = (y + aug_height / 2) / frame_height
        width = aug_width / frame_width
        height = aug_height / frame_height

    return frame, (x_center, y_center, width, height)

def whiteout_areas(frame, whiteout_bboxes):
    """Whiteout specific areas of the frame given bounding boxes."""
    if isinstance(frame, np.ndarray):
        frame_pil = Image.fromarray(frame)
    else:
        frame_pil = frame

    draw = ImageDraw.Draw(frame_pil)
    for bbox in whiteout_bboxes:
        draw.rectangle(bbox, fill="white")

    if isinstance(frame, np.ndarray):
        return np.array(frame_pil)
    else:
        return frame_pil

def insert_augmented_images(frames, augmented_images, class_indices, whiteout_bboxes=[],padding_crop = False):
    """Randomly place multiple augmented images into frames and return the frames with bounding boxes."""
    num_augmented = len(augmented_images)
    num_frames = len(frames)
    
    # Calculate how many images to place per frame
    if num_augmented > num_frames:
        num_replacements = num_augmented // num_frames + 1
    else:
        num_replacements = 1
    
    # Prepare bounding boxes list
    bounding_boxes = []

    for i, augmented_image in enumerate(augmented_images):
        frame_index = i % num_frames
        frame = frames[frame_index]

        # Whiteout specific areas in the frame
        if whiteout_bboxes:
            frame = whiteout_areas(frame, whiteout_bboxes)
        
        # Place the augmented image on the frame
        frame_with_augmentation, bbox = place_augmented_image(frame, augmented_image, padding_crop)
        frames[frame_index] = frame_with_augmentation
        
        # Append bounding box information
        bounding_boxes.append((frame_index, class_indices[i % len(class_indices)], bbox))

    # Reorganize bounding boxes by frame
    frames_with_augmentations = [frames[i] for i in range(num_frames)]
    
    # Sort bounding boxes by frame index
    bounding_boxes.sort(key=lambda x: x[0])
    
    # Reorganize bounding boxes by frame index
    frame_bounding_boxes = {i: [] for i in range(num_frames)}
    for frame_index, class_id, bbox in bounding_boxes:
        frame_bounding_boxes[frame_index].append((class_id, bbox))
    
    # Flatten bounding boxes for each frame
    bounding_boxes = [frame_bounding_boxes[i] for i in range(num_frames)]

    return frames_with_augmentations, bounding_boxes

def save_yolo_format(frames, bounding_boxes, output_folder, class_names):
    """Save frames and labels in YOLO format."""
    # Create necessary directories
    images_folder = os.path.join(output_folder, 'images')
    labels_folder = os.path.join(output_folder, 'labels')
    
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)

    classes_file = os.path.join(output_folder, 'classes.txt')
    
    # Write class names to class.txt
    with open(classes_file, 'w') as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}\n")

    # Save images and labels
    for i, (frame, bboxes) in enumerate(zip(frames, bounding_boxes)):
        image = Image.fromarray(frame)
        
        # Ensure image is in RGB mode before saving
        if(i%2==0):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_filename = os.path.join(images_folder, f"frame_{i:04d}.jpg")
            label_filename = os.path.join(labels_folder, f"frame_{i:04d}.txt")

        else:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            image_filename = os.path.join(images_folder, f"frame_{i:04d}.png")
            label_filename = os.path.join(labels_folder, f"frame_{i:04d}.txt")
        
        # Save the frame as an image
        image.save(image_filename)
        
        # Save the labels for the augmented images in this frame
        with open(label_filename, 'w') as f:
            for class_id, bbox in bboxes:
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

