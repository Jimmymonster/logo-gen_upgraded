from augmenter import Augmenter
from utils import load_images_from_directory, insert_augmented_images, save_yolo_format, save_yolo_obbox_format
import os,shutil
import config
import time
from PIL import Image

start_time = time.time()

def read_yolo_project_with_class_map(image_folder, label_folder):
    image_list = []         # List of PIL images
    label_list = []         # List of lists, each containing bounding boxes [x1, y1, x2, y2, x3, y3, x4, y4] for each image
    class_index_list = []   # List mapping each bounding box to its class index for each image

    # Iterate through all image files
    # i=0
    for image_file in os.listdir(image_folder):
        # i+=1
        # if i==51:
        #     break
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            label_path = os.path.join(label_folder, image_file.rsplit('.', 1)[0] + '.txt')
            
            # Load the image
            image = Image.open(image_path)
            image= image.convert('RGB')
            image_list.append(image)

            # Initialize label storage for this image
            image_labels = []   # To store bounding boxes for this image
            class_indices = []  # To store class indices for this image

            # Read the corresponding label file
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    labels = f.readlines()

                img_w, img_h = image.size

                # Parse each label line
                for label in labels:
                    parts = label.strip().split()
                    class_id, x_center, y_center, width, height = map(float, parts)

                    # Convert relative coordinates to absolute pixel values
                    x_center *= img_w
                    y_center *= img_h
                    width *= img_w
                    height *= img_h

                    # Calculate corner points for the bounding box
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center - height / 2
                    x3 = x_center + width / 2
                    y3 = y_center + height / 2
                    x4 = x_center - width / 2
                    y4 = y_center + height / 2

                    # Store the bounding box in (x1, y1, x2, y2, x3, y3, x4, y4) format
                    image_labels.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

                    # Add the class index for this bounding box
                    class_indices.append(class_id)
            
            # Append labels and class indices for this image
            label_list.append(image_labels if image_labels else [[(0, 0), (0, 0), (0, 0), (0, 0)]])
            class_index_list.append(class_indices if class_indices else [-1])


    return image_list, label_list, class_index_list

def save_yolo_augmented_data(augmented_images, obboxs, class_index_list, output_image_folder, output_label_folder, classes_file, output_class_file,  base_name):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    # Iterate through augmented images and their corresponding bounding boxes and class indices
    for i, (aug_image, bboxes, class_indices) in enumerate(zip(augmented_images, obboxs, class_index_list)):
        # Define the image and label filenames
        image_filename = f"{base_name}_{i}.png"
        label_filename = f"{base_name}_{i}.txt"
        
        # Save the augmented image
        image_path = os.path.join(output_image_folder, image_filename)
        # Check if the image is in CMYK mode
        if aug_image.mode == 'CMYK':
            # Convert the image to RGB (PNG doesn't support CMYK directly)
            aug_image = aug_image.convert('RGBA')
        # Save the image in PNG format
        aug_image.save(image_path, 'PNG', icc_profile=None)

        # Prepare the label content in YOLO format
        img_w, img_h = aug_image.size
        label_path = os.path.join(output_label_folder, label_filename)
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_indices):
                # Extract bounding box points
                x1, y1= bbox[0]
                x2, y2= bbox[1]
                x3, y3= bbox[2]
                x4, y4= bbox[3]
                # Calculate the bounding box center, width, and height in relative values
                minx = min(x1,x2,x3,x4)
                maxx = max(x1,x2,x3,x4)
                miny = min(y1,y2,y3,y4)
                maxy = max(y1,y2,y3,y4)
                x_center = (minx + maxx) / 2.0 / img_w
                y_center = (miny + maxy) / 2.0 / img_h
                width = (maxx - minx) / img_w
                height = (maxy - miny) / img_h
                if(class_id != -1):
                # Write the label in YOLO format: class_id x_center y_center width height
                    f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def augment_folder(input_yolo_dir, output_yolo_dir):
    yolo_project_folder = input_yolo_dir
    image_folder = os.path.join(yolo_project_folder,'images')
    label_folder = os.path.join(yolo_project_folder,'labels')
    classes_file = os.path.join(yolo_project_folder,'classes.txt')
    output_folder = output_yolo_dir
    output_image_folder = os.path.join(output_folder,'images')
    output_label_folder = os.path.join(output_folder,'labels')
    output_class_file = os.path.join(output_folder,'classes.txt')

    #clear output path
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filelist = os.listdir(yolo_project_folder)
    for filename in filelist:
        file_path = os.path.join(yolo_project_folder, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, output_folder)

    augmenter = Augmenter()
    # num_images = len(os.listdir(label_folder))
    # augmenter.add_augmentation('set_perspective',angle=25,direction=(0,-1))
    num_images = len(os.listdir(label_folder))*5
    one_chunk = num_images//5
    augmenter.add_augmentation('rotation',angle_range=(-15,-15),image_range=(0,one_chunk-1))
    augmenter.add_augmentation('rotation',angle_range=(0,0),image_range=(one_chunk,2*one_chunk-1))
    augmenter.add_augmentation('rotation',angle_range=(15,15),image_range=(2*one_chunk,3*one_chunk-1))
    augmenter.add_augmentation('set_perspective',angle=25,direction=(0,1),image_range=(3*one_chunk,4*one_chunk-1))
    augmenter.add_augmentation('set_perspective',angle=25,direction=(0,-1),image_range=(4*one_chunk,5*one_chunk-1))
    image_list,label_list,class_list = read_yolo_project_with_class_map(image_folder,label_folder)

    dict_name="name"
    augmenter.add_dict(dict_name,image_list)
    augmenter.add_obbox_dict(dict_name,label_list)
    augmenter.set_obbox_class_index(dict_name,class_list)

    augmented_image, class_index_list = augmenter.augment(dict_name,num_images,random=True)
    augmented_image,obboxs= augmented_image
    save_yolo_augmented_data(augmented_image, obboxs, class_index_list, output_image_folder, output_label_folder, classes_file, output_class_file, "augment_image")


base_dir = "C:/Users/thanapob/My File/yolo-dataset/11base"
# dir_list= os.listdir(base_dir)
# for item in dir_list:
#     if os.path.isdir(base_dir):
#         augment_folder(base_dir+item,base_dir[:-1]+"_augmented/"+item)
#     else:
#         shutil.copy(base_dir+item,base_dir[:-1]+"_augmented/"+item)

item = "shell"
augment_folder(os.path.join(base_dir,item),f"C:/Users/thanapob/My File/yolo-dataset/11augmented_2/{item}")
# item = "bcp"
# augment_folder(os.path.join(base_dir,item),f"C:/Users/thanapob/My File/yolo-dataset/11augmented_2/{item}")
# augment_folder(os.path.join(base_dir,item),base_dir[:-1]+"_augmented/"+item)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")