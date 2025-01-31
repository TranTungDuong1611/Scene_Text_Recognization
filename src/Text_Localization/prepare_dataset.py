import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import xml.etree.ElementTree as ET
import shutil
import yaml

from sklearn.model_selection import train_test_split

location_path = r'Dataset/locations.xml'
tree = ET.parse(location_path)
root = tree.getroot()


def get_gt_bboxes(location_path):
    """get all the gt bbox of text in dataset

    Args:
        location_path: (path)
    Return:
        gt_imagepaths[1] (list): image's name
        gt_locations (list): bboxes of each image
    """
    gt_imagepaths = []
    gt_imagesizes = []
    gt_locations = []
    
    for image in root:
        # get path to image
        image_name = image[0].text
        image_path = os.path.join('Dataset', image_name)
        gt_imagepaths.append(image_path)
        
        # get the image size
        w = image[1].get('x')
        h = image[1].get('y')
        gt_imagesizes.append([w, h])
        
        # bboxes in the image
        bbs = []
        for bbox in image[2]:
            x = np.int64(float(bbox.get('x')))
            y = np.int64(float(bbox.get('y')))
            width = np.int64(float(bbox.get('width')))
            height = np.int64(float(bbox.get('height')))
            bbs.append([x, y, width, height])
            
        gt_locations.append(bbs)
    
    return gt_imagepaths, gt_imagesizes, gt_locations
    
gt_imagepaths, gt_imagesizes, gt_locations = get_gt_bboxes(location_path)

def visualize_gt_bboxes(image_path, gt_locations):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for gt_location in gt_locations:
        x, y, width, height = gt_location
        
        image = cv2.rectangle(image, (x, y), (x+width, y+height), color=(255, 0, 0), thickness=2)
        
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    
def convert_yolo_format(gt_locations, gt_imagesizes):
    gt_locations_yolo = []
    
    for image, image_size in zip(gt_locations, gt_imagesizes):
        gt_location_yolo = []
        for gt_location in image:
            x, y, w, h = gt_location
            image_width, image_height = image_size
            
            xc = (x + w/2) / float(image_width)
            yc = (y + h/2) / float(image_height)
            width = w / float(image_width)
            height = h / float(image_height)
            
            # class = 0 -> meaning contains text
            class_id = 0
            gt_location_yolo.append([class_id, xc, yc, width, height])
        
        gt_locations_yolo.append(gt_location_yolo)
        
    return gt_locations_yolo
        
gt_locations_yolo = convert_yolo_format(gt_locations, gt_imagesizes)

def save_data_into_yolo_folder(data, src_img_dir, save_dir):
    # Create folder if not exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Make images and labels folder
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)
    
    # write data into yolo folder
    for dt in data:
        # copy data
        image_path = dt[0]
        shutil.copy(image_path, os.path.join(save_dir, 'images'))
        
        #copy labels
        image_name = os.path.basename(image_path)
        image_name = os.path.splitext(image_name)[0]
        
        with open(os.path.join(save_dir, 'labels', f'{image_name}.txt'), "w") as f:
            for label in dt[1]:
                label_str = " ".join(map(str, label))
                f.write(f'{label_str}\n')

seed = 0
val_size = 0.2
test_size = 0.15
dataset = [[gt_imagepath, gt_location_yolo] for gt_imagepath, gt_location_yolo in zip(gt_imagepaths, gt_locations_yolo)]
train_data, test_data = train_test_split(dataset, test_size=test_size, random_state=42, shuffle=True)
train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42, shuffle=True)

save_yolo_data_dir = 'yolo_data'
os.makedirs(save_yolo_data_dir, exist_ok=True)
save_data_into_yolo_folder(
    data=train_data,
    src_img_dir=save_yolo_data_dir,
    save_dir=os.path.join(save_yolo_data_dir, 'train')
)
save_data_into_yolo_folder(
    data=val_data,
    src_img_dir=save_yolo_data_dir,
    save_dir=os.path.join(save_yolo_data_dir, 'val')
)
save_data_into_yolo_folder(
    data=test_data,
    src_img_dir=save_yolo_data_dir,
    save_dir=os.path.join(save_yolo_data_dir, 'test')
)

class_label = ['text']
# Create data.yaml file
data_yaml = {
    "path": 'yolo_data',
    'train': 'train/images',
    'test': 'test/images',
    'val': 'val/images',
    'nc': 1,
    'names': class_label
}

yolo_yaml_path = os.path.join(save_yolo_data_dir, 'data.yaml')
with open(yolo_yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)