import os
import random
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

root_path = 'Dataset'
# words_path = os.path.join(root_path, 'words.xml')
# tree = ET.parse(words_path)
# root = tree.getroot()


def extract_data_from_xml(root_path):
    words_path = os.path.join(root_path, 'words.xml')
    tree = ET.parse(words_path)
    root = tree.getroot()
    
    image_paths = []
    image_sizes = []
    image_labels = []
    bboxes = []
    
    for image in root:
        imagename = image[0].text
        image_path = os.path.join(root_path, imagename)
        image_paths.append(image_path)
        
        image_height = image[1].get('x')
        image_width = image[1].get('y')
        image_sizes.append([image_height, image_width])
        
        bboxes_in_image = []
        labels_in_bboxes = []
        for bbox in image[2]:
            x = float(bbox.get('x'))
            y = float(bbox.get('y'))
            width = float(bbox.get('width'))
            height = float(bbox.get('height'))
            bboxes_in_image.append([x, y, width, height])
            
            # get text in this bbox
            labels = bbox.find('tag').text
            labels_in_bboxes.append(labels)
            
        bboxes.append(bboxes_in_image)
        image_labels.append(labels_in_bboxes)
        
    return image_paths, image_sizes, bboxes, image_labels

def visualize_gt_bboxes(image_path, gt_locations, gt_labels):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for gt_location, gt_label in zip(gt_locations, gt_labels):
        x, y, width, height = gt_location
        x, y, width, height = int(x), int(y), int(width), int(height)
        
        image = cv2.rectangle(image, (x, y), (x+width, y+height), color=(255, 0, 0), thickness=2)
        image = cv2.putText(image, gt_label, (x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 3, color=(255, 0, 0), thickness=2)
        
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def split_bboxes_from_image(image_paths, image_labels, bboxes, save_dir):
    """create a new dataset contains bboxes and corresponding labels

    Args:
        image_paths
        image_labels
        bboxes
        save_dir
    Return:
        non-return
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('unvalid_images', exist_ok=True)
    
    bboxes_idx = 0
    unvalid_bboxes = 0
    new_labels = []         # List to store labels
    for image_path, bbox, label in zip(image_paths, bboxes, image_labels):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            print(image_path)
            continue
        
        for bb, lb in zip(bbox, label):
            x, y, width, height = bb
            x, y, width, height = int(x), int(y), int(width), int(height)
            
            cropped_text = image[y:y+height, x:x+width]
            
            # Filter if x, y, width, height is invalid cordinates
            if x < 0 or y < 0 or width < 0 or height < 0:
                continue
            
            # Filter text contain special characters
            if 'é' in [lb[i].lower() for i in range(len(lb))] or 'ñ' in [lb[i].lower() for i in range(len(lb))] or '£' in [lb[i].lower() for i in range(len(lb))]:
                continue
            
            # Filter out if text is too light or too dark
            if np.mean(cropped_text) < 30 or np.mean(cropped_text) > 230:
                cv2.imwrite(f'unvalid_images\\unvalid_image{unvalid_bboxes}_{lb}.jpg', cropped_text)
                unvalid_bboxes += 1
                continue
            
            # Filter out if image is too small
            if width < 10 or height < 10:
                cv2.imwrite(f'unvalid_images\\unvalid_image{unvalid_bboxes}_{lb}.jpg', cropped_text)
                unvalid_bboxes += 1
                continue
            
            new_image_path = os.path.join(save_dir, f'cropped_image{bboxes_idx}.jpg')
            cv2.imwrite(new_image_path, cropped_text)
            new_label = new_image_path + '\t' + lb
            new_labels.append(new_label)
            bboxes_idx += 1
            
        # Write labels into a text file
        with open(os.path.join(save_dir, 'labels.txt'), "w") as f:
            for new_label in new_labels:
                f.write(f'{new_label}\n')
                

def build_vocab(root_dir):
    img_paths = []
    labels = []
    
    # Read labels from text file
    with open(os.path.join(save_dir, 'labels.txt'), "r") as f:
        for label in f:
            labels.append(label.strip().split("\t")[1])
            img_paths.append(label.strip().split("\t")[0])
            
    # build the vocab
    vocab = set()
    for label in labels:
        for i in range(len(label)):
            vocab.add(label[i].lower())
            
    # "blank" character
    vocab = list(sorted(vocab))
    vocab = "".join(vocab)
    blank_char = '@'
    vocab += blank_char
    
    # build a dictionary convert from vocab to idx and idx to vocab
    char_to_idx = {
        char: idx + 1 for idx, char in enumerate(vocab)
    }
    idx_to_char = {
        idx: char for char, idx in char_to_idx.items()
    }
    
    return char_to_idx, idx_to_char, labels
    


def encode(label, char_to_idx, labels):
    max_length_label = np.max([len(lb) for lb in labels])
    
    # encode label
    encoded_label = torch.tensor(
                        [char_to_idx[char] for char in label],
                        dtype=torch.int32
                    )
    label_len = len(encoded_label)
    length = torch.tensor(
                label_len,
                dtype=torch.int32
            )
    padded_label = F.pad(
                        encoded_label,
                        (0, max_length_label-label_len),
                        value=-1
                    )
    return padded_label, length

def decode(encoded_label, idx_to_char):
    label = []
    for i in range(len(encoded_label)):
        if encoded_label[i] == 0:
            break
        else:
            label.append(idx_to_char[encoded_label[i]])
    label = "".join(label)
    return label


image_paths, image_sizes, bboxes, image_labels = extract_data_from_xml(root_path)
save_dir = 'Dataset/ocr_dataset'
split_bboxes_from_image(image_paths, image_labels, bboxes, save_dir)
char_to_idx, idx_to_char, labels = build_vocab(root_path)