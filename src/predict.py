import os
import sys
import json
import cv2
import argparse
import matplotlib.pyplot as plt

import ultralytics
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet101
from torchvision import transforms

sys.path.append(os.getcwd())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from ultralytics import YOLO
from src.Text_Recognization.text_recognization import *
from src.Text_Recognization.prepare_dataset import *

# config
def load_json_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    return config

config = load_json_config('src/config.json')

# char to idx
char_to_idx, idx_to_char = build_vocab('Dataset')

# text detection model
text_det_model_path = 'checkpoints/yolov11m.pt'
yolo = YOLO(text_det_model_path)

# text recognition model
text_rec_model_path = 'checkpoints/crnn_extend_vocab.pt'

# rcnn model
rcnn_model = CRNN(vocab_size=74, hidden_size=config['CRNN']['hidden_size'], n_layers=config['CRNN']['n_layers'])
rcnn_model.load_state_dict(torch.load(text_rec_model_path, weights_only=True, map_location=torch.device('cpu')))

def text_detection(img_path, text_det_model):
    text_det_results = text_det_model(img_path, verbose=False)[0]
    
    bboxes = text_det_results.boxes.xyxy.tolist()
    classes = text_det_results.boxes.cls.tolist()
    names = text_det_results.names
    confs = text_det_results.boxes.conf.tolist()
    
    return bboxes, classes, names, confs

def visualize_gt_bboxes_yolo(image_path, gt_location_yolo):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to original format
    for data in gt_location_yolo:
        xmin, ymin, xmax, ymax = data
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2)
        
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def text_recognization(image, data_transforms, text_reg_model, idx_to_char=idx_to_char, device=device):
    transformsed_image = data_transforms(image)
    transformsed_image = transformsed_image.unsqueeze(0).to(device)
    text_reg_model.to(device)
    text_reg_model.eval()
    
    with torch.no_grad():
        preds = text_reg_model(transformsed_image)
        _, idx = torch.max(preds, dim=2)
        idx = idx.view(-1)
        text = decode(idx, idx_to_char, char_to_idx)
        
    return text, idx

def visualize_detection(image, detections):
    plt.figure(figsize=(10, 8))
    
    for bbox, detected_classes, conf, text, _ in detections:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        image = cv2.putText(image, f"{conf:.2f} {text}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return image
    
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((100, 400)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def prediction(image, text_det_model=yolo, text_reg_model=rcnn_model, idx_to_char=idx_to_char, char_to_idx=char_to_idx, data_transforms=data_transforms, device=device):
    # detection
    bboxes, classes, names, confs = text_detection(image, text_det_model)
    
    predictions = []
    for bbox, cls, conf in zip(bboxes, classes, confs):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        detected_text = image[y1:y2, x1:x2]
        text, encoded_text = text_recognization(detected_text, data_transforms, text_reg_model, idx_to_char, device)
        predictions.append((bbox, cls, conf, text, encoded_text))
        print(bbox, cls, conf, text)
    
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='Path to the image')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the image')
    args = parser.parse_args()
    image_path = args.image_path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detections = prediction(image)
    image = visualize_detection(image, detections)
    
    if args.save_path:
        print(f"Saving the image to {os.path.join(args.save_path, 'predicted_image.jpg')}")
        cv2.imwrite(os.path.join(args.save_path, 'predicted_image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
if __name__ == '__main__':
    main()
