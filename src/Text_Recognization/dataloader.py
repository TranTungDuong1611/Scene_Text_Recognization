import torch
import torchvision
import json
import sys

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from src.Text_Recognization.prepare_dataset import *

# data augmentation
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((100, 420)),
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5
            ),
            transforms.GaussianBlur(3),
            transforms.RandomAffine(
                degrees=1,
                shear=1
            ),
            transforms.RandomPerspective(
                distortion_scale=0.3,
                p=0.5
            ),
            transforms.RandomRotation(degrees=15),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((100, 420)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
}

def load_json_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    return config

# Dataloader
class STRDataset(Dataset):
    def __init__(self, image_paths, labels, char_to_idx, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.transforms= transforms
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            image = self.transforms(image)
        
        label_encoded, length = encode(self.labels[idx], self.char_to_idx, self.labels)
        
        return image, label_encoded, length
    
def get_dataloader():
    val_size = 0.1
    test_size = 0.1
    root_path = 'Dataset'
    config_path = 'src/config.json'

    # get image paths and labels
    image_paths, labels = get_imagepaths_and_labels(root_path)
    char_to_idx, idx_to_char = build_vocab(root_path)


    config = load_json_config(config_path)

    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=val_size, random_state=42, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=42, shuffle=True)
    train_dataset = STRDataset(X_train, y_train, char_to_idx, transforms=data_transforms['train'])
    train_loader = DataLoader(train_dataset, batch_size=config['CRNN']['batch_size'], shuffle=True)

    val_dataset = STRDataset(X_val, y_val, char_to_idx, transforms=data_transforms['val'])
    val_loader = DataLoader(val_dataset, batch_size=config['CRNN']['batch_size'], shuffle=True)

    test_dataset = STRDataset(X_test, y_test, char_to_idx, transforms=data_transforms['val'])
    test_loader = DataLoader(test_dataset, batch_size=config['CRNN']['batch_size'], shuffle=True)
    
    return train_loader, val_loader, test_loader