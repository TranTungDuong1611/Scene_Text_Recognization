import ultralytics
import json
from ultralytics import YOLO

yolo_yaml_path = 'yolo_data/data.yaml'
config_path = 'src/config.json'

def load_json_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    return config

config = load_json_config(config_path)

# Load model
model = YOLO('yolo11m.pt')

# Train model
results = model.train(
    data=yolo_yaml_path,
    epochs=config['yolov11']['epochs'],
    imgsz=config['yolov11']['image_size'],
    cache=config['yolov11']['cache'],
    patience=config['yolov11']['patience'],
    plots=config['yolov11']['plots']
)

# Evaluate model
model_path = 'checkpoints/yolov11m.pt'
model = YOLO(model_path)

metrics = model.val()