import gradio as gr
import numpy as np
import os
import json
import cv2
import sys
import torch
import torch.nn as nn
import torchvision

sys.path.append(os.getcwd())
from src.predict import *

def visualize_image(image, detections):
    for bbox, detected_class, conf, text, _ in detections:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        image = cv2.putText(image, f"{conf:.2f} {text}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    return image

def pipeline(image):
    image = np.array(image)
    
    predictions = prediction(image)
    
    # Filter low conf boxes
    filter_predictions = []
    dict_predictions = {}
    num_textbox = 1
    for bbox, cls, conf, text, encoded_text in predictions:
        if conf > 0.7:
            filter_predictions.append([bbox, cls, conf, text, encoded_text])
            
            xmin, ymin, xmax, ymax = bbox
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            dict_predictions.update({
                f"textbox {num_textbox}":{
                    "bounding box": str([xmin, ymin, xmax, ymax]),
                    "conf": np.round(conf, 2),
                    "text": text
                }
            })
            num_textbox += 1
            
    image = visualize_image(image, filter_predictions)
    return image, json.dumps(dict_predictions, indent=5)

demo = gr.Interface(
    fn=pipeline,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=[
            gr.Image(type="pil", label="Output Image"),
            gr.Textbox(type="text", label="Recognized Text")
        ],
    title="Scene Text Recognization",
    description="Recognize text in scene images"
)

demo.launch(share=True)