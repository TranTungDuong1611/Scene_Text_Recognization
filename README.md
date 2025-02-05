# Scene Text Recognition

## Overview
This project implements a two-stage scene text recognition pipeline. First, the `yolov11` model detects text regions; then, the `CRNN` model reads the text. It efficiently handles challenges like diverse fonts, complex backgrounds, and various orientations.

Download the dataset [here](https://drive.google.com/file/d/1kUy2tuH-kKBlFCNA0a9sqD2TG4uyvBnV/view).

## Pipeline
![Pipeline](images/pipeline.png)

## Results
![Result 1](images/result1.png)
![Result 6](images/result6.png)
![Result 7](images/result7.png)
![Result 3](images/result3.png)
![Result 4](images/result4.png)
![Result 5](images/result5.png)

## Deploy on Hugging Face
Try the live demo on Hugging Face Spaces: [Scene Text Recognition](https://huggingface.co/spaces/TungDuong/Scene_Text_Recognization)

![Hugging Face Interface](images/hg.png)

## How to Run
1. Navigate to the project directory:
   ```bash
   cd path/to/Scene_Text_Recognization
2. Run the prediction script:
    ```bash
    python src/predict.py --image_path=path/to/your/image --save_path=path/to/saved/directory
# Reporoduce
## Dataset Structure

    Dataset
    ├── apanar_06.08.2002
    │   └── image.jpg
    │   └── ...
    │── lfsosa_12.08.2002
    │   └── image.jpg
    │   └── ...
    ├── ryoungt_03.09.2002
    │   └── image.jpg
    │   └── ...
    ├── ryoungt_05.08.2002
    │   └── image.jpg
    │   └── ...
    ├── locations.xml
    ├── segmentation.xml
    ├── words.xml

## Preparing the Datasets
- For YOLO dataset
    ```bash
    python src/Text_Localization/prepare_dataset.py
- For CRNN dataset
    ```bash
    python src/Text_Recognization/prepare_dataset.py