
# IMPORTS 

import os
from ultralytics import YOLO



def train_yolo(model_dir: str, yaml_name: str = 'branch_segmentation.yaml',model_name: str = "yolov11n-seg.pt", epochs : int = 10) -> None:
    """
    Trains the YOLO model on the dataset.

    Args:
        model_dir (str): Path to the directory containing the YOLO model configuration file.
        epochs (int): Number of epochs to train the model.
    """

    model_path = os.path.join(model_dir, model_name)
    assert os.path.exists(model_path), f"Model file {model_path} not found."

    model = YOLO(model_path)

    assert os.path.exists(yaml_name), f"YAML file {yaml_name} not found."

    model.train(data = yaml_name, 
                epochs = epochs, 
                imgsz = 640) # keep a lot of the default params rn
    
