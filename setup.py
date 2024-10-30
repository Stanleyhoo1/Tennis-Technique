import cv2
import numpy as np
import os
import urllib.request
import sys
import torch
import time
import datetime
import csv
from torchvision import transforms
from PIL import Image
from moviepy.editor import *

# https://github.com/WongKinYiu/yolov7/tree/main, accessed on January 7th, 2024, YOLOv7 model for human pose estimation
# Slightly modified to fit needs, only included 1 model and had to modify functions so I could load the pose estimation model through a function
        
def get_yolov7_model(model):
        """
        Download YoloV7 model from a yoloV7 model list
        """
        modelid = model

        if not os.path.exists(modelid):
            print("Downloading the model:",
                  os.path.basename(modelid), "from:", modelid)
            urllib.request.urlretrieve(modelid,
                                       filename=os.path.basename(modelid))
            print("Done\n")

        if os.path.exists(modelid):
            print("Downloaded model files:")

def loading_yolov7_model(yolomodel, device):
    """
    Loading yolov7 model
    """
    print("Loading model:", yolomodel)
    model = torch.load(yolomodel, map_location=device)['model']
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)

    return model, yolomodel

def image_view(imagefile, w=15, h=10):
    """
    Displaying an image from an image file
    """
    plt.figure(figsize=(w, h))
    plt.axis('off')
    plt.imshow(cv2.cvtColor(cv2.imread(imagefile),
                            cv2.COLOR_BGR2RGB))
    
def running_inference(image, model):
    """
    Running yolov7 model inference
    """
    from utils.datasets import letterbox
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    image = letterbox(image, 960,
                      stride=64,
                      auto=True)[0]  # shape: (567, 960, 3)
    image = transforms.ToTensor()(image)  # torch.Size([3, 567, 960])

    if torch.cuda.is_available():
        image = image.half().to(device)

    image = image.unsqueeze(0)  # torch.Size([1, 3, 567, 960])

    with torch.no_grad():
        output, _ = model(image)

    return output, image

def get_model():
    YOLO_DIR = 'yolov7'

    RESULTS_DIR = 'results'

    YOLOV7_MODEL = ["https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt"]

    user_path = os.getcwd()
    current_folder = os.path.basename(user_path)
    if current_folder != YOLO_DIR:
        os.chdir(YOLO_DIR)
    
    from yolov7.utils.datasets import letterbox
    from yolov7.utils.general import non_max_suppression_kpt
    from yolov7.utils.plots import output_to_keypoint, plot_skeleton_kpts

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    get_yolov7_model(YOLOV7_MODEL[0])

    YOLOV7MODEL = os.path.basename(YOLOV7_MODEL[0])

    try:
        print("Loading the model...")
        model, yolomodel = loading_yolov7_model(yolomodel=YOLOV7MODEL, device=device)
        print("Using the", YOLOV7MODEL, "model")
        print("Done")
        return model

    except:
        print("[Error] Cannot load the model", YOLOV7MODEL)