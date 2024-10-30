import cv2
import matplotlib.pyplot as plt
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
from Setup import running_inference
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

# Contains the code to run to process a videofile and store the data in a CSV file

# Adapted from draw_keypoints function in YOLOv7 model, modified to only track the center figure, in this usage the tennis player
def draw_keypoints(output, image, model, confidence=0.25, threshold=0.65):
    """
    Draw YoloV7 pose keypoints
    """
    output = non_max_suppression_kpt(
        output,
        confidence,  # Confidence Threshold
        threshold,  # IoU Threshold
        nc=model.yaml['nc'],  # Number of Classes
        nkpt=model.yaml['nkpt'],  # Number of Keypoints
        kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = cv2.cvtColor(nimg.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    cors = []
        
    areas = []
        
    for idx in range(output.shape[0]):
        kpts = output[idx, 7:].T
        steps = 3
        num_kpts = len(kpts) // steps
    
        xcors = []
        ycors = []

        for kid in range(num_kpts):
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            xcors.append(int(x_coord))
            ycors.append(int(y_coord))
        max_x = np.max(xcors)
        max_y = np.max(ycors)
        min_x = np.min(xcors)
        min_y = np.min(ycors)
        areas.append((max_x-min_x) * (max_y-min_y))

    # Determine the image center
    img_center_x = nimg.shape[1] / 2
    img_center_y = nimg.shape[0] / 2

    largest_idx = np.argmax(areas)

    # Draw keypoints for the most centered figure
    xcors, ycors = plot_skeleton_kpts(nimg, output[largest_idx, 7:].T, 3)
    for xcor, ycor in zip(xcors, ycors):
        cors.append((xcor, ycor))

    return nimg, cors

# Adapted from the YOLOv7 model, modified to return x and y coordinates of the keypoints
def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    # Plot the skeleton and keypoints for coco dataset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps
    
    # Inititalize x and y cors list
    xcors = []
    ycors = []

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        # Append x and y coordinates to the list
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        xcors.append(int(x_coord))
        ycors.append(int(y_coord))
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
    
    return xcors, ycors

# Reads keypoints from the saved csv files
def read_keypoints_from_csv(csv_file):
    keypoints = [[] for _ in range(17)]  # 17 keypoints
    frames = []

    # Opens CSV file
    with open(csv_file, 'r') as file:
        csvreader = csv.reader(file)
        next(csvreader)  # Skip header
        for row in csvreader:
            frame = int(float(row[0]))  # Convert to float first, then to int
            keypoint = int(float(row[1]))  # Convert to float first, then to int
            x = float(row[2])
            y = float(row[3])
            keypoints[keypoint].append((x, y)) # (x, y) coordinates of the keypoint
            # New frame
            if keypoint == 0:
                frames.append(frame)

    return keypoints, frames

# Normalizes the x and y coordinates on a 0 to 1 scale, scaled to that specific video
def normalize_csv(csvfile):
    keypoints, frames = read_keypoints_from_csv(csvfile)
    
    # Flatten all x and y coordinates to find global min and max
    all_x = [kp[0] for kps in keypoints for kp in kps]
    all_y = [kp[1] for kps in keypoints for kp in kps]
    
    # Min/max for x and y, used to normalize everything on a 0-1 scale
    max_x, min_x = max(all_x), min(all_x)
    max_y, min_y = max(all_y), min(all_y)
    
    # Saves normalized data in the same csv file, replaces old data
    with open(csvfile, 'w', newline='') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(['Frame', 'Keypoint', 'X', 'Y', 'Max X', 'Min X', 'Max Y', 'Min Y'])  # Write header
        
        for i in range(17):
            for j in range(len(frames)):
                normalized_x = (keypoints[i][j][0] - min_x) / (max_x - min_x) if max_x != min_x else 0.0
                normalized_y = (keypoints[i][j][1] - min_y) / (max_y - min_y) if max_y != min_y else 0.0
                csvwriter.writerow([frames[j], i, normalized_x, normalized_y, max_x, min_x, max_y, min_y])

# Interpolation function
def distribute(lst, size):
    if size <= 0:
        return []
    
    new_lst = ['' for _ in range(size)]
    indexes = np.linspace(0, size - 1, len(lst))
    for i in range(len(lst)):
        new_lst[int(indexes[i])] = lst[i]
    
    # Interpolate the values
    for i in range(len(indexes) - 1):
        start_idx = int(indexes[i])
        end_idx = int(indexes[i + 1])
        interpolated_values = np.linspace(new_lst[start_idx], new_lst[end_idx], end_idx - start_idx + 1)
        for j in range(start_idx, end_idx + 1):
            new_lst[j] = interpolated_values[j - start_idx]
    
    return new_lst

# Expands the video to 800 frames using interpolation
def expand_video(csvfile):
    keypoints, frames = read_keypoints_from_csv(csvfile)
    with open(csvfile, 'w', newline='') as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(['Frame', 'Keypoint', 'X', 'Y'])  # Write header
        
        for i in range(17):
            xcors = [keypoints[i][j][0] for j in range(len(frames))]
            ycors = [keypoints[i][j][1] for j in range(len(frames))]
            new_xcors = distribute(xcors, 800)
            new_ycors = distribute(ycors, 800)
            for j in range(800):
                csvwriter.writerow([j, i, new_xcors[j], new_ycors[j]])

# Adapted from YOLOv7 model, modified to save normalized and interpolated data in CSV files, and return the CSV file
def yoloV7_pose_video(videofile, model, confidence=0.25, threshold=0.65):
    """
    Processing the video using YoloV7
    """
    start = time.time()
    # Reading video
    video = VideoFileClip(videofile)

    # Stats
    duration = video.duration
    fps = round(video.fps)
    nbframes = round(duration * fps)

    print("Processing video:", videofile, "using confidence min =", confidence,
          "and threshold =", threshold)
    print("\nVideo duration =", duration, "seconds")
    print("FPS =", fps)
    print("Total number of frames =", nbframes, "\n")

    # Capture the results frames into a video
    capture = cv2.VideoCapture(videofile)

    idx = 1
    
    cors = []
    
    videoname = videofile.split('/')[-1].split('.')[0]

    csv_file = f'../Tests/{videoname}.csv'
    
    if os.path.exists(csv_file):
        return csv_file
    
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame', 'Keypoint', 'X', 'Y'])  # Write header
        while capture.isOpened():
            (ret, frame) = capture.read()

            if ret == True:
                if idx % fps == 1:
                    nbremainframes = nbframes - idx
                    pctdone = round(idx / nbframes * 100)
                    print("Processed frames =", f"{idx:06}",
                          "| Number of remaining frames:", f"{nbremainframes:06}",
                          "| Done:", pctdone, "%")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output, frame = running_inference(frame, model)
                frame, frame_cor = draw_keypoints(output, frame, model, confidence, threshold)

                for i, (x, y) in enumerate(frame_cor):
                    csvwriter.writerow([idx, i, x, y])
                    
                cors.append(frame_cor)

            else:
                break

            idx += 1
            
    expand_video(csv_file)
    normalize_csv(csv_file)

    processed_time = round(time.time() - start)
    time_per_frame = round(processed_time / (idx - 1), 2)
    print("\nDone in", processed_time, "seconds")
    print("Time per frame =", time_per_frame, "seconds")

    return csv_file