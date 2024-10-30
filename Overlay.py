# Module imports
import cv2
import numpy as np
import os
import urllib.request
import sys
import torch
import time
import datetime
import csv
import random
import tensorflow as tf

# Function imports
from get_data import read_keypoints_from_csv, yoloV7_pose_video
from setup import running_inference
from classification import calculate_similarity, minimize_difference
from torchvision import transforms
from PIL import Image
from moviepy.editor import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

# Defines good and bad videos
me = [75, 81, 82, 83, 84, 85, 86, 87, 88]

bad = [50, 53, 54, 65, 66, 68, 69, 76, 77, 78, 79, 80, 90, 98, 105, 106, 108, 111, 127]
for i in range(146, 203):
    bad.append(i)
    
good = [i for i in range(10, 134) if i not in bad and i not in me]


# Gets the min and max x and y coordinates of a video, stored in CSV files
def get_min_max(csv_file):
    max_x, min_x, max_y, min_y = 0, 0, 0, 0
    with open(csv_file, 'r') as file:
        csvreader = csv.reader(file)
        next(csvreader)  # Skip header
        line = next(csvreader)
        max_x, min_x, max_y, min_y = float(line[4]), float(line[5]), float(line[6]), float(line[7])
    return max_x, min_x, max_y, min_y

# Gets the number of frames in the video
def get_vid_length(videofile):
    start = time.time()
    # Reading video
    video = VideoFileClip(videofile)

    # Stats
    duration = video.duration
    fps = round(video.fps)
    nbframes = round(duration * fps)

    return nbframes

# Gets the keypoints
def draw_model_keypoints(image, model_kp, confidence=0.25, threshold=0.65):
    """
    Draw YoloV7 pose keypoints
    """

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = cv2.cvtColor(nimg.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Draw keypoints for the most centered figure
    xcors, ycors = plot_model_skeleton_kpts(nimg, model_kp, 3)
    cors = list(zip(xcors, ycors))  # Corrected: Initialize 'cors' here

    return nimg, cors

# Adapted from YOLOv7 model, modified to plot the skeleton keypoints and connects them
def plot_model_skeleton_kpts(im, kpts, steps, orig_shape=None):
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
    num_kpts = len(kpts)
    xcors = []
    ycors = []

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[kid][0], kpts[kid][1]
        xcors.append(int(x_coord))
        ycors.append(int(y_coord))
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[sk[0] - 1][0]), int(kpts[sk[0] - 1][1]))
        pos2 = (int(kpts[sk[1] - 1][0]), int(kpts[sk[1] - 1][1]))

        if pos1[0] % 640 == 0 or pos1[1] % 640 == 0 or pos1[0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0] < 0 or pos2[1] < 0:
            continue
        
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
    
    return xcors, ycors

# Adapted from YOLOv7 model function yoloV7_pose_video, modified to overlay the predicted swing onto the original video, given a list of coordinates
def overlay_model(overlay_kp, videofile, indexes, model, confidence=0.25, threshold=0.65):
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = videofile.split('/')[-1].split('.')[0]
    outputvideofile = f"../results/{video_name}_predicted_swing.mp4"
    outvideo = cv2.VideoWriter(outputvideofile, fourcc, 30.0,
                               (int(capture.get(3)), int(capture.get(4))))
    idx = 1
    
    cors = []
    
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
            frame, frame_cor = draw_model_keypoints(frame, overlay_kp[int(indexes[idx-1])])
            frame = cv2.resize(frame,
                               (int(capture.get(3)), int(capture.get(4))))

            cv2.imwrite(
                "results/videoframe_" + os.path.basename(videofile) + '_' +
                str(f"{idx:06}.jpg"), frame)
            outvideo.write(frame)  # output to video file

        else:
            break

        idx += 1

    processed_time = round(time.time() - start)
    time_per_frame = round(processed_time / (idx - 1), 2)
    print("\nDone in", processed_time, "seconds")
    print("Time per frame =", time_per_frame, "seconds")
    print("\nSaved video:", outputvideofile)

    capture.release()
    outvideo.release()

    return outputvideofile

# Difference between keypoints in the 2 videos
def video_data(good, bad, offset):
    good_vid_kp, frames = read_keypoints_from_csv(f'../Data/keypoints {good}.csv')
    bad_vid_kp, frames = read_keypoints_from_csv(bad)
    all_kp = []
    if offset < 0:
        offset = abs(offset)
        for j in range(len(frames) - offset - 1):  # For each frame
            keypoints = []
            for kp in range(17):
                x, y = good_vid_kp[kp][j][0] - bad_vid_kp[kp][j+offset][0], good_vid_kp[kp][j][1] - bad_vid_kp[kp][j+offset][1]
                keypoints.extend([x, y])
            all_kp.append(keypoints)

    else:
        for j in range(len(frames) - offset - 1):  # For each frame
            keypoints = []
            for kp in range(17):
                x, y = good_vid_kp[kp][j+offset][0] - bad_vid_kp[kp][j][0], good_vid_kp[kp][j+offset][1] - bad_vid_kp[kp][j][1]
                keypoints.extend([x, y])   
            all_kp.append(keypoints)
    
    return all_kp

# Reads the keypoints from the csv file, similar to read_keypoints_from_csv function but also takes into account an offset (I used a function to find the optimal frame offset to maximise similarities between the uploaded video and a randomized bad swing that the video will be compared to)
def get_keypoints_from_csv(csv_file, offset):
    keypoints = [[] for _ in range(17)]  # Assuming 17 keypoints
    frames = []

    # Read keypoints from CSV file
    with open(csv_file, 'r') as file:
        csvreader = csv.reader(file)
        next(csvreader)  # Skip header
        for row in csvreader:
            frame = int(float(row[0]))  # Convert to float first, then to int
            keypoint = int(float(row[1]))  # Convert to float first, then to int
            x = float(row[2])
            y = float(row[3])
            keypoints[keypoint].append((x, y))
            if keypoint == 0:
                frames.append(frame)
       
    adjusted_kp=[]
    if offset<0:
        total=800-abs(offset)-1
        for i in keypoints:
            adjusted_kp.append(i[:total])
    else:
        start=abs(offset)+1
        for i in keypoints:
            adjusted_kp.append(i[start:])

    return adjusted_kp, frames

# AI model that predicts the output of an "ideal" swing
def model_good_swing(good_video, bad_video, random_videos, epochs_run):
    # - ==> 2nd vid ahead
    # + ==> 1st vid ahead
    offset = minimize_difference(f'../Data/keypoints {good_video}.csv', bad_video)
    
    random_good_videos = []
    random_bad_videos = []
    predictions = []
    
    for i in range(random_videos):
        random_good_videos.append(good[int(random.random()*len(good))])
        random_bad_videos.append(bad[int(random.random()*len(bad))])
        
        
    random_good = []
    random_bad = []
    
    for i in random_good_videos:
        kp, frames = get_keypoints_from_csv(f'../Data/keypoints {i}.csv', offset)
        random_good.append(kp)
        
    for i in random_bad_videos:
        kp, frames = get_keypoints_from_csv(f'../Data/keypoints {i}.csv', offset)
        random_bad.append(kp)
        
    good_test, frames = get_keypoints_from_csv(f'../Data/keypoints {good_video}.csv', offset)
    bad_test, frames = get_keypoints_from_csv(bad_video, offset)
        
    random_differences = []

    test_differences = video_data(good_video, bad_video, offset)
    
    for i in range(random_videos):
        random_differences.append(video_data(random_good_videos[i], f'../Data/keypoints {random_bad_videos[i]}.csv', offset))

    # Convert to numpy arrays
    X_train = np.array([random_bad][0])
    y_train = np.array([random_differences])
    X_test = np.array([bad_test])
    y_test = np.array([test_differences])

    # Standardize the data
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    num_tests, num_features, num_frames, num_videos = X_train.shape
    
    num_features*=2

    X_train = X_train.reshape(-1, num_features)  # Flatten for scaling
    y_train = y_train.reshape(-1, num_features)
    X_test = X_test.reshape(-1, num_features)

    X_train = scaler1.fit_transform(X_train)
    y_train = scaler2.fit_transform(y_train)
    X_test = scaler1.transform(X_test)

    X_train = X_train.reshape(num_tests, num_frames, num_features)
    y_train = y_train.reshape(num_tests, num_frames, num_features)
    X_test = X_test.reshape(num_tests//20, num_frames, num_features)

    # Define the model
    model = Sequential([
        LSTM(128, activation='tanh', input_shape=(num_frames, 34), return_sequences=True),
        Dense(256, activation='relu'),
        Dense(64, activation='relu'),
        TimeDistributed(Dense(34, activation='tanh'))  # Output layer
    ])
    
    # Compile the model with the MSE loss function
    model.compile(optimizer=Adam(learning_rate=5e-4), loss='mse', metrics=['mae'])

    # Learning rate scheduler
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    # Train the model with teacher forcing
    history = model.fit(X_train, y_train, epochs=epochs_run, batch_size=1, verbose=1, callbacks=[reduce_lr, early_stopping])

    # Save the model and make predictions
    model_predictions = model.predict(X_test)
    
    model_predictions = scaler2.inverse_transform(model_predictions.reshape(-1, num_features)).reshape(1, num_frames, num_features)
    
    rms_error_x = []
    rms_error_y = []
    
    rms_error_x1 = []
    rms_error_y1 = []
    
    for j in range(num_frames):
        keypoint_predictions = []
        if j != 0:
            for i in range(0, 34, 2):
                x, y = bad_kp[i//2][j][0]+model_predictions[0][j][i], bad_kp[i//2][j][1]+model_predictions[0][j][i+1]

                keypoint_predictions.append((x, y))

            predictions.append(keypoint_predictions)
            
            
        else:
            bad_kp, frames = read_keypoints_from_csv(f'../Data/{bad_video}')
            for i in range(17):
                x = bad_kp[i][j][0]
                y = bad_kp[i][j][1]
                keypoint_predictions.append((x, y))
            predictions.append(keypoint_predictions)

    return predictions

# Function that calls other functions to return output video of the swing
def overlay_swing_path(videopath, pose_model):
    video_csv = yoloV7_pose_video(videopath, pose_model)
    predictions = model_good_swing(good[int(random.random()*len(good))], video_csv, 30, 30)
    prediction_indexes = np.linspace(0, len(predictions)-1, get_vid_length(videopath))
    max_x, min_x, max_y, min_y = get_min_max(f'../Tests/{video_csv}')
    expanded_predictions = []

    for i in predictions:
        frame_kp = []
        for j in i:
            x = j[0] * (max_x-min_x) + min_x
            y = j[1] * (max_y-min_y) + min_y
            frame_kp.append((x, y))
        expanded_predictions.append(frame_kp)

    return overlay_model(expanded_predictions, videopath, prediction_indexes, pose_model)