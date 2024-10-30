# Module imports
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
import random
import tensorflow as tf

# Function imports
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from torchvision import transforms
from PIL import Image
from moviepy.editor import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Get_Data import read_keypoints_from_csv, yoloV7_pose_video

# Defines the videos (while gathering data, it was done a little out of order)
model_videos = [19, 38, 48, 72, 73]

me = [75, 81, 82, 83, 84, 85, 86, 87, 88]

bad = [50, 53, 54, 65, 66, 68, 69, 76, 77, 78, 79, 80, 90, 98, 105, 106, 108, 111, 127]
for i in range(146, 203):
    bad.append(i)
    
good = [i for i in range(10, 134) if i not in bad and i not in me]

# Different function to read data from CSV file, returns different output
def get_data_from_csv(csv_file):
    data = {}
    with open(csv_file, 'r') as file:
        csvreader = csv.reader(file)
        next(csvreader)  # Skip header
        for row in csvreader:
            video = int(float(row[0]))  # Convert to float first, then to int
            keypoint = int(float(row[1]))  # Convert to float first, then to int
            similarity = float(row[2])  # Convert to float
            
            # If the video is not in the dictionary, add it with an empty list
            if video not in data:
                data[video] = []
            
            # Append the similarity score to the list for this video
            data[video].append(similarity)

    return data

# Calculautes a similarity score between 2 videos given a frame offset, takes all keypoints into consideration
def calculate_similarity(csv1, csv2, offset):
    keypoints1, frames1 = read_keypoints_from_csv(csv1)
    keypoints2, frames2 = read_keypoints_from_csv(csv2)

    # Number of frames must be equal for comparison
    if len(frames1) != len(frames2):
        raise ValueError("The number of frames in the two CSV files must be the same.")

    total_distance = 0
    num_points = 0

    # Offset < 0 means that video 2 is ahead, else video 1 is ahead
    if offset < 0:
        offset = abs(offset)
        for i in range(17):  # For each keypoint
            for j in range(len(frames1) - offset):  # For each frame
                x1, y1 = keypoints1[i][j][0], keypoints1[i][j][1]
                x2, y2 = keypoints2[i][j+offset][0], keypoints2[i][j+offset][1]
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                total_distance += distance
                num_points += 1
    else:
        for i in range(17):  # For each keypoint
            for j in range(len(frames1) - offset):  # For each frame
                x1, y1 = keypoints1[i][j+offset][0], keypoints1[i][j+offset][1]
                x2, y2 = keypoints2[i][j][0], keypoints2[i][j][1]
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                total_distance += distance
                num_points += 1

    average_distance = total_distance / num_points if num_points != 0 else float('inf')
    similarity_score = 1 / (1 + average_distance)  # A simple way to convert average distance to similarity

    return similarity_score

# Returns the optimal frame shift to maximize similarity scores between 2 videofiles
def minimize_difference(csv1, csv2):
    similarity = []
    for i in range(301):
        similarity.append(calculate_similarity(csv1, csv2, i-150))
    min_dif = max(similarity)
    shift = similarity.index(min_dif) - 150
#     print(f'Similarity score: {min_dif}')
#     print(f'Frame shift: {shift}')
    return shift

# Calculautes a similarity score between 2 videos given a frame offset, takes only 1 keypoint into consideration
def calculate_keypoint_similarity(csv1, csv2, offset, kp):
    keypoints1, frames1 = read_keypoints_from_csv(csv1)
    keypoints2, frames2 = read_keypoints_from_csv(csv2)

    if len(frames1) != len(frames2):
        raise ValueError("The number of frames in the two CSV files must be the same.")

    total_distance = 0
    num_points = 0

    if offset < 0:
        offset = abs(offset)
        for j in range(len(frames1) - offset):  # For each frame
            x1, y1 = keypoints1[kp][j][0], keypoints1[kp][j][1]
            x2, y2 = keypoints2[kp][j+offset][0], keypoints2[kp][j+offset][1]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            total_distance += distance
            num_points += 1
    else:
        for j in range(len(frames1) - offset):  # For each frame
            x1, y1 = keypoints1[kp][j+offset][0], keypoints1[kp][j+offset][1]
            x2, y2 = keypoints2[kp][j][0], keypoints2[kp][j][1]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            total_distance += distance
            num_points += 1

    average_distance = total_distance / num_points if num_points != 0 else float('inf')
    similarity_score = 1 / (1 + average_distance)  # A simple way to convert average distance to similarity

    return similarity_score

# Gets the training data, not used in the final version
def get_training(model_videos, good, bad):
    training_videos = []
    
    # Gets the similarity scores and appends a good or bad label to them
    def process_videos(video_list, label):
        for i in video_list:
            averaged_similarity = []
            all_similarity = []
            for vid in model_videos:
                similarity = []
                shift = minimize_difference(f'../Data/keypoints {vid}.csv', f'../Data/keypoints {i}.csv')
                for kp in range(17):
                    similarity.append(calculate_keypoint_similarity(f'../Data/keypoints {vid}.csv', f'../Data/keypoints {i}.csv', shift, kp))
                all_similarity.append(similarity)
            
            for kp in range(17):
                total_similarity = 0
                for j in range(len(all_similarity)):
                    total_similarity += all_similarity[j][kp]
                averaged_similarity.append(total_similarity / len(all_similarity))
            averaged_similarity.append(label)
            training_videos.append(averaged_similarity)

    # Process good videos
    process_videos(good, 1)

    # Process bad videos
    process_videos(bad, 0)

    return training_videos

# AI model to predict swing (Original, uses a sequential model and similarity scores accross keypoints)
# def predict_swing(videos):

#     saved_data = get_data_from_csv('data.csv')
    
#     saved_training_data = []
#     for i in good:
#         changed = saved_data[i].copy()
#         changed.append(1)
#         saved_training_data.append(changed)
#     for i in bad:
#         changed = saved_data[i].copy()
#         changed.append(0)
#         saved_training_data.append(changed)
    
#     data = np.array(saved_training_data)

#     X = data[:, :-1]  # First 17 columns: similarity scores
#     y = data[:, -1]   # Last column: labels

#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Standardize the data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # Define the model
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(17,)),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])

#     # Compile the model with the custom loss function
#     model.compile(optimizer=Adam(learning_rate=5e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
#     # Learning rate scheduler
#     reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    
#     # Early stopping
#     early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

#     # Train the model
#     model.fit(X_train, y_train, epochs=100, batch_size=2, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])
    
#     test = get_test_data(model_videos, videos)
    
#     # Standardize the test data
#     test_data = np.array(test)
#     test_data = scaler.transform(test)

#     # Make predictions
#     predictions = model.predict(test_data)
#     predicted_labels = (predictions > 0.7).astype(int)
#     for i in range(len(predicted_labels)):
#         print(f'Predicted swing quality: {"Bad" if predicted_labels[i] == 0 else "Good"}')

#     print() 

#     for i in range(len(predictions)):
#         print(f'Video {videos[i]}: {predictions[i]}')
        
#     return predicted_labels

# Gets the test similarity data for the uploaded video, for the original model
# def get_test_data(model_videos, test):
#     test_data = []
    
#     for i in test:
#         averaged_similarity = []
#         all_similarity = []
#         for vid in model_videos:
#             similarity = []
#             shift = minimize_difference(f'keypoints {vid}.csv', f'keypoints {i}.csv')
#             for kp in range(17):
#                 similarity.append(calculate_keypoint_similarity(f'keypoints {vid}.csv', f'keypoints {i}.csv', shift, kp))
#             all_similarity.append(similarity)

#         for kp in range(17):
#             total_similarity = 0
#             for j in range(len(all_similarity)):
#                 total_similarity += all_similarity[j][kp]
#             averaged_similarity.append(total_similarity / len(all_similarity))
#         averaged_similarity.append(label)
#         test_data.append(averaged_similarity)

#     return test_data

# Function to set a seed to get rid of randomness
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# New AI model, uses a Sequential model as well but uses the actual coordinates of the keypoints as inputs
def classify_swing(videos, pose_model):
    set_seeds(42)
    
    keypoints_data = []
    labels = []
    min_max_values = {}
    
    # Extra data I added later
    extra_good = [i for i in range(34, 51)]
    extra_good.append(1)
    extra_good.append(2)
    for i in range(5, 10):
        extra_good.append(i)
        
    extra_bad = [i for i in range(13, 34)]
    extra_bad.append(3)
    extra_bad.append(4)
    extra_bad.append(10)
    extra_bad.append(11)
    
    # Process good videos
    for i in good:
        csv_file = f'../Data/keypoints {i}.csv'
        keypoints, frames = read_keypoints_from_csv(csv_file)
        keypoints = np.array(keypoints).reshape(17, -1)  # Shape: (17, num_frames*2)
        keypoints_data.append(keypoints.flatten())
        labels.append(1)
        
    for i in extra_good:
        csv_file = f'../Data/Test - {i}.csv'
        keypoints, frames = read_keypoints_from_csv(csv_file)
        keypoints = np.array(keypoints).reshape(17, -1)
        keypoints_data.append(keypoints.flatten())
        labels.append(1)
    
    # Process bad videos
    for i in bad:
        csv_file = f'../Data/keypoints {i}.csv'
        keypoints, frames = read_keypoints_from_csv(csv_file)
        keypoints = np.array(keypoints).reshape(17, -1)
        keypoints_data.append(keypoints.flatten())
        labels.append(0)
        
    for i in extra_bad:
        csv_file = f'../Data/Test - {i}.csv'
        keypoints, frames = read_keypoints_from_csv(csv_file)
        keypoints = np.array(keypoints).reshape(17, -1)
        keypoints_data.append(keypoints.flatten())
        labels.append(0)
    
    # Convert to numpy arrays
    X = np.array(keypoints_data)
    y = np.array(labels)
    
    # Stratified splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=8,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, reduce_lr, early_stopping],
        shuffle=True
    )
    
    # Load the best model
    model.load_weights('best_model.keras')
    
    # Prepare test data
    test_keypoints_data = []
    test_min_max_values = {}
    for video_file in videos:
        test_csv_file = yoloV7_pose_video(video_file, pose_model)
        keypoints, frames = read_keypoints_from_csv(test_csv_file)
        keypoints = np.array(keypoints).reshape(17, -1)
        test_keypoints_data.append(keypoints.flatten())
    
    # Convert to numpy array and standardize
    X_new = np.array(test_keypoints_data)
    X_new = scaler.transform(X_new)
    
    # Predict on new data
    predictions = model.predict(X_new)
    predicted_labels = (predictions > 0.5).astype(int)
    
    vid = 0
    
    for i, label in enumerate(predicted_labels):
        print(f'Predicted swing quality for video {videos[i]}: {"Good" if label == 1 else "Bad"}')
        print(f'Confidence score: {predictions[i][0]}')
        print()
        vid += 1
