def read_keypoints_from_csv(csv_file):
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

    return keypoints, frames

import numpy as np

def calculate_similarity(csv1, csv2, offset):
    keypoints1, frames1 = read_keypoints_from_csv(csv1)
    keypoints2, frames2 = read_keypoints_from_csv(csv2)

    if len(frames1) != len(frames2):
        raise ValueError("The number of frames in the two CSV files must be the same.")

    total_distance = 0
    num_points = 0

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
    similarity_score = 1 / (1 + average_distance)  # A simple way to convert distance to similarity

    return similarity_score

def minimize_difference(csv1, csv2):
    similarity = []
    for i in range(301):
        similarity.append(calculate_similarity(csv1, csv2, i-150))
    min_dif = max(similarity)
    shift = similarity.index(min_dif) - 150
    print(f'Similarity score: {min_dif}')
    print(f'Frame shift: {shift}')
    return shift

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
    similarity_score = 1 / (1 + average_distance)  # A simple way to convert distance to similarity

    return similarity_score

def get_training(model_videos, good, bad):
    training_videos = []
    
    def process_videos(video_list, label):
        for i in video_list:
            averaged_similarity = []
            all_similarity = []
            for vid in model_videos:
                similarity = []
                shift = minimize_difference(f'keypoints {vid}.csv', f'keypoints {i}.csv')
                for kp in range(17):
                    similarity.append(calculate_keypoint_similarity(f'keypoints {vid}.csv', f'keypoints {i}.csv', shift, kp))
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

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assume 'training_data' is defined and loaded
data = np.array(training_data)

X = data[:, :-1]  # First 17 columns: similarity scores
y = data[:, -1]   # Last column: labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Assign weights to each keypoint
keypoint_weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 6.0, 2.0, 8.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], dtype=np.float32)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(17,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Custom loss function with keypoint weights
def weighted_binary_crossentropy(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weighted_loss = loss * tf.reduce_sum(tf.cast(keypoint_weights, tf.float32)) / tf.reduce_sum(tf.cast(keypoint_weights, tf.float32))
    return weighted_loss

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)


def get_test_data(model_videos, test):
    test_data = []
    
    for i in test:
        averaged_similarity = []
        all_similarity = []
        for vid in model_videos:
            similarity = []
            shift = minimize_difference(f'keypoints {vid}.csv', f'keypoints {i}.csv')
            for kp in range(17):
                similarity.append(calculate_keypoint_similarity(f'keypoints {vid}.csv', f'keypoints {i}.csv', shift, kp))
            all_similarity.append(similarity)

        for kp in range(17):
            total_similarity = 0
            for j in range(len(all_similarity)):
                total_similarity += all_similarity[j][kp]
            averaged_similarity.append(total_similarity / len(all_similarity))
        averaged_similarity.append(label)
        test_data.append(averaged_similarity)

    return test_data

