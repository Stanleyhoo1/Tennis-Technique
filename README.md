
# Tennis Technique AI Model

**Model Used to Extract Data:** YOLOv7 ([link to YOLOv7 repository](https://github.com/WongKinYiu/yolov7/tree/main))

## Project Description

This project aims to empower tennis players to improve their technique independently, without relying solely on coaching availability. I developed two AI models for this purpose:

1. **Swing Classification Model**: This model classifies a tennis swing as "good" or "bad" based on certain technical aspects.
2. **Swing Overlay Model**: This model predicts an optimal swing path for the player and overlays a skeleton representation of this path onto the player’s video.

Users only need to provide the path to their video file, with all other parameters set to default. The video input should meet the following requirements:
- **Angle**: Side view
- **Content**: Single stroke per video (e.g., one forehand swing)

This is an example of a good video file:
![](https://github.com/Stanleyhoo1/Tennis-Technique/blob/main/Example%20Video.gif)
  
Currently, this model supports only **right-handed forehand swings**. I plan to expand it to other strokes as I gather more data. Please note that this project is still in development, with the swing classification model achieving approximately **80% accuracy**. The overlay model is functional but requires further refinement for optimal accuracy. A demo of both models being used in Jupyter Notebook can be viewed in the Final_Test notebook. The other notebooks were used for testing and graphing data, you can view my process of designing the model through those.

**Note**: These models take quite a while to run depending on video length so please be patient if you don't see any output immediately. If all instructions are followed correctly, these models should work.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone [THIS REPOSITORY URL]
   cd [YOUR PROJECT FOLDER]
   ```

2. **Download the YOLOv7 Model**:
   - Download the YOLOv7 model and place that folder in the Tennis-Technique folder. Follow the instructions from the YOLOv7 [GitHub repository](https://github.com/WongKinYiu/yolov7/tree/main) to download the model (or you can just clone the YOLOv7 repository in the Tennis-Technique folder).
  
  
3. **Setup**:
   - Open Command Prompt or Terminal
   - Navigate to this folder (`cd path/to/Tennis-Technique`)
   - Run the following commands in this exact order:
     - This will install the neccesary packages

       ```python
       pip install -r requirements.txt
       ```
     - This will load the YOLOv7 model we will be using to extract data from the videos and import the modules we will be using
       ```python
       python -i __init__.py
       ```
4. **Run the Models**:
   Follow the instructions below for using both models.

---

## AI Swing Classification Model

1. **Function Call**:
   - Run the classification model by calling the `classify_swing` function.
   - **Parameters**: Pass the paths of the video files you want to classify as an argument in a list format.
   - Example:
     ```python
     classify_swing(["path/to/your/video.mp4", "path/to/your/video.mp4"], model)
     ```

2. **Output**:
   - The model will output whether the swing is classified as "good" or "bad" with an 80% accuracy level.
   - You can test this model on the example video in this repository, you should get a "good" swing

## AI Swing Overlay Model

1. **Function Call**:
   - Run the overlay model by calling the `overlay_swing_path` function.
   - **Parameters**: Pass the path of the video file as an argument.
   - Example:
     ```python
     overlay_swing_path("path/to/your/video.mp4", model)
     ```

2. **Output**:
   - The model will return the path of the output video file, which is the original video with the predicted swing path overlayed on the player, showing the suggested optimal swing. The output video file will be in the results folder that you created in the Tennis-Technique folder
   - You can see an example output video file in the results folder

---

## Future Improvements
I am working to improve both models by gathering more data and fine-tuning model parameters. Upcoming features include:
- Support for left-handed players
- Additional stroke types (backhand, serve, etc.)
- Improved classification and overlay accuracy

---

This project is intended for players looking to refine their technique with minimal input from external coaches. **Note**: While the models provide guidance, they are not a substitute for professional coaching and should be used as supplementary tools for improvement.

---

## Appendix: Neural Network Model Development
I built two main models for this project: a classification model and a swing overlay model.

**Classification Model**: This model classifies each swing as "good" or "bad" based on key technique markers. I used YOLOv7 to pull keypoints from each video frame, then normalized and interpolated these points to make sure everything lined up across different videos. This way, the model could pick out patterns regardless of video length or player position. It uses a seqeuntial binary classification model to predict these classifications, and currently has about an 80% accuracy rate.

**Swing Overlay Model**: This model’s goal is to predict an ideal swing path and overlay it on the video. It’s a time-series regression model using LSTM layers, which looks at differences between "good" and "bad" swings and calculates small adjustments needed to improve each frame. In the final output, you’ll see a skeleton overlay that shows the optimal swing path right alongside the player’s actual swing, so they can see where to make changes.

Both models took a lot of trial and error with normalization, dataset tweaks, and model tuning to get results I was happy with.
