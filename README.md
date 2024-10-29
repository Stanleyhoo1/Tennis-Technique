
# Tennis Technique AI Model

**Model Used:** YOLOv7 ([link to YOLOv7 repository](https://github.com/WongKinYiu/yolov7/tree/main))

## Project Description

This project aims to empower tennis players to improve their technique independently, without relying solely on coaching availability. I developed two AI models for this purpose:

1. **Swing Classification Model**: This model classifies a tennis swing as "good" or "bad" based on certain technical aspects.
2. **Swing Overlay Model**: This model predicts an optimal swing path for the player and overlays a skeleton representation of this path onto the player’s video.

Users only need to provide the path to their video file, with all other parameters set to default. The video input should meet the following requirements:
- **Angle**: Side view
- **Content**: Single stroke per video (e.g., one forehand swing)
  
Currently, this model supports only **right-handed forehand swings**. I plan to expand it to other strokes as I gather more data. Please note that this project is still in development, with the swing classification model achieving approximately **80% accuracy**. The overlay model is functional but requires further refinement for optimal accuracy.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone [YOUR REPOSITORY URL]
   cd [YOUR PROJECT FOLDER]
   ```

2. **Download the YOLOv7 Model**:
   - Download the YOLOv7 model weights and place them in the project directory. Follow the instructions from the YOLOv7 [GitHub repository](https://github.com/WongKinYiu/yolov7/tree/main) to download the model.

3. **Run the Models**:
   Follow the instructions below for using both models.

---

## AI Swing Classification Model

1. **Function Call**:
   - Run the classification model by calling the `classify_swing` function (replace this with your function name if different).
   - **Parameters**: Pass the path of the video file as an argument.
   - Example:
     ```python
     classify_swing("path/to/your/video.mp4")
     ```

2. **Output**:
   - The model will output whether the swing is classified as "good" or "bad" with an 80% accuracy level.

## AI Swing Overlay Model

1. **Function Call**:
   - Run the overlay model by calling the `overlay_swing_path` function (replace this with your function name if different).
   - **Parameters**: Pass the path of the video file as an argument.
   - Example:
     ```python
     overlay_swing_path("path/to/your/video.mp4")
     ```

2. **Output**:
   - The model will display the original video with the predicted swing path overlayed on the player, showing the suggested optimal swing.

---

## Future Improvements
I am working to improve both models by gathering more data and fine-tuning model parameters. Upcoming features include:
- Support for left-handed players
- Additional stroke types (backhand, serve, etc.)
- Improved classification and overlay accuracy

---

This project is intended for players looking to refine their technique with minimal input from external coaches. **Note**: While the models provide guidance, they are not a substitute for professional coaching and should be used as supplementary tools for improvement.
