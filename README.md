Tennis Technique AI Model

Used YOLOv7 model from https://github.com/WongKinYiu/yolov7/tree/main

Description:
The aim of this project is to help people take action and improve their tennis technique on their own without the restriction of coaching availability. In this project, I built 2 AI models, one to determine whether or not a swing is "good" or "bad", and the other to predict an optimal swing path for the player and to lay over the predicted skeleton onto the video of the player. All the user had to do to run these models are to call the python functions that run the AI program, the only thing the user needs to pass in is the path to their video file, all other arguments will be default. Videos must have a specific requirement: they must be from a side angle and cut such that there is only 1 stroke in the video. Right now, this model only works for right-handed forehands, I will work on expanding this to other strokes once I collect more data. Also a disclaimer, this model is not 100% accurate, it only has about an 80% accuracy when classifying swings as "good" or "bad" and is not the most accurate for the predicted swing overlay. This project is still in development, I'm working on improving the models in the near future, hopefully by gathering more data and tweaking the models a bit.

How to run it:

1. Clone this repo
2. Download the YOLOv7 model from https://github.com/WongKinYiu/yolov7/tree/main, preferably in this folder
3. Follow the instructions below for using both models

AI Swing Classification Model:

1. 

AI Swing Overlay Model:

1. 