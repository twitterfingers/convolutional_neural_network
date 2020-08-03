import cv2
from cv2 import cv2
import numpy as np
import os
import math
import tensorflow as tf
from Predict_Image import predict_image

cap = cv2.VideoCapture("Video Location")
frameRate = cap.get(5) # How many frame per each video

model = tf.keras.models.load_model("Model Location")
ls = list()

while(cap.isOpened()):
    frameId = frameRate
    ret, frame = cap.read()
    
    if(ret != True):
         break

    copyframe = frame.copy()
    image_label = predict_image(frame, model)

    if image_label == "Successful":
        print("Pass Label to OpenPose")
    else:
        continue

cap.release()

