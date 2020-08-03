import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
from cv2 import cv2
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

DATADIR = "/content/drive/My Drive/Big Data Final/"
SPORT = ["Successful", "UnSuccessful"]

IMG_SIZE = 70

training_data = []


def Data():
    for category in SPORT:
        location = os.path.join(DATADIR, category)
        class_value = SPORT.index(category)
        for images in os.listdir(location):
         
            frame = cv2.imread(os.path.join(location, images))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (70, 70)).astype("float32")
            # frame = np.array(frame).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            training_data.append([frame, class_value])
           


Data()
random.shuffle(training_data)

X = []  # Feature Set (train x)
y = []  # Labels (test y)

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 70, 70, 1)

#Save Data
pickle_out = open("/content/drive/My Drive/Big Data Final/X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("/content/drive/My Drive/Big Data Final/y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
