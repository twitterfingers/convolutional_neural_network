import cv2
from cv2 import cv2
import numpy as np
import tensorflow as tf
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

SPORT = ["Successful", "UnSuccessful"]

IMG_SIZE = 70
def predict_image(location , model_location):
       frame = cv2.imread(location)
       #Convert to Grauscale
       frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       frame = cv2.resize(frame, (70, 70)).astype("float32")
       frame = np.array(frame).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
       
       model = tf.keras.models.load_model(model_location)
       predictions = model.predict(location)
       
       final_result = SPORT[int(predictions[0][0])]

       return final_result



