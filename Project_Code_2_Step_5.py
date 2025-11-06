# Step 5: Predict and visualize 
import json 
import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import datasets, layers, models

from tensorflow.keras.preprocessing import image


model = keras.models.load_model("trained_model_2.h5")

with open("class_labels_2.json", "r") as f:
    class_labels_2 = json.load(f)
print("Class labels Model 2:", class_labels_2)    

#Load Crack, Missing Head and Paint Off jpg images


img_Load_Crack = "C:/Users/Harve/Downloads/Project 2 Data/Data/test/crack/test_crack.jpg"
img_Missing_Head = "C:/Users/Harve/Downloads/Project 2 Data/Data/test/missing-head/test_missinghead.jpg"
img_Paint_Off = "C:/Users/Harve/Downloads/Project 2 Data/Data/test/paint-off/test_paintoff.jpg"

#Set each image to a varaibles with target size (500,500), turn into an array, and expand dimension with 0 axis. 
#Rescale image for faster computation.

img1 = image.load_img(img_Load_Crack, target_size = (500,500))
x = image.img_to_array(img1)
x = np.expand_dims(x, axis = 0)
x /= 255

img2 = image.load_img(img_Missing_Head, target_size = (500,500))
y = image.img_to_array(img2)
y = np.expand_dims(y, axis = 0)
y /= 255

img3 = image.load_img(img_Paint_Off, target_size = (500,500))
z = image.img_to_array(img3)
z = np.expand_dims(z, axis = 0)
z /= 255




preds1 = model.predict(x)[0]
predicted_index = np.argmax(preds1)
predicted1 = class_labels_2[predicted_index]

preds2 = model.predict(y)[0]
predicted_index_2 = np.argmax(preds2)
predicted2 = class_labels_2[predicted_index_2]

preds3 = model.predict(z)[0]
predicted_index_3 = np.argmax(preds3)
predicted3 = class_labels_2[predicted_index_3]

#Shows images in plots category
py.imshow(img1)
py.title(f"Predicted: {predicted1}")
py.axis("off")
py.show()

py.imshow(img2)
py.title(f"Predicted: {predicted2}")
py.axis("off")
py.show()

py.imshow(img3)
py.title(f"Predicted: {predicted3}")
py.axis("off")
py.show()

#Print all defect precentage results for each image.
print("Crack: All classes")
for label, prob in zip(class_labels_2, preds1):
    print(f" {label}: {prob*100:.2f}%")

print("Missing Head: All classes")
for label, prob in zip(class_labels_2, preds2):
    print(f" {label}: {prob*100:.2f}%")

print("Paint Off: All classes")
for label, prob in zip(class_labels_2, preds3):
    print(f" {label}: {prob*100:.2f}%")