import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import datasets, layers, models



# Step 1: Image Data Generators

#Split train, valid and test into subsets


# Set Batch size, image height, width, and channel (500, 500, 3)

batch_size = 32
Color_img_shape = (500, 500, 3)

# Set variables to directed paths of train, test and test.

Data_Train = 'C:/Users/Harve/Downloads/Project 2 Data/Data/train'
Data_Val = 'C:/Users/Harve/Downloads/Project 2 Data/Data/valid'
Data_Test = 'C:/Users/Harve/Downloads/Project 2 Data/Data/test'


# Import ImageDataGenerator to train, valid and test to preprocess of image data
# Large images require large memory and computational power. Reducing to smaller dimension 1./255 lead to faster training
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
     
   rescale = 1./255, 
   shear_range = 0.2, #Improves model ability to generalize unseen data
   zoom_range = 0.2, #Model benefits from large datasets
   horizontal_flip = True) # Expands traning Datasets

valid_datagen = ImageDataGenerator(
    rescale = 1./255)

test_datagen = ImageDataGenerator(
    rescale = 1./255)

# Set up Generators 

train_generator = train_datagen.flow_from_directory(
    Data_Train, #Accessing Train datasets
    target_size = (500, 500), #Input value (500, 500)
    batch_size = 32,  #Set Batch Size
    class_mode = 'categorical') 

valid_generator = valid_datagen.flow_from_directory(
    Data_Val,
    target_size = (500,500),
    batch_size = 32,
    class_mode = 'categorical') 


#Save class labels 
class_labels = list(train_generator.class_indices.keys())
with open("class_labels.json", "w") as f:
    json.dump(class_labels, f)

# Setup Generator 

test_generator = test_datagen.flow_from_directory(
    Data_Test,
    target_size = (500, 500),
    batch_size = 32,
    class_mode = 'categorical',
    )


# Step 2: Neural Network Architecture Design: Tensorflow

# Conv2D layers 



from tensorflow.keras.layers import Conv2D

#Utlized 4 layers for it to learn deeper features and more accurate classification.
#RelU is perfferred for simple DCNN datasets. 

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (500, 500, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3,3), activation = 'relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation = "softmax") #Softmax for final layer because of multiclassification
    
])


model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 1e-3), #Adaptive learning Rates (Adam) as optimizer.
    loss="categorical_crossentropy",
    metrics=["accuracy"]  #Monitoring Accuarcy Metric
    )

# Step 3: Train Model

EPOCHS = 1 # This is changed for now because I had added json last and did not have time to run 15 Epochs.
#However my report has evidence of Epoch 30 Model 1 and Epoch 15 Model 2

#ndim

# Prevents overfitting, once validation accuracy decreases 3 times consistently.
early_stop = keras.callbacks.EarlyStopping(
    monitor = "val_accuracy",
    patience = 3,
    restore_best_weights = True  #Chooses best metrics
    )


# Fit model into train generator, validation data, epoch and callbacks. 

history = model.fit(
   train_generator,
   validation_data = valid_generator ,
   epochs = EPOCHS,
   batch_size= batch_size ,
   callbacks=(early_stop),
   verbose = 1
   )

# Evaluate the Test Model 

test_loss, test_acc = model.evaluate(test_generator, verbose = 1)
print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")


#Plot Epoch vs Accruacy and Epoch vs Loss

model.save("trained_model.h5")
print("Model saved successfully.")

# Accuracy vs Epoch
py.figure(figsize = (6, 4))
py.plot(history.history["accuracy"], label="Train Acc")
py.plot(history.history["val_accuracy"], label = "Val Acc")
py.xlabel("Epoch")
py.ylabel("Accuracy")
py.title("Accuracy vs Epoch") 
py.legend()
py.grid(True)
py.show()

# Loss vs Epoch
py.figure(figsize = (6, 4))
py.plot(history.history["loss"], label = "Train loss")
py.plot(history.history["val_loss"], label="Val Loss")
py.xlabel("Epoch")
py.ylabel("Loss")
py.title("Loss Vs Epoch")
py.legend()
py.grid(True)
py.show()

 

 