# Data Processing to perform any type of image classification task
# Desired width, height and channel of the image for model traning

import numpy as np
import matplotlib.pyplot as py

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# For reproductability

np.random.seed(42)
keras.utils.set_random_seed(42)

# Scale for only validation data, re-scaling, shear range and zoom range 
# Images in the dataset are grayscale with pixel in [0, 255] (integers).

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2, 
    horizontal_flip = True
    
)


#Re-scaling for validation data only
val_datagen = ImageDataGenerator(rescale=1./255)



from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen 


train_dataset = train_datagen.flow_dataset_from_directory(
   r"C:\Users\h1sehra\Downloads\Project 2 Data\Data\train",
        target_size = (500, 500), 
        batch_size = 32,
        class_mode = 'categorical'
        
)
        
val_dataset = val_datagen.flow_dataset_from_directory(
    "C:\Users\h1sehra\Downloads\Project 2 Data\Data\Valid",
        target_size = (500, 500),
        batch_size = 32,
        class_mode = 'categorical'

)


# Step 2: Neural Network Architecture Design 
