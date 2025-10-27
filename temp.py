# Data Processing to perform any type of image classification task
# Desired width, height and channel of the image for model traning

import numpy as np
import matplotlib.pyplot as py

from tensorflow import keras
from tensorflow.keras import layers

# For reproductability

np.random.seed(42)
keras.utils.set_random_seed(42)


train_dataset = keras.utlis.image_dataset_from_directory(
    "Downloads/Project 2 Data/Data/train",
        image_size = (500, 500), 
        batch_size = 32,
        label_model = 'categorical'
        
)
        
val_dataset = keras.utlis.image_dataset_from_directory(
    "Downloads/Project 2 Data/Data/Valid",
        image_size = (500, 500),
        batch_size = 32,
        label_model = 'categorical'

)

test_dataset = keras.utils.image_dataset_from_directory(
    "Downlaods/Project 2 Data/Data/Valid",
        image_size = (500, 500),
        batch_size = 32,
        label_model = 'categorical'
        
)