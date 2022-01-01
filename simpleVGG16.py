import requests
import os
import numpy as np
import math

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

from keras.callbacks import LearningRateScheduler, EarlyStopping

from keras.optimizers import gradient_descent_v2 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from models.vgg16 import VGG16
from utils import saveFigures

### Don't be greedy on GPU RAM
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

### Constant configuration parameters
img_width = 32
img_height = 32
img_channels = 3

batch_size = 512
training_epochs = 250

output_dim = 10

### Load data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images.astype(np.float32)/255.
test_images = test_images.astype(np.float32)/255.

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

datagen = ImageDataGenerator()
datagen.fit(train_images)

## First iteration

### Set tunable parameters
learning_rate = 0.05
lr_drop = 100.
dropout = 0.5

#lr_decay = 1e-2
#momentum = 0.9

### Define model
model = VGG16(
    dropout=dropout,
    num_classes=output_dim,
    img_width=img_width,
    img_height=img_height,
    img_channels=img_channels,
)

def lr_scheduler(epoch):
    return learning_rate * (math.e ** (-epoch / lr_drop))
reduce_lr = LearningRateScheduler(lr_scheduler)

early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=25,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

optimizer = gradient_descent_v2.SGD(learning_rate=learning_rate, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

### Compute and evaluate function
history = model.fit(
    datagen.flow(
        train_images, 
        train_labels, 
        batch_size=batch_size), 
    epochs=training_epochs, 
    verbose=1,     
    callbacks=[reduce_lr,early_stop],
    steps_per_epoch=train_images.shape[0] // batch_size, 
    validation_data=(test_images, test_labels))

### Save figures
saveFigures("ExperimentName", history, "accuracy_0", "loss_0")
