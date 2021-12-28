import requests
import os
import numpy as np
import time
import math

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.datasets import cifar10

from keras.layers import Input, Dense
from keras.callbacks import LearningRateScheduler

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import gradient_descent_v2 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from models.vgg16 import VGG16

## Set API back-end direction
backend="http://172.19.0.1:8003/api/"

## Set how many samples you want to get
tota_iter = 20

experiment = {
    "name": "PRDL-exp1",
    "n_ins": 2, 
    "input_names": ["Learning rate", "LR drop", "dropout"], 
    "input_mms": [[0, 0.1],[10, 190],[0,1]],
    "n_objs": 1,
    "objective_names": ["Accuracy"],
    "objective_mms": [True],
    "n_cons": 0,
    "kernel": 'RBF',
    "acq_id": 2,
    "acqfunct_hps": {"N": 1000, "M": 50}
}

response = requests.post(backend+'startexp', json=experiment)
if response.status_code==200:
    response = response.json()
    for key in response:
        experiment[key]=response[key]


if not os.path.exists(experiment["name"]):
    os.makedirs(experiment["name"])

print(experiment)

### Don't be greedy on GPU RAM
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


### Constant configuration parameters
img_width = 32
img_height = 32
img_channels = 3

batch_size = 1024
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
    callbacks=[reduce_lr],
    steps_per_epoch=train_images.shape[0] // batch_size, 
    validation_data=(test_images, test_labels))

### Send results to API
y = [history.history["val_accuracy"][-1]]
x = [learning_rate, lr_drop, dropout]
response = requests.post(backend+"setsample/"+str(experiment["id"]), json={"x": x, "y": y})


## Main Loop
for i in range(tota_iter):
    response = requests.get(backend+"getsample/next/"+str(experiment["id"]))

    ## Charge next configuration 
    [learning_rate, lr_drop, dropout]  = response.json()["next_x"]

    ## Define the model
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

    optimizer = gradient_descent_v2.SGD(learning_rate=learning_rate, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    ## Evaluate the model
    history = model.fit(
        datagen.flow(
            train_images, 
            train_labels, 
            batch_size=batch_size), 
        epochs=training_epochs, 
        verbose=1,     
        callbacks=[reduce_lr],
        steps_per_epoch=train_images.shape[0] // batch_size, 
        validation_data=(test_images, test_labels))

    y = [history.history["val_accuracy"][-1]]
    x = [learning_rate, lr_drop]

    print("-"*60)
    print(x, y)
    print("-"*60)
    response = requests.post(backend+"setsample/"+str(experiment["id"]), json={"x": x, "y": y})
