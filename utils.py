import numpy as np
import matplotlib.pyplot as plt
import json
from keras.models import Model
def saveFigures(folder, history, accuracy_img_name, loss_img_name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Save the figure
    plt.savefig(folder+'/'+accuracy_img_name)
    plt.clf()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Save the figure
    plt.savefig(folder+'/'+loss_img_name)
    plt.clf()

#json def
def write_json(new_data, filename):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data.update(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4, separators=(',', ': '))
def create_json(data, filename):
    with open(filename,'w') as file:
        json.dump(data, file, indent = 4, separators=(',', ': '))

def load_previous_weights(model,weights_path):
    if weights_path:
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except:
            print("Impossible to find weight path. Returning untrained model")
    return model


