import numpy as np
import matplotlib.pyplot as plt

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