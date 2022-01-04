from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

def plotcm(path_to_results, real_labels, pred_labels, classes,
                          experiment_rootdir, normalize=True):
    """
    Plot and save confusion matrix computed from predicted and real labels.
    
    # Arguments
        path_to_results: Location where saving confusion matrix.
        real_labels: List of real labels.
        pred_prob: List of predicted probabilities.
        normalize: Boolean, whether to apply normalization.
    """
    
    # Generate confusion matrix
    cm = confusion_matrix(real_labels, pred_labels)
    fig, ax = plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, float('%.1f'%(cm[i, j])),
                 horizontalalignment="center")
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig_save_path = os.path.join(experiment_rootdir, "CM.png")
    plt.savefig(fig_save_path)
    plt.show()