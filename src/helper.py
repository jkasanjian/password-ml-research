import numpy as np
import json
import os
from AUROC import AUROC
from sklearn import metrics
from joblib import dump, load
from time import perf_counter
from constants import (
    DATA_JSON,
    RESULT_JSON,
    DATA_PARTITIONS_DIR,
    DATA_RATIOS,
    MODEL_GROUPS,
    MODEL_VARIATIONS,
)


def read_data():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)
    # features = data['subject']
    del data["subject"]
    subjects = list(data.keys())
    return data, subjects


def save_time_data(ratio, base_model, model_variation, pca, train_or_test, avg_time):
    with open(RESULT_JSON) as json_file:
        results_data = json.load(json_file)

    time_key = "avg train time" if train_or_test == "train" else "total test time"
    pca_status = "pca_on" if pca else "pca_off"

    results_data[ratio][base_model + "_group"][model_variation][pca_status][time_key] = avg_time

    with open(RESULT_JSON, "w") as outfile:
        json.dump(results_data, outfile)


def directoryExist(name):
    if not os.path.isdir(name):
        os.makedirs(name)


# Need to fix for balanced and unbalanced
def load_model(name, s, pca, ratio):
    p = "_pca" if pca else ""
    """ Returns the corresponding model for a subject """
    path = 'models/' + ratio + '/' + s + '/models/' + name + p + '.joblib'
    return load(path)


def get_test_data(subject, pca, ratio):
    """Returns the testing data partition for the given subject.
    is_balanced is a boolean field. if true, returns test data
    that is classed-balanced. if false, returns test data with
    the same class proportions as the entire dataset"""

    path = 'data/partitions/' + ratio + '/' + subject
    if pca:
        x_path = path + "/x_pca_test.npy"
        y_path = path + "/y_pca_test.npy"
    else:
        x_path = path + "/x_test.npy"
        y_path = path + "/y_test.npy"

    x_test = np.load(x_path)
    y_test = np.load(y_path)

    return x_test.astype(np.float), y_test.astype(np.float)


def get_train_data(subject, ratio, pca):
    """Returns the training data partition for the given subject"""

    path = "data/partitions/" + ratio + "/" + subject
    if pca:
        x_path = path + "/x_pca_train.npy"
        y_path = path + "/y_pca_train.npy"
    else:
        x_path = path + "/x_train.npy"
        y_path = path + "/y_train.npy"

    x_train = np.load(x_path)
    y_train = np.load(y_path)

    return x_train.astype(np.float), y_train.astype(np.float)



if __name__ == "__main__":
    """ Main method """
    # save_time_data("SVM", "svm_alone", "train", 420.69)
