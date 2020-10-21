from os import path, mkdir
import numpy as np
import json
from joblib import dump, load

DATA_JSON = 'data/password_data.json'

def read_data():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)
    # features = data['subject']
    del data['subject']
    subjects = list(data.keys())
    return data, subjects


def load_model(name, s):
    ''' Returns the corresponding model for a subject '''
    return load('/Desktop/password-ml-research/src/models/all_data/' + s + "/" + name + ".joblib")


def get_test_data(subject, is_balanced):
    ''' Returns the testing data partition for the given subject.
        is_balanced is a boolean field. if true, returns test data
        that is classed-balanced. if false, returns test data with
        the same class proportions as the entire dataset ''' 

    if is_balanced:
        path = 'data/partitions/balanced_data/'
    else:
        path = 'data/partitions/all_data/'
    x_test = np.load(path + subject + '/x_test.npy')
    y_test = np.load(path + subject + '/y_test.npy')

    return x_test, y_test


def get_train_data(subject, is_balanced):
    ''' Returns the training data partition for the given subject.
        is_balanced is a boolean field. if true, returns train data
        that is classed-balanced. if false, returns train data with
        the same class proportions as the entire dataset ''' 

    if is_balanced:
        path = 'data/partitions/balanced_data/'
    else:
        path = 'data/partitions/all_data/'
    x_train = np.load(path + subject + '/x_train.npy')
    y_train = np.load(path + subject + '/y_train.npy')

    return x_train, y_train






if __name__ == "__main__":
    ''' Main method '''
    # test models