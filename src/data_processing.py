from os import path, mkdir
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATA_PARTITIONS = "data/partitions/"


def read_data():
    """ Reads the data file and returns as a dictionary """
    with open(path.abspath("data/password_data.json")) as json_file:
        data = json.load(json_file)
    labels = data["subject"]
    del data["subject"]
    return data, labels


def get_pos_neg(data, subject):
    """ Returns positive and negative data for a given subject"""
    x_pos = data[subject]
    x_neg = []
    for s in data:
        if s != subject:
            x_neg.extend(data[s])
    return x_pos, x_neg


def partition_data_all():
    """ Partitions the data using all datapoints and saves in files """
    data, _ = read_data()

    for s in data:
        x_pos, x_neg = get_pos_neg(data, s)
        x = np.array(x_pos + x_neg)
        y = np.array([1 for i in range(len(x_pos))] + [-1 for i in range(len(x_neg))])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

        if not path.isdir(DATA_PARTITIONS + "all_data/" + s):
            mkdir(DATA_PARTITIONS + "all_data/" + s)
        np.save(DATA_PARTITIONS + "all_data/" + s + "/x_train.npy", x_train)
        np.save(DATA_PARTITIONS + "all_data/" + s + "/x_test.npy", x_test)
        np.save(DATA_PARTITIONS + "all_data/" + s + "/y_train.npy", y_train)
        np.save(DATA_PARTITIONS + "all_data/" + s + "/y_test.npy", y_test)


def partition_data_balanced():
    """ Partitions the data in class balanced form and saves in files """
    data, _ = read_data()

    for s in data:
        x_pos, x_neg = get_pos_neg(data, s)
        x = []
        y = []
        x.extend(x_pos)
        for i in range(len(x_pos)):
            r = np.random.randint(0, len(x_neg))
            x.append(x_neg.pop(r))
        y.extend([1 for i in range(len(x_pos))] + [-1 for i in range(len(x_pos))])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

        if not path.isdir(DATA_PARTITIONS + "balanced_data/" + s):
            mkdir(DATA_PARTITIONS + "balanced_data/" + s)
        np.save(DATA_PARTITIONS + "balanced_data/" + s + "/x_train.npy", x_train)
        np.save(DATA_PARTITIONS + "balanced_data/" + s + "/x_test.npy", x_test)
        np.save(DATA_PARTITIONS + "balanced_data/" + s + "/y_train.npy", y_train)
        np.save(DATA_PARTITIONS + "balanced_data/" + s + "/y_test.npy", y_test)


if __name__ == "__main__":
    """ Main method """
    partition_data_all()
    partition_data_balanced()
    print("----------FINISHED EXECUTION----------")