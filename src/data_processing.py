import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from os import path, mkdir
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

def run_pca(X,y):

    pca = PCA(n_components=17) # estimate only 2 PCs
    X = pca.fit_transform(X) # project the original data into the PCA space
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    return X_train, X_test, y_train, y_test


def pca_data_all():

    scaler = StandardScaler()            
    data, _ = read_data()

    for s in data:
        x_pos, x_neg = get_pos_neg(data, s)
        x = np.array(x_pos + x_neg)
        scaler.fit(x)
        x = scaler.transform(x)
        y = np.array([1 for i in range(len(x_pos))] + [-1 for i in range(len(x_neg))])
        x_train, x_test, y_train, y_test = run_pca(x,y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

        if not path.isdir(DATA_PARTITIONS + "all_data/" + s):
            mkdir(DATA_PARTITIONS + "all_data/" + s)
        np.save(DATA_PARTITIONS + "all_data/" + s + "/x_pca_train.npy", x_train)
        np.save(DATA_PARTITIONS + "all_data/" + s + "/x_pca_test.npy", x_test)
        np.save(DATA_PARTITIONS + "all_data/" + s + "/y_pca_train.npy", y_train)
        np.save(DATA_PARTITIONS + "all_data/" + s + "/y_pca_test.npy", y_test)



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

        if not path.isdir(DATA_PARTITIONS + "pos-50/" + s):
            mkdir(DATA_PARTITIONS + "pos-50/" + s)
        np.save(DATA_PARTITIONS + "pos-50/" + s + "/x_train.npy", x_train)
        np.save(DATA_PARTITIONS + "pos-50/" + s + "/x_test.npy", x_test)
        np.save(DATA_PARTITIONS + "pos-50/" + s + "/y_train.npy", y_train)
        np.save(DATA_PARTITIONS + "pos-50/" + s + "/y_test.npy", y_test)


def partition_data_ratio(pos_ratio):
    """ Partitions the data according to ration of positive to negative data """
    data, _ = read_data()

    for s in data:
        x_pos, x_neg = get_pos_neg(data, s)
        x = []
        y = []
        x.extend(x_pos)
        
        num_neg = round((len(x_pos)*100)/pos_ratio)
        for _ in range(num_neg):
            r = np.random.randint(0, len(x_neg))
            x.append(x_neg.pop(r))
        y.extend([1 for _ in range(len(x_pos))] + [-1 for _ in range(len(x)-len(x_pos))])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

        data_dir = DATA_PARTITIONS + "pos-" + str(pos_ratio)
        if not path.isdir(data_dir):
            mkdir(data_dir)
        if not path.isdir(data_dir + "/" + s):
            mkdir(data_dir + "/" + s)
        np.save(data_dir + "/" + s + "/x_train.npy", x_train)
        np.save(data_dir + "/" + s + "/x_test.npy", x_test)
        np.save(data_dir + "/" + s + "/y_train.npy", y_train)
        np.save(data_dir + "/" + s + "/y_test.npy", y_test)




if __name__ == "__main__":
    """ Main method """
    ratios = [10, 20, 30, 40, 60, 70, 80, 90]
    for r in ratios:
        partition_data_ratio(r)

    print("----------FINISHED EXECUTION----------")