import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from os import path, mkdir
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from constants import DATA_PARTITIONS_DIR, SUBSET_SIZE



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
    return x_pos.copy(), x_neg.copy()


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


        if not path.isdir(DATA_PARTITIONS_DIR + "all_data/" + s):
            mkdir(DATA_PARTITIONS_DIR + "all_data/" + s)
        np.save(DATA_PARTITIONS_DIR + "all_data/" + s + "/x_pca_train.npy", x_train)
        np.save(DATA_PARTITIONS_DIR + "all_data/" + s + "/x_pca_test.npy", x_test)
        np.save(DATA_PARTITIONS_DIR + "all_data/" + s + "/y_pca_train.npy", y_train)
        np.save(DATA_PARTITIONS_DIR + "all_data/" + s + "/y_pca_test.npy", y_test)



def partition_data_all():
    """ Partitions the data using all datapoints and saves in files """
    data, _ = read_data()

    for s in data:
        x_pos, x_neg = get_pos_neg(data, s)
        x = np.array(x_pos + x_neg)
        y = np.array([1 for i in range(len(x_pos))] + [-1 for i in range(len(x_neg))])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

        path = DATA_PARTITIONS_DIR + "all_data/" + s
        if not path.isdir(path):
            mkdir(path)
        np.save(path + "/x_train.npy", x_train)
        np.save(path + s + "/x_test.npy", x_test)
        np.save(path + "all_data/" + s + "/y_train.npy", y_train)
        np.save(path + "all_data/" + s + "/y_test.npy", y_test)


def partition_data_balanced():
    """ Partitions the data in class balanced form and saves in files 
    Utilizes all 400 of the positive samples, and randomly selects 
    400 negative samples to produce the subset"""
    data, _ = read_data()
    scaler = StandardScaler()
    for s in data:
        x_pos, x_neg = get_pos_neg(data, s)
        x = []
        y = []
        x.extend(x_pos)
    
        for _ in range(len(x_pos)):
            r = np.random.randint(0, len(x_neg))
            x.append(x_neg.pop(r))

        y.extend([1 for _ in range(len(x_pos))] + [-1 for _ in range(len(x)-len(x_pos))])

        X = np.array(x,dtype = 'float64')
        Y = np.array(y,dtype = 'float64')
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
    
        data_dir = DATA_PARTITIONS_DIR + "balanced"
        if not path.isdir(data_dir):
            mkdir(data_dir)
        if not path.isdir(data_dir + "/" + s):
            mkdir(data_dir + "/" + s)
        np.save(data_dir + "/" + s + "/x_train.npy", x_train)
        np.save(data_dir + "/" + s + "/x_test.npy", x_test)
        np.save(data_dir + "/" + s + "/y_train.npy", y_train)
        np.save(data_dir + "/" + s + "/y_test.npy", y_test)

        # Does PCA simultaneously
        scaler.fit(X)
        X = scaler.transform(X)
        x_train, x_test, y_train, y_test = run_pca(X, Y)
        np.save(data_dir + "/" + s + "/x_pca_train.npy", x_train)
        np.save(data_dir + "/" + s + "/x_pca_test.npy", x_test)
        np.save(data_dir + "/" + s + "/y_pca_train.npy", y_train)
        np.save(data_dir + "/" + s + "/y_pca_test.npy", y_test)
        


def partition_data_ratio(pos_ratio):
    """ Partitions the data according to ration of positive to negative data """
    data, _ = read_data()
    scaler = StandardScaler()
    for s in data:
        x_pos, x_neg = get_pos_neg(data, s)
        x = []
        y = []
    
        num_pos = round((SUBSET_SIZE * pos_ratio)/100)
        num_neg = SUBSET_SIZE - num_pos

        for _ in range(num_pos):
            r = np.random.randint(0, len(x_pos))
            x.append(x_pos.pop(r))

        for _ in range(num_neg):
            r = np.random.randint(0, len(x_neg))
            x.append(x_neg.pop(r))

        y.extend([1 for _ in range(num_pos)] + [-1 for _ in range(num_neg)])

        X = np.array(x,dtype = 'float64')
        Y = np.array(y,dtype = 'float64')

        print(len(X))

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
    
        data_dir = DATA_PARTITIONS_DIR + "pos-" + str(pos_ratio)
        if not path.isdir(data_dir):
            mkdir(data_dir)
        if not path.isdir(data_dir + "/" + s):
            mkdir(data_dir + "/" + s)
        np.save(data_dir + "/" + s + "/x_train.npy", x_train)
        np.save(data_dir + "/" + s + "/x_test.npy", x_test)
        np.save(data_dir + "/" + s + "/y_train.npy", y_train)
        np.save(data_dir + "/" + s + "/y_test.npy", y_test)

        # Does PCA simultaneously
        scaler.fit(X)
        X = scaler.transform(X)
        x_train, x_test, y_train, y_test = run_pca(X, Y)

        np.save(data_dir + "/" + s + "/x_pca_train.npy", x_train)
        np.save(data_dir + "/" + s + "/x_pca_test.npy", x_test)
        np.save(data_dir + "/" + s + "/y_pca_train.npy", y_train)
        np.save(data_dir + "/" + s + "/y_pca_test.npy", y_test)



if __name__ == "__main__":
    """ Main method """
    # ratios = [10, 20, 30, 40, 60, 70, 80, 90]
    # for r in ratios:
    #     partition_data_ratio(r)

    # partition_data_balanced()

    print("----------FINISHED EXECUTION----------")