# temporarily need
from sklearn import metrics

###################
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from time import perf_counter
from helper import (
    get_train_data,
    read_data,
    save_time_data,
    directoryExist
)
from constants import MODELS_DIR



class KNN_Model:

    def __init__(self):
        _, self.subjects = read_data()


    def startTraining(self, model_variation, pca, ratio):
        # Trains one type of model given flags
        if model_variation == "KNN_Grid":
            self.knn_training_with_Grid(pca, ratio)
        elif model_variation == "KNN_Adaboost":
            self.knn_training_with_Adaboost(pca, ratio)
        elif model_variation == "KNN_Bagging":
            self.knn_training_with_Bagging(pca, ratio)


    def knn_training_with_Grid(self, pca, ratio):
        # Calculates and saves the best hyperparameters for each subject's KNN model
        print("~~~~~~~~~~Starting KNN Gridsearch~~~~~~~~~~")

        time_data = []

        leaf_size = list(range(1, 50))
        n_neighbors = list(range(1, 30))
        p = [1, 2]
        
        hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            knn_clf = KNeighborsClassifier(algorithm="brute", metric="minkowski")
            clf = GridSearchCV(knn_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            clf.fit(X_train, Y_train)
            end_time = perf_counter()

            directoryExist(MODELS_DIR + ratio + "/" + s)
            directoryExist(MODELS_DIR + ratio + "/" + s + '/models')

            fname = MODELS_DIR + ratio + "/" + s + "/models/KNN_Grid"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(clf, fname)

            print("Done with KNN_Grid for", s)

        save_time_data(ratio, "KNN", "KNN_Grid", pca, "train", sum(time_data) / len(time_data))


    def knn_training_with_Adaboost(self, pca, ratio):
        # Error on ab_clf.fit() - ValueError: KNeighborsClassifier doesn't support sample_weight.

        print("~~~~~~~~~~Starting KNN Adaboost~~~~~~~~~~")
        time_data = []


        for s in self.subjects:
            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            ab_clf = AdaBoostClassifier(
                KNeighborsClassifier(algorithm="brute", metric="minkowski")
            )
           
            start_time = perf_counter()
            ab_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            directoryExist(MODELS_DIR + ratio + "/" + s)
            directoryExist(MODELS_DIR + ratio + "/" + s + '/models')

            fname = MODELS_DIR + ratio + "/" + s + "/models/KNN_Adaboost"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(ab_clf, fname)

            print("Done with KNN_Adaboost for", s)

        save_time_data(ratio, "KNN", "KNN_Adaboost", pca, "train", sum(time_data) / len(time_data))


    def knn_training_with_Bagging(self, pca, ratio):
        print("~~~~~~~~~~Starting KNN Bagging~~~~~~~~~~")
        time_data = []
    
        for s in self.subjects:
            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            bagging_clf = BaggingClassifier(
                KNeighborsClassifier(algorithm="brute", metric="minkowski")
            )

            start_time = perf_counter()
            bagging_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            directoryExist(MODELS_DIR + ratio + "/" + s)
            directoryExist(MODELS_DIR + ratio + "/" + s + 'models')

            fname = MODELS_DIR + ratio + "/" + s + "/models/KNN_Bagging"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(bagging_clf, fname)

            print("Done with KNN_Bagging for", s)

        save_time_data(ratio, "KNN", "KNN_Bagging", pca, "train", sum(time_data) / len(time_data))


if __name__ == "__main__":
    KNN = KNN_Model()
