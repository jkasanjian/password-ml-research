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
from Helper import (
    get_train_data,
    read_data,
    get_results,
    directoryExist,
    save_time_data,
)
from time import perf_counter

DATA_SPLIT = "data/split/"
MODELS_KNN = "src/models/all_data/"
MODELS_KNN_UB = "src/models/unbalanced_data/"


class KNN_Model:
    def __init__(self, balanced=False):
        _, self.subjects = read_data()

    def startTraining(self, reg=True, ada=False, Bagging=False):
        print("\n\n\n--------------TRAINING KNN--------------\n")
        if reg:
            self.knn_training()
        if ada:
            self.knn_training_with_adaBoost()
        if Bagging:
            self.knn_training_with_Bagging()

    def startTesting(self):
        model_names = ["KNN", "Bagging_KNN"]
        for model in model_names:
            get_results(self.subjects, model, "KNN")

    def knn_training(self, all_data=True):
        # Calculates and saves the best hyperparameters for each subject's KNN model
        time_data = []

        leaf_size = list(range(1, 50))
        n_neighbors = list(range(1, 30))
        p = [1, 2]
        hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data)
            knn_clf = KNeighborsClassifier(algorithm="brute", metric="minkowski")
            clf = GridSearchCV(knn_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            clf.fit(X_train, Y_train)
            end_time = perf_counter()

            directoryExist(MODELS_KNN + s)
            total_time = end_time - start_time
            time_data.append(total_time)
            print("Finished training:", s)
            dump(clf, MODELS_KNN + s + "/KNN.joblib")

        save_time_data("KNN", "KNN", "train", sum(time_data) / len(time_data))

    def knn_training_with_adaBoost(self, all_data=True):
        time_data = []

        # leaf_size = list(range(1, 50))
        # n_neighbors = list(range(1, 30))
        # p = [1, 2]
        # hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data)
            ab_clf = AdaBoostClassifier(
                KNeighborsClassifier(algorithm="brute", metric="minkowski")
            )
            # clf = GridSearchCV(ab_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            ab_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            directoryExist(MODELS_KNN + s)
            total_time = end_time - start_time
            time_data.append(total_time)
            print("Finished training:", s)
            dump(ab_clf, MODELS_KNN + s + "/Adaboost_KNN.joblib")

        save_time_data("KNN", "Adaboost_KNN", "train", sum(time_data) / len(time_data))

    def knn_training_with_Bagging(self, all_data=True):
        time_data = []
        # leaf_size = list(range(1, 50))
        # n_neighbors = list(range(1, 30))
        # p = [1, 2]
        # hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data)
            bagging_clf = BaggingClassifier(
                KNeighborsClassifier(algorithm="brute", metric="minkowski")
            )
            # clf = GridSearchCV(bagging_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            bagging_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            directoryExist(MODELS_KNN + s)
            total_time = end_time - start_time
            time_data.append(total_time)
            print("Finished training:", s)
            dump(bagging_clf, MODELS_KNN + s + "/Bagging_KNN.joblib")

        save_time_data("KNN", "Bagging_KNN", "train", sum(time_data) / len(time_data))


if __name__ == "__main__":
    KNN = KNN_Model()
    KNN.startTraining(True, True, True)
    KNN.startTesting()

    # X_test, Y_test = get_test_data("s037",True)
    # model = load_model("KNN","s037")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('KNN done training')
    # X_test, Y_test = get_test_data("s055",True)
    # model = load_model("KNN","s055")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('KNN done training')
    # X_test, Y_test = get_test_data("s050",True)
    # model = load_model("KNN","s030")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('KNN done training')