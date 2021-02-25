import numpy as np
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from time import perf_counter
from helper import (
    get_train_data,
    read_data,
    save_time_data,
)
from constants import MODELS_DIR


class SVM_Model:
    def __init__(self, balanced=False):
        _, self.subjects = read_data()

    def startTraining(self, model_variation, pca, ratio):
        # Trains one type of model given flags
        if model_variation == "SVM_Grid":
            self.svm_training_with_Grid(pca, ratio)
        elif model_variation == "SVM_Adaboost":
            self.svm_training_with_Adaboost(pca, ratio)
        elif model_variation == "SVM_Bagging":
            self.svm_training_with_Bagging(pca, ratio)


    def svm_training_with_Grid(self, pca, ratio):
        # Calculates and saves the best hyperparameters for each subject's KNN model
        print("~~~~~~~~~~Starting SVM Gridsearch~~~~~~~~~~")

        time_data = []

        # SVM with gridsearch is still as optimal as bagging, bagging still optimizes even if its very low margins
        kernel = ["linear", "rbf", "poly", "sigmoid"]

        C = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]
        degree = [i for i in range(1, 7)]
        hyperparameters = dict(kernel=kernel, C=C, degree=degree)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            svm_clf = SVC(gamma="auto", decision_function_shape="ovo", probability=True)
            clf = GridSearchCV(svm_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            clf.fit(X_train, Y_train)
            end_time = perf_counter()

            fname = MODELS_DIR + ratio + "/" + s + "/models/SVM_Grid"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(clf, fname)

            print("Done with SVM_Grid for", s)

        save_time_data(ratio, "SVM", "SVM_Grid", pca, "train", sum(time_data) / len(time_data))


    def svm_training_with_Adaboost(self, pca, ratio):
        print("~~~~~~~~~~Starting SVM Adaboost~~~~~~~~~~")

        time_data = []

        # Adaboost with the best estimator from grid search (SVC estimator) sucks
        # Also why does it make the accuracies weaker?
        # Can we boost something from bagging?
        # maybe just pass in adaboost with just SVM alone
        for s in self.subjects:

            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            svm_clf = SVC(probability=True, gamma="auto", decision_function_shape="ovo")
            ada_clf = AdaBoostClassifier(svm_clf, n_estimators=10)

            start_time = perf_counter()
            ada_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            fname = MODELS_DIR + ratio + "/" + s + "/models/SVM_Adaboost"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(ada_clf, fname)

            print("Done with SVM_Adaboost for", s)

        save_time_data(ratio, "SVM", "SVM_Adaboost", pca, "train", sum(time_data) / len(time_data))


    def svm_training_with_Bagging(self, pca, ratio):
        print("~~~~~~~~~~Starting SVM Bagging~~~~~~~~~~")

        time_data = []

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            sv_clf = SVC(probability=True, gamma="auto", decision_function_shape="ovo")
            bag_clf = BaggingClassifier(sv_clf, n_estimators=100)

            start_time = perf_counter()
            bag_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            fname = MODELS_DIR + ratio + "/" + s + "/models/SVM_Bagging"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(bag_clf, fname)

            print("Done with SVM_Bagging for", s)

        save_time_data(ratio, "SVM", "SVM_Bagging", pca, "train", sum(time_data) / len(time_data))


if __name__ == "__main__":
    SVM = SVM_Model()


   
