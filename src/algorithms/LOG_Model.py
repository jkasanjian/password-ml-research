###################
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump
from sklearn.model_selection import GridSearchCV
from logitboost import LogitBoost
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from time import perf_counter
from helper import (
    get_train_data,
    read_data,
    save_time_data,
)
from constants import MODELS_DIR


class LOG_Model:

    def __init__(self):
        _, self.subjects = read_data()


    def startTraining(self, model_variation, pca, ratio):
        if model_variation == "LOG_Grid":
            self.log_training_with_Grid(pca, ratio)
        elif model_variation == "LOG_LBoost":
            self.log_training_with_LogitBoost(pca, ratio)
        elif model_variation == "LOG_Adaboost":
            self.log_training_with_Adaboost(pca, ratio)
        elif model_variation == "LOG_Bagging":
            self.log_training_with_Bagging(pca, ratio)

    
    def log_training_with_Grid(self, pca, ratio):
        # Calculates and saves the best hyperparameters for each subject's
        print("~~~~~~~~~~Starting LOG Gridsearch~~~~~~~~~~")

        time_data = []

        penalty = ["l1", "l2", "elasticnet", "none"]
        C = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]
        solver = ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
        max_iter = [100, 1000, 2500, 5000]

        hyperparameters = dict(penalty=penalty, C=C, solver=solver, max_iter=max_iter)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            log_clf = LogisticRegression()
            grid_clf = GridSearchCV(log_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            grid_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            fname = MODELS_DIR + ratio + "/" + s + "/models/LOG_Grid"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(grid_clf, fname)

            print("Done with LOG_Grid for", s)

        save_time_data(ratio, "LOG", "LOG_Grid", pca, "train", sum(time_data) / len(time_data))

   
   
    def log_training_with_Adaboost(self, pca, ratio):
        time_data = []

        print("~~~~~~~~~~Starting LOG Adaboost~~~~~~~~~~")

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            ab_clf = AdaBoostClassifier(LogisticRegression(), n_estimators=100)

            start_time = perf_counter()
            ab_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            fname = MODELS_DIR + ratio + "/" + s + "/models/LOG_Adaboost"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(ab_clf, fname)

            print("Done with LOG_Adaboost for", s)
            
        save_time_data(ratio, "LOG", "LOG_Adaboost", pca, "train", sum(time_data) / len(time_data))



    def log_training_with_LogitBoost(self, pca, ratio):
        time_data = []

        print("~~~~~~~~~~Starting LogitBoost~~~~~~~~~~")

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            lb_clf = LogitBoost(n_estimators=200, bootstrap=True)

            start_time = perf_counter()
            lb_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            fname = MODELS_DIR + ratio + "/" + s + "/models/LOG_LBoost"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(lb_clf, fname)

            print("Done with LOG_LBoost for", s)

        save_time_data(ratio, "LOG", "LOG_LBoost", pca, "train", sum(time_data) / len(time_data))

    
    def log_training_with_Bagging(self, pca, ratio):
        time_data = []

        print("~~~~~~~~~~Starting LOG_Bagging~~~~~~~~~~")


        for s in self.subjects:
            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            bagging_clf = BaggingClassifier(LogisticRegression(), n_estimators=15)

            start_time = perf_counter()
            bagging_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            fname = MODELS_DIR + ratio + "/" + s + "/models/LOG_Bagging"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(bagging_clf, fname)

            print("Done with LOG_Bagging for", s)

        save_time_data(ratio, "LOG", "LOG_Bagging", pca, "train", sum(time_data) / len(time_data))



if __name__ == "__main__":
    LOG = LOG_Model()
    
