# temporarily need
from sklearn import metrics

###################
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump
from sklearn.model_selection import GridSearchCV
from logitboost import LogitBoost
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from Helper import (
    directoryExist,
    get_train_data,
    read_data,
    get_results,
    get_test_data,
    save_time_data,
)
from time import perf_counter

DATA_SPLIT = "data/split/"
MODELS_LOG = "src/models/all_data/"
MODELS_LOG_UB = "src/models/unbalanced_data/"


class LOG_Model:
    def __init__(self, balanced=False):
        _, self.subjects = read_data()

    def startTraining(self, reg=True, logit=False, ada=False, Bagging=False):
        print("\n\n\n--------------TRAINING LOG--------------\n")
        if reg:
            self.log_training()
        if logit:
            self.log_training_with_LogitBoost()
        if ada:
            self.log_training_with_Adaboost()
        if Bagging:
            self.log_training_with_Bagging()

    def startTesting(self):
        model_names = ["LOG", "Adaboost_LOG", "LBoost_LOG", "Bagging_LOG"]
        for i in model_names:
            get_results(self.subjects, i, "LOG")

    def log_training(self, all_data=True):
        # Calculates and saves the best hyperparameters for each subject's
        # Logistic Regression model
        # penalty = ["l2"]
        # C = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]
        # solver = ["lbfgs", "newton-cg", "sag", "saga"]
        # max_iter = [100, 1000, 2500]
        time_data = []

        penalty = ["l1", "l2", "elasticnet", "none"]
        C = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]
        solver = ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]
        max_iter = [100, 1000, 2500, 5000]

        hyperparameters = dict(penalty=penalty, C=C, solver=solver, max_iter=max_iter)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            log_clf = LogisticRegression()
            grid_clf = GridSearchCV(log_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            grid_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            directoryExist(MODELS_LOG + s)
            total_time = end_time - start_time
            time_data.append(total_time)
            dump(grid_clf, MODELS_LOG + s + "/LOG.joblib")
            print("done", s)

        save_time_data("LOG", "LOG", "train", sum(time_data) / len(time_data))

    def log_training_with_Adaboost(self, all_data=True):
        time_data = []

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            ab_clf = AdaBoostClassifier(LogisticRegression(), n_estimators=100)

            start_time = perf_counter()
            ab_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            directoryExist(MODELS_LOG + s)
            total_time = end_time - start_time
            time_data.append(total_time)
            dump(ab_clf, MODELS_LOG + s + "/Adaboost_LOG.joblib")
            print("done", s)

        save_time_data("LOG", "Adaboost_LOG", "train", sum(time_data) / len(time_data))

    def log_training_with_LogitBoost(self, all_data=True):
        time_data = []

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            lb_clf = LogitBoost(n_estimators=200, bootstrap=True)

            start_time = perf_counter()
            lb_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            directoryExist(MODELS_LOG + s)
            total_time = end_time - start_time
            time_data.append(total_time)
            dump(lb_clf, MODELS_LOG + s + "/LBoost_LOG.joblib")
            print("done", s)

        save_time_data("LOG", "LBoost_LOG", "train", sum(time_data) / len(time_data))

    def log_training_with_Bagging(self, all_data=True):
        time_data = []

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            bagging_clf = BaggingClassifier(LogisticRegression(), n_estimators=15)

            start_time = perf_counter()
            bagging_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            directoryExist(MODELS_LOG + s)
            total_time = end_time - start_time
            time_data.append(total_time)

            dump(bagging_clf, MODELS_LOG + s + "/Bagging_LOG.joblib")
            print("done", s)

        save_time_data("LOG", "Bagging_LOG", "train", sum(time_data) / len(time_data))


if __name__ == "__main__":
    LOG = LOG_Model()
    LOG.startTraining(True, True, True, True)
    LOG.startTesting()
