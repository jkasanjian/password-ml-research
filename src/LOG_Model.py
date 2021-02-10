###################
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump
from sklearn.model_selection import GridSearchCV
from logitboost import LogitBoost
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
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
MODELS_LOG_ALL = "./models/all_data/"
MODELS_LOG_BAL = "./models/balanced_data/"
MODELS_LOG_UB = "./models/unbalanced_data/"


class LOG_Model:

    def __init__(self):
        _, self.subjects = read_data()

    def startTraining(self, reg=True, logit=False, ada=False, Bagging=False, pca = False):
        print("\n\n\n--------------TRAINING LOG--------------\n")
        if reg:
            self.log_training(pca = pca)
        if logit:
            self.log_training_with_LogitBoost(pca = pca)
        if ada:
            self.log_training_with_Adaboost(pca = pca)
        if Bagging:
            self.log_training_with_Bagging(pca = pca)

    def startTesting(self,pca = False):
        model_names = [ "Adaboost_LOG"]
        for i in model_names:
            get_results(self.subjects, i, "LOG", pca)
    

    def log_training(self, balanced_data=False,pca = False):


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
            X_train, Y_train = get_train_data(s, balanced_data,pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            log_clf = LogisticRegression()
            grid_clf = GridSearchCV(log_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            grid_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if(balanced_data):
                directoryExist(MODELS_LOG_BAL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(grid_clf, MODELS_LOG_BAL + s + "/models/LOG.joblib")
                print("done", s)
            
            else:
                directoryExist(MODELS_LOG_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(grid_clf, MODELS_LOG_ALL + s + "/models/LOG.joblib")
                print("done", s)

        save_time_data("LOG", "LOG", "train", sum(time_data) / len(time_data))

    def log_training_with_Adaboost(self, balanced_data = False, pca =False):
        time_data = []
        p = "" if pca == False else "_pca"

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, balanced_data,pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            ab_clf = AdaBoostClassifier(LogisticRegression(), n_estimators=100)

            start_time = perf_counter()
            ab_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if(balanced_data):
                directoryExist(MODELS_LOG_BAL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(ab_clf, MODELS_LOG_BAL + s + "/models/Adaboost_LOG" + p +".joblib")
                print("done", s)
            else:
                directoryExist(MODELS_LOG_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(ab_clf, MODELS_LOG_ALL + s + "/models/Adaboost_LOG" + p +".joblib")
                print("done", s)

        save_time_data("LOG", "Adaboost_LOG", "train", sum(time_data) / len(time_data))

    def log_training_with_LogitBoost(self, balanced_data = False, pca = False):
        print(balanced_data,pca)
        time_data = []
        p = "" if pca == False else "_pca"

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, balanced_data,pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            lb_clf = LogitBoost(n_estimators=200, bootstrap=True)

            start_time = perf_counter()
            lb_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if(balanced_data):
                directoryExist(MODELS_LOG_BAL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(lb_clf, MODELS_LOG_BAL + s + "/models/LBoost_LOG" + p +".joblib")
                print("done", s)
            
            else:
                directoryExist(MODELS_LOG_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(lb_clf, MODELS_LOG_ALL + s + "/models/LBoost_LOG" + p + ".joblib")
                print("done", s)

        save_time_data("LOG", "LBoost_LOG", "train", sum(time_data) / len(time_data))

    def log_training_with_Bagging(self, balanced_data = False, pca = False):
        time_data = []
        p = "" if pca == False else "_pca"

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, balanced_data,pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            bagging_clf = BaggingClassifier(LogisticRegression(), n_estimators=15)

            start_time = perf_counter()
            bagging_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if(balanced_data):
                directoryExist(MODELS_LOG_BAL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(bagging_clf, MODELS_LOG_BAL + s + "/models/Bagging_LOG" + p + ".joblib")
                print("done", s)
            
            else:
                directoryExist(MODELS_LOG_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(bagging_clf, MODELS_LOG_ALL + s + "/models/Bagging_LOG" + p + ".joblib")
                print("done", s)

        save_time_data("LOG", "Bagging_LOG", "train", sum(time_data) / len(time_data))


if __name__ == "__main__":
    LOG = LOG_Model()
    # LOG.startTraining(False, True, True, True, True)
    LOG.startTesting(pca = True)
