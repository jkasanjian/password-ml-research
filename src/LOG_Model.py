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
MODELS_LOG_BAL = "./models/pos-"


class LOG_Model:

    def __init__(self):
        _, self.subjects = read_data()

    def startTraining(self, grid=True, logit=False, ada=False, Bagging=False, all_data = False, pca = False, ratio = "10"):
        print("\n\n\n--------------TRAINING LOG--------------\n")
        if grid:
            self.log_training_gridSearch(pca = pca, ratio= ratio, all_data = all_data)
        if logit:
            self.log_training_with_LogitBoost(pca = pca, ratio= ratio, all_data = all_data)
        if ada:
            self.log_training_with_Adaboost(pca = pca, ratio = ratio, all_data = all_data)
        if Bagging:
            self.log_training_with_Bagging(pca = pca,ratio = ratio, all_data = all_data)

    def startTesting(self,pca = False,all_data = False, ratio = "10"):
        model_names = ["Adaboost_LOG", "Bagging_LOG","LBoost_LOG"]
        for i in model_names:
            get_results(self.subjects, i, "LOG", pca = pca, all_data = all_data, ratio = ratio)
    

    def log_training_gridSearch(self, all_data=False,pca = False, ratio = "10"):
        p = "" if pca == False else "_pca"
        print("~~~~~~~~~~Starting Gridsearch~~~~~~~~~~")

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
            X_train, Y_train = get_train_data(s, all_data,pca, ratio = ratio)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            log_clf = LogisticRegression()
            grid_clf = GridSearchCV(log_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            grid_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if(all_data):
                directoryExist(MODELS_LOG_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(grid_clf, MODELS_LOG_ALL + s + "/models/LOG" + p + ".joblib")
                print("done", s)
            
            else:
                directoryExist(MODELS_LOG_BAL + ratio + "/" + s + "/models/")
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(grid_clf, MODELS_LOG_BAL +  ratio + "/" + s + "/models/LOG" + p + ".joblib")
                print("done", s)

        # save_time_data("LOG", "LOG", "train", sum(time_data) / len(time_data))

   
   
    def log_training_with_Adaboost(self, all_data = False, pca =False, ratio = "10"):
        time_data = []
        p = "" if pca == False else "_pca"

        print("~~~~~~~~~~Starting Adaboost~~~~~~~~~~")

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data , pca, ratio = ratio)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            ab_clf = AdaBoostClassifier(LogisticRegression(), n_estimators=100)

            start_time = perf_counter()
            ab_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if(all_data):
                directoryExist(MODELS_LOG_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(ab_clf, MODELS_LOG_ALL + s + "/models/Adaboost_LOG" + p +".joblib")
                print("done", s)
            else:
                directoryExist(MODELS_LOG_BAL +  ratio + "/" + s + "/models/" )
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(ab_clf, MODELS_LOG_BAL +  ratio + "/" + s + "/models/Adaboost_LOG" + p +".joblib")
                print("done", s)

        # save_time_data("LOG", "Adaboost_LOG", "train", sum(time_data) / len(time_data))

    def log_training_with_LogitBoost(self, all_data = False, pca = False, ratio = "10"):
        print(all_data,pca)
        time_data = []
        p = "" if pca == False else "_pca"


        print("~~~~~~~~~~Starting LogitBoost~~~~~~~~~~")

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data,pca, ratio = ratio)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            lb_clf = LogitBoost(n_estimators=200, bootstrap=True)
            print(X_train.shape)
            start_time = perf_counter()
            lb_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if(all_data):
                directoryExist(MODELS_LOG_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(lb_clf, MODELS_LOG_ALL + s + "/models/LBoost_LOG" + p +".joblib")
                print("done", s)
            
            else:
                directoryExist(MODELS_LOG_BAL + ratio + "/" + s + "/models/")
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(lb_clf, MODELS_LOG_BAL +  ratio + "/" + s + "/models/LBoost_LOG" + p + ".joblib")
                print("done", s)

        # save_time_data("LOG", "LBoost_LOG", "train", sum(time_data) / len(time_data))

    
    def log_training_with_Bagging(self, all_data = False, pca = False, ratio = "10"):
        time_data = []
        p = "" if pca == False else "_pca"

        print("~~~~~~~~~~Starting Bagging~~~~~~~~~~")


        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data,pca,ratio = ratio)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            bagging_clf = BaggingClassifier(LogisticRegression(), n_estimators=15)

            start_time = perf_counter()
            bagging_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if(all_data):
                directoryExist(MODELS_LOG_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(bagging_clf, MODELS_LOG_ALL + s + "/models/Bagging_LOG" + p + ".joblib")
                print("done", s)
            
            else:
                directoryExist(MODELS_LOG_BAL + ratio + "/" + s + "/models/")
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(bagging_clf, MODELS_LOG_BAL + ratio + "/" + s + "/models/Bagging_LOG" + p + ".joblib")
                print("done", s)

        # save_time_data("LOG", "Bagging_LOG", "train", sum(time_data) / len(time_data))


if __name__ == "__main__":
    LOG = LOG_Model()
    # for r in ratios:
    #     LOG.startTraining(False, True, True, True, ratio = str(r), pca = True)
    #     LOG.startTesting (all_data = False, ratio = str(r), pca = True)

    ratios = [10,20,30,40,60,70,80,90]
    # for r in ratios:
        
    #     LOG.startTraining(True, True, True, True, ratio = str(r), pca = True)
    #     LOG.startTesting (all_data = True, ratio = str(r), pca = True)

    #     LOG.startTraining(True, True, True, True, ratio = str(r), pca = True)
    #     LOG.startTesting (all_data = False, ratio = str(r), pca = True)

    # LOG.startTraining(False, True, True, True, pca = True, all_data = True)
    LOG.startTesting (all_data = True, pca = True)
