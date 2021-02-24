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
MODELS_KNN_ALL = "./models/all_data/"
MODELS_KNN_BAL = "./models/pos-/"



class KNN_Model:
    def __init__(self):
        _, self.subjects = read_data()

    def startTraining(self, grid=True, ada=False, bagging=False, all_data = False, pca = False, ratio = "10"):
        print("\n\n\n--------------TRAINING KNN--------------\n")
        if grid:
            self.knn_training(pca = pca, ratio= ratio, all_data = all_data)
        if ada:
            self.knn_training_with_adaBoost(pca= pca, ratio= ratio, all_data = all_data)
        if bagging:
            self.knn_training_with_Bagging(pca = pca, ratio= ratio, all_data = all_data)

    def startTesting(self,pca =False, all_data = False, ratio = "10"):
        model_names = ["KNN","Bagging_KNN", "Adaboost_KNN"]
        for model in model_names:
            get_results(self.subjects, model, "KNN",pca, all_data, ratio = ratio )

    

    def knn_training(self, all_data=True, pca = False, ratio = "10"):
        # Calculates and saves the best hyperparameters for each subject's KNN model
        p = "" if pca == False else "_pca"
        time_data = []

        leaf_size = list(range(1, 50))
        n_neighbors = list(range(1, 30))
        p = [1, 2]
        hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data = all_data ,pca = pca , ratio = ratio)
            knn_clf = KNeighborsClassifier(algorithm="brute", metric="minkowski")
            clf = GridSearchCV(knn_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if all_data: 
                directoryExist(MODELS_KNN_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                print("Finished training:", s)
                dump(clf, MODELS_KNN_ALL + s + "/models/KNN" + p + ".joblib")

            else:
                directoryExist(MODELS_KNN_BAL + ratio + "/" + s + "/models")
                total_time = end_time - start_time
                time_data.append(total_time)
                print("Finished training:", s)
                dump(clf, MODELS_KNN_BAL + ratio + "/" + s + "/models/KNN " + p + ".joblib")

        p_res = "pca_on" if pca == True else "pca_off"
        save_time_data("KNN", "KNN_grid", "train", sum(time_data) / len(time_data))

    def knn_training_with_adaBoost(self, all_data=True, pca = False, ratio = "10"):
        p = "" if pca == False else "_pca"
        time_data = []


        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data = all_data,pca = pca, ratio = ratio )
            ab_clf = AdaBoostClassifier(
                KNeighborsClassifier(algorithm="brute", metric="minkowski")
            )
           

            start_time = perf_counter()
            ab_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if all_data: 
                directoryExist(MODELS_KNN_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                print("Finished training:", s)
                dump(ab_clf, MODELS_KNN_ALL + s + "/models/Adaboost_KNN " + p + ".joblib")
            
            else:
                directoryExist(MODELS_KNN_BAL + ratio + "/" + s + "/models")
                total_time = end_time - start_time
                time_data.append(total_time)
                print("Finished training:", s)
                dump(ab_clf, MODELS_KNN_BAL + ratio + "/" + s + "/models/Adaboost_KNN " + p + ".joblib")

        p_res = "pca_on" if pca == True else "pca_off"
        save_time_data("KNN", "Adaboost_KNN", p_res, "train", sum(time_data) / len(time_data))

    def knn_training_with_Bagging(self, all_data=True, pca = False, ratio = "10"):
        p = "" if pca == False else "_pca"
        time_data = []
    

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data = all_data, pca = pca, ratio = ratio)
            bagging_clf = BaggingClassifier(
                KNeighborsClassifier(algorithm="brute", metric="minkowski")
            )

            start_time = perf_counter()
            bagging_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if all_data:
                directoryExist(MODELS_KNN_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                print("Finished training:", s)
                dump(bagging_clf, MODELS_KNN_ALL+ s + "/models/Bagging_KNN" + p + ".joblib")
            
            else:
                directoryExist(MODELS_KNN_BAL + ratio + "/" + s + "/models")
                total_time = end_time - start_time
                time_data.append(total_time)
                print("Finished training:", s)
                dump(bagging_clf, MODELS_KNN_BAL+ ratio + "/" + s + "/models/Bagging_KNN" + p + ".joblib")

        p_res = "pca_on" if pca == True else "pca_off"
        save_time_data("KNN", "Bagging_KNN", p_res, "train", sum(time_data) / len(time_data))


if __name__ == "__main__":
    KNN = KNN_Model()
    ratios = [10,20,30,40,60,70,80,90]
    for r in ratios:
        
        KNN.startTraining(False, True, True, all_data = False, ratio = str(r), pca = False)
        KNN.startTesting (all_data = False, ratio = str(r), pca = False)

        KNN.startTraining(False, True, True, all_data = False, ratio = str(r), pca = True)
        KNN.startTesting (all_data = False, ratio = str(r), pca = True)

    KNN.startTraining(False, True, True, all_data = True, ratio = str(r), pca = False)
    KNN.startTesting (all_data = True, ratio = str(r), pca = True)
    
    # KNN.startTesting(pca = True)
