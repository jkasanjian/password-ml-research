import os
from sklearn.svm import SVC
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
MODELS_SVM_ALL = "./models/all_data/"
MODELS_SVM_BAL = "./models/pos-"


class SVM_Model:
    def __init__(self, balanced=False):
        _, self.subjects = read_data()

    def startTraining(self, grid=True, ada=False, Bagging=False, pca = False, ratio = "10" , all_data = False):
        print("\n\n\n--------------TRAINING SVM--------------\n")
        if grid:
            self.svm_training(pca = pca, ratio = ratio, all_data = all_data)
        if ada:
            self.svm_training_with_adaBoost(pca = pca, ratio = ratio, all_data = all_data)
        if Bagging:
            self.svm_training_with_Bagging(pca = pca, ratio = ratio, all_data = all_data)

    def startTesting(self, pca = False, all_data = True, ratio = "10"):
        model_names = [ "Adaboost_SVM", "Bagging_SVM"]
        for model in model_names:
            get_results(self.subjects, model, "SVM",pca, all_data, ratio)

    def svm_training(self, all_data=True, pca = False, ratio = "10"):
        p = "" if pca == False else "_pca"
        time_data = []

        # SVM with gridsearch is still as optimal as bagging, bagging still optimizes even if its very low margins
        kernel = ["linear", "rbf", "poly", "sigmoid"]

        C = [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]
        degree = [i for i in range(1, 7)]
        hyperparameters = dict(kernel=kernel, C=C, degree=degree)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data,pca,ratio)
            svm_clf = SVC(gamma="auto", decision_function_shape="ovo", probability=True)
            clf = GridSearchCV(svm_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if all_data:
                directoryExist(MODELS_SVM_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(clf, MODELS_SVM_ALL + s + "/models/SVM" + p + ".joblib")
                print("Finished training:", s)

            else: 
                directoryExist(MODELS_SVM_BAL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(clf, MODELS_SVM_BAL + ratio + "/" + s + "/models/SVM" + p + ".joblib")
                print("Finished training:", s)

        # save_time_data("SVM", "SVM", "train", sum(time_data) / len(time_data))

    def svm_training_with_adaBoost(self, all_data=True, pca = False, ratio = "10"):
        p = "" if pca == False else "_pca"
        time_data = []
        print("----------------- Performing Adaboost Training ------------------")
        # Adaboost with the best estimator from grid search (SVC estimator) sucks
        # Also why does it make the accuracies weaker?
        # Can we boost something from bagging?
        # maybe just pass in adaboost with just SVM alone
        for s in self.subjects:

            X_train, Y_train = get_train_data(s, all_data, pca, ratio)
            svm_clf = SVC(probability=True, gamma="auto", decision_function_shape="ovo")
            ada_clf = AdaBoostClassifier(svm_clf, n_estimators=10)

            start_time = perf_counter()
            ada_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if all_data:
                directoryExist(MODELS_SVM_ALL+ s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(ada_clf, MODELS_SVM_ALL + s + "/models/Adaboost_SVM" + p + ".joblib")
                print("finished:", s)

            else:
                directoryExist(MODELS_SVM_BAL+ s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(ada_clf, MODELS_SVM_BAL + ratio + "/" + s + "/models/Adaboost_SVM" + p + ".joblib")
                print("finished:", s)

        # save_time_data("SVM", "Adaboost_SVM", "train", sum(time_data) / len(time_data))


    def svm_training_with_Bagging(self, all_data=True, pca = False, ratio = "10"):
        p = "" if pca == False else "_pca"
        time_data = []
        print("----------------- Performing Bagging Training ------------------")

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data, pca, ratio)
            sv_clf = SVC(probability=True, gamma="auto", decision_function_shape="ovo")
            bag_clf = BaggingClassifier(sv_clf, n_estimators=100)

            start_time = perf_counter()
            bag_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if all_data:
                directoryExist(MODELS_SVM_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(bag_clf, MODELS_SVM_ALL + s + "/Bagging_SVM" + p + ".joblib")
                print("Finished subject", s)

            else:
                directoryExist(MODELS_SVM_BAL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(bag_clf, MODELS_SVM_BAL + ratio + "/" + s + "/Bagging_SVM" + p + ".joblib")
                print("Finished subject", s)

        # save_time_data("SVM", "Bagging_SVM", "train", sum(time_data) / len(time_data))


if __name__ == "__main__":
    SVM = SVM_Model()
    ratios = [10, 20, 30, 40, 60, 70, 80, 90]
    for r in ratios:
        
        SVM.startTraining(True, True, True, True, pca = False, ratio = str(r), all_data = False)
        SVM.startTesting(pca = False, all_data = False, ratio = str(r))

        SVM.startTraining(True, True, True, True, pca = True, ratio = str(r), all_data = False)
        SVM.startTesting(pca = True, all_data = False, ratio = str(r))

    SVM.startTraining(True, True, True, True, pca = True, ratio = str(r), all_data = True)
    SVM.startTesting(pca = True, all_data = True, ratio = str(r))
    # SVM.startTraining(False, True, True,True)

   
