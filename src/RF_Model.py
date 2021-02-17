# temporarily need
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from IPython.display import Image
from sklearn.tree import export_graphviz
from subprocess import check_call
from Helper import (
    get_results,
    get_train_data,
    read_data,
    directoryExist,
    get_results,
    save_time_data,
)
from time import perf_counter


DATA_SPLIT = "data/split/"
MODELS_RF_ALL = "./models/all_data/"
MODELS_RF_BAL= "./models/pos-"


class RF_Model:
    def __init__(self, balanced=False):
        _, self.subjects = read_data()

    def startTraining(self, grid=False, ada=False, Bagging=False, all_data = False, pca = False, ratio = "10"):
        print("\n\n\n--------------TRAINING RF--------------\n")
        if grid:
            self.rf_training_with_gridSearch(pca = pca, ratio = ratio, all_data = all_data)
        if ada:
            self.rf_training_with_adaBoost(pca = pca, ratio = ratio, all_data = all_data)
        if Bagging:
            self.rf_training_with_Bagging(pca = pca, ratio = ratio, all_data = all_data)

    def startTesting(self, pca = False, all_data = False, ratio = "10"):
        model_names = ["RF","Adaboost_RF", "Bagging_RF"]
        for i in model_names:
            get_results(self.subjects, i, "RF",pca, all_data , ratio)

    # Generates a Random Forest regression model for each subject
    def rf_training_with_gridSearch(self, all_data=True,pca = False,ratio = "10"):
        p = "" if pca == False else "_pca"
        time_data = []

        n_estimators = [int(x) for x in np.linspace(start=10, stop=300, num=10)]
        max_features = ["auto", "sqrt"]
        max_depth = list(range(5, 55, 10))
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        hyperparameters = dict(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
        )

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data,pca = False, ratio = "10")
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            rf_clf = RandomForestClassifier()
            clf = GridSearchCV(rf_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if(all_data):
                directoryExist(MODELS_RF_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(clf, MODELS_RF_ALL + s + "/models/RF" + p + ".joblib")
                print("Finished training:", s)

            else: 
                directoryExist(MODELS_RF_BAL + ratio + "/"  + s + "/models/")
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(clf, MODELS_RF_BAL + ratio + "/" + s + "/models/RF" + p + ".joblib")
                print("Finished training:", s)

        # save_time_data("RF", "RF", "train", sum(time_data) / len(time_data))

    def rf_training_with_adaBoost(self, all_data=True,pca = False, ratio="10"):
        time_data = []
        p = "" if pca == False else "_pca"


        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data,pca,ratio)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            rf_clf = RandomForestClassifier(
                n_estimators=60, max_depth=10, criterion="gini"
            )
            ada_clf = AdaBoostClassifier(rf_clf, n_estimators=100, learning_rate=1)

            start_time = perf_counter()
            ada_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if all_data:

                directoryExist(MODELS_RF_ALL + s)
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(ada_clf, MODELS_RF_ALL + s + "/models/Adaboost_RF" + p +".joblib")
                print(s, "Finished")

            else: 
                directoryExist(MODELS_RF_BAL + ratio + "/"  + s + "/models/")
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(ada_clf, MODELS_RF_BAL + ratio + "/" + s + "/models/Adaboost_RF" + p +".joblib")
                print(s, "Finished")



        # save_time_data("RF", "Adaboost_RF", "train", sum(time_data) / len(time_data))

    def rf_training_with_Bagging(self, all_data=True,pca = False, ratio = "10"):
        time_data = []
        p = "" if pca == False else "_pca"

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, all_data,pca,ratio)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            bagging_clf = BaggingClassifier(
                RandomForestClassifier(
                    n_estimators=100, max_depth=10, criterion="gini"
                ),
                n_estimators=10,
            )

            start_time = perf_counter()
            bagging_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            if all_data: 
                
                directoryExist(MODELS_RF_ALL + s)
                # print("Finished:",s[2:])
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(bagging_clf, MODELS_RF_ALL + s + "/models/Bagging_RF" + p+ ".joblib")
                print("Finished training:", s)

            else:
                directoryExist(MODELS_RF_BAL + ratio + "/"  + s + "/models/")
                # print("Finished:",s[2:])
                total_time = end_time - start_time
                time_data.append(total_time)
                dump(bagging_clf, MODELS_RF_BAL + ratio + "/"  + s + "/models/Bagging_RF" + p+ ".joblib")
                print("Finished training:", s)
                
            

        # save_time_data("RF", "Bagging_RF", "train", sum(time_data) / len(time_data))

    # TODO finish implementing visual for each users tree

    # def visualize_tree(clf, s):
    # #Creates a graphics diagram of a tree from the random forest
    #     estimator = clf.estimators_[5]

    #     export_graphviz(estimator, out_file=TREE_GRAPHS+s+'tree.dot',
    #                 feature_names = get_features(),
    #                 class_names = ['genuine user', 'imposter'],
    #                 rounded = True, proportion = False,
    #                 precision = 2, filled = True)

    #     check_call(['dot', '-Tpng', TREE_GRAPHS+s+'tree.dot', '-o', TREE_GRAPHS+s+'tree.png', '-Gdpi=600'])
    #     Image(filename=TREE_GRAPHS+s+'tree.png')
    #     remove(TREE_GRAPHS+s+'tree.dot')


if __name__ == "__main__":
    RF = RF_Model()
    ratios = [20,30,40, 60, 70, 80, 90]
    for r in ratios:
        print("Starting training for ratio-",r)
        RF.startTraining(True, True, True, True, pca = True, ratio = str(r), all_data = False)
        RF.startTesting(pca = True, all_data = False, ratio = str(r))

        RF.startTraining(True, True, True, True, pca = False, ratio = str(r), all_data = False)
        RF.startTesting(pca = False, all_data = False, ratio = str(r))

    RF.startTraining(True, True, True, True, pca = False, ratio = str(r), all_data = False)
    RF.startTesting(pca = True, all_data = True, ratio = str(r))