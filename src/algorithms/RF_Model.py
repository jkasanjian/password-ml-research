import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from IPython.display import Image
from sklearn.tree import export_graphviz
from subprocess import check_call
from time import perf_counter
from helper import (
    get_train_data,
    read_data,
    save_time_data,
)
from constants import MODELS_DIR


class RF_Model:
    def __init__(self, balanced=False):
        _, self.subjects = read_data()


    def startTraining(self, model_variation, pca, ratio):
        # Trains one type of model given flags
        if model_variation == "RF_Grid":
            self.rf_training_with_Grid(pca, ratio)
        elif model_variation == "RF_Adaboost":
            self.rf_training_with_Adaboost(pca = pca, ratio = ratio, all_data = all_data)
        elif model_variation == "RF_Bagging":
            self.rf_training_with_Bagging(pca = pca, ratio = ratio, all_data = all_data)


    def rf_training_with_Grid(self, pca, ratio):
        print("~~~~~~~~~~Starting RF Gridsearch~~~~~~~~~~")

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
            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            rf_clf = RandomForestClassifier()
            clf = GridSearchCV(rf_clf, hyperparameters, scoring="f1", n_jobs=-1)

            start_time = perf_counter()
            clf.fit(X_train, Y_train)
            end_time = perf_counter()

            fname = MODELS_DIR + ratio + "/" + s + "/models/RF_Grid"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(clf, fname)

            print("Done with RF_Grid for", s)

        save_time_data(ratio, "RF", "RF_Grid", pca, "train", sum(time_data) / len(time_data))


    def rf_training_with_Adaboost(self, pca, ratio):
        print("~~~~~~~~~~Starting RF Adaboost~~~~~~~~~~")

        time_data = []

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, ratio, pca)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)

            rf_clf = RandomForestClassifier(
                n_estimators=60, max_depth=10, criterion="gini"
            )
            ada_clf = AdaBoostClassifier(rf_clf, n_estimators=100, learning_rate=1)

            start_time = perf_counter()
            ada_clf.fit(X_train, Y_train)
            end_time = perf_counter()

            fname = MODELS_DIR + ratio + "/" + s + "/models/RF_Adaboost"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(ada_clf, fname)

            print("Done with RF_Adaboost for", s)

        save_time_data(ratio, "RF", "RF_Adaboost", pca, "train", sum(time_data) / len(time_data))


    def rf_training_with_Bagging(self, pca, ratio):
        print("~~~~~~~~~~Starting RF Bagging~~~~~~~~~~")
        
        time_data = []

        for s in self.subjects:
            X_train, Y_train = get_train_data(s, ratio, pca)
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

            fname = MODELS_DIR + ratio + "/" + s + "/models/RF_Bagging"
            if pca:
                fname += "_pca"
            fname += ".joblib"

            total_time = end_time - start_time
            time_data.append(total_time)
            dump(bagging_clf, fname)

            print("Done with RF_Bagging for", s)

        save_time_data(ratio, "RF", "RF_Bagging", pca, "train", sum(time_data) / len(time_data))


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