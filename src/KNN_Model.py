#temporarily need 
from sklearn import metrics
###################
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from Helper import get_test_data, get_train_data, read_data, load_model

DATA_SPLIT = 'data/split/'
MODELS_KNN = 'src/models/all_data/'
MODELS_KNN_UB = 'src/models/unbalanced_data/'



class KNN_Model:

    def __init__(self, balanced = False):
        _, self.subjects = read_data()

    def startTraining(self,reg = True, ada = False, Bagging = False):
        if(reg):
            knn_training()
        if(ada):
            knn_training_with_adaBoost()
        if(Bagging):
            knn_training_with_Bagging()
    
    def startTesting(self):
        #TODO begin testing here
        print()


    def knn_training(self, all_data = True): 
    #Calculates and saves the best hyperparameters for each subject's KNN model
       
        leaf_size = list(range(1,50))
        n_neighbors = list(range(1,30))
        p = [1,2]
        hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            knn_clf = KNeighborsClassifier(algorithm= 'brute', metric = 'minkowski')
            clf = GridSearchCV(knn_clf, hyperparameters, scoring='f1', n_jobs=-1)
            clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_KNN + s):
                os.makedirs(MODELS_KNN + s)
            dump(clf, MODELS_KNN + s + '/KNN.joblib')


        
    def knn_training_with_adaBoost(self, all_data = True):

        leaf_size = list(range(1,50))
        n_neighbors = list(range(1,30))
        p = [1,2]
        hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            ab_clf = AdaBoostClassifier(KNeighborsClassifier(algorithm= 'brute', metric = 'minkowski'))
            clf = GridSearchCV(ab_clf, hyperparameters, scoring='f1', n_jobs=-1)
            clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_KNN + s):
                os.makedirs(MODELS_KNN + s)
            dump(clf, MODELS_KNN + s + '/Adaboost_KNN.joblib')

        

    def knn_training_with_Bagging(self, all_data = True):

        leaf_size = list(range(1,50))
        n_neighbors = list(range(1,30))
        p = [1,2]
        hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            bagging_clf = BaggingClassifier(KNeighborsClassifier(algorithm= 'brute', metric = 'minkowski'))
            clf = GridSearchCV(bagging_clf, hyperparameters, scoring='f1', n_jobs=-1)
            clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_KNN + s):
                os.makedirs(MODELS_KNN + s)
            dump(clf, MODELS_KNN + s + '/Bagging_KNN.joblib')


# if __name__ == "__main__":
    KNN = KNN_Model()
    KNN.KNN_training()
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