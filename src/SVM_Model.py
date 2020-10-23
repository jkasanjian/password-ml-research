
import numpy as np
#temporarily need 
from sklearn import metrics
###################
import os
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from Helper import get_test_data, get_train_data, read_data, load_model

DATA_SPLIT = 'data/split/'
MODELS_SVM = 'src/models/all_data/'
MODELS_SVM_UB = 'src/models/unbalanced_data/'



class SVM_Model:

    def __init__(self, balanced = False):
        _, self.subjects = read_data()

    def startTraining(self,reg = True, ada = False, Bagging = False):
        if(reg):
            svm_training()
        if(ada):
            svm_training_with_adaBoost()
        if(Bagging):
            svm_training_with_Bagging()
    
    def startTesting(self):
        #TODO begin testing here
        print()


    def svm_training(self, all_data = True): 
        #SVM with gridsearch is still as optimal as bagging, bagging still optimizes even if its very low margins
        kernel = ["linear","rbf","poly"]
        C = [i for i in range(1,1000,10)]
        degree = [1,2]
        hyperparameters = dict(kernel=kernel, C=C, degree=degree)

        for s in self.subjects:

            X_train, Y_train = get_train_data(s,all_data)
            svm_clf = SVC(gamma ='auto',decision_function_shape='ovo')
            clf = GridSearchCV(svm_clf, hyperparameters, scoring='f1', n_jobs=-1)
            clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_SVM + s):
                os.makedirs(MODELS_SVM + s)
            dump(clf, MODELS_SVM + s + '/SVM.joblib')
            print("Finished training:",s)

    def svm_training_with_adaBoost(self, all_data = True):
        
        print("----------------- Performing Adaboost Training ------------------")
        #Adaboost with the best estimator from grid search (SVC estimator) sucks
        #Also why does it make the accuracies weaker?
        #Can we boost something from bagging?
        #maybe just pass in adaboost with just SVM alone 
        kernel = ["linear","rbf","poly"]
        C = [i for i in range(1,1000,10)]
        degree = [1,2]
        hyperparameters = dict(kernel = kernel, C = C, degree=degree)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,True)
            
            svm_clf = SVC(probability = True, gamma ='auto',decision_function_shape='ovo')
            #clf = GridSearchCV(svm_clf, hyperparameters, scoring='f1', n_jobs=-1)
            #It needs smaple weights 
            ada_clf = AdaBoostClassifier(svm_clf)
            ada_clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_SVM + s):
                os.makedirs(MODELS_SVM + s)
            dump(ada_clf, MODELS_SVM + s + '/Adaboost_SVM.joblib')
            print("finished:",s)
        

    def svm_training_with_Bagging(self, all_data = True):
        print("----------------- Performing Bagging Training ------------------")
        kernel = ["linear","rbf","poly"]
        C = [i for i in range(1,1000,10)]
        degree = [1,2]
        n_estimators = [1,10,20,100]
        hyperparameters = dict(kernel=kernel, C=C, degree=degree)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,True)
            sv = SVC(probability = True, gamma ='auto', decision_function_shape='ovo')
            clf = GridSearchCV(sv, hyperparameters, scoring='f1', n_jobs=-1)
            clf.fit(X_train, Y_train)
            Bag = BaggingClassifier(clf.best_estimator_)
            Bag.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_SVM + s):
                os.makedirs(MODELS_SVM + s)
            dump(Bag, MODELS_SVM + s + '/Bagging_SVM.joblib')
            print("Finished subject",s)

    

if __name__ == '__main__':
    
    SVM = SVM_Model()
    # SVM.svm_training(True)
    for s in SVM.subjects:
        X_test, Y_test = get_test_data(s,True)
        model = load_model("SVM",s)
        Y_pred = model.predict(X_test)
        print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

    # X_test, Y_test = get_test_data("s037",True)
    # model = load_model("SVM","s037")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('SVM done training')
    # X_test, Y_test = get_test_data("s055",True)
    # model = load_model("SVM","s055")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('SVM done training')
    # X_test, Y_test = get_test_data("s050",True)
    # model = load_model("SVM","s030")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('SVM done training')