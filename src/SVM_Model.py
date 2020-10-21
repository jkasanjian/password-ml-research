
import numpy as np
#temporarily need 
from sklearn import metrics
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

        kernel = ["linear","rbf","poly"]
        C = [i for i in range(1,1000,10)]
        degree = [1,2]
        hyperparameteres = dict(kernel=kernel, C=C, degree=degree)

        for s in self.subjects:

            X_train, Y_train = get_train_data(s,all_data)
            svm_clf = SVC(gamma ='auto',decision_function_shape='ovo')
            clf = GridSearchCV(svm_clf, hyperparameteres, scoring='f1', n_jobs=-1)
            clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_SVM + s):
                os.makedirs(MODELS_SVM + s)
            dump(clf, MODELS_SVM + s + '/SVM.joblib')
            print("Finished training:",s)

    def svm_training_with_adaBoost(self, all_data = True):
        
        kernel = ["linear","rbf","poly"]
        C = [i for i in range(1,1000,10)]
        degree = [1,2]
        hyperparameteres = dict(kernel=kernel, C=C, degree=degree)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,True)
            ab_clf = AdaBoostClassifier(SVC(probability = True, gamma ='auto',decision_function_shape='ovo'))
            clf = GridSearchCV(ab_clf, hyperparameteres, scoring='f1', n_jobs=-1)
            clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_SVM + s):
                os.makedirs(MODELS_SVM + s)
            dump(clf, MODELS_SVM + s + '/Adaboost_SVM.joblib')
        

    def svm_training_with_Bagging(self, all_data = True):

        kernel = ["linear","rbf","poly"]
        C = [i for i in range(1,1000,10)]
        degree = [1,2]
        hyperparameteres = dict(kernel=kernel, C=C, degree=degree)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,True)
            bagging_clf = BaggingClassifier(SVC(gamma ='auto',decision_function_shape='ovo'))
            clf = GridSearchCV(bagging_clf, hyperparameteres, scoring='f1', n_jobs=-1)
            clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_SVM + s):
                os.makedirs(MODELS_SVM + s)
            dump(clf, MODELS_SVM + s + '/Bagging_SVM.joblib')

    

if __name__ == '__main__':
    # SVM = SVM_Model()
    # SVM.svm_training()
    # _, subjects = read_data()
    # for s in subjects:
    X_test, Y_test = get_test_data("s037",True)
    model = load_model("SVM","s037")
    Y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    print('SVM done training')
    X_test, Y_test = get_test_data("s055",True)
    model = load_model("SVM","s055")
    Y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    print('SVM done training')
    X_test, Y_test = get_test_data("s050",True)
    model = load_model("SVM","s030")
    Y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    print('SVM done training')