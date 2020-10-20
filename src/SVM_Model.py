from sklearn.svm import SVC
import numpy as np
from os import mkdir, remove
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from Helper import get_test_data, get_train_data, read_data

DATA_SPLIT = 'data/split/'
MODELS_SVM = 'models/svm/'


class SVM_Model:

    def __init__(self):
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


    def svm_training(self): 

        kernel = ["linear","rbf","poly"]
        C = [i for i in range(1,1000,10)]
        degree = [1,2]
        hyperparameteres = dict(kernel=kernel, C=C, degree=degree)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,True)
            #self.X_test, self.Y_test = get_test_data(s,True)
            svm_clf = SVC(gamma ='auto',decision_function_shape='ovo')
            clf = GridSearchCV(svm_clf, hyperparameteres, scoring='f1', n_jobs=-1)
            clf.fit(X_train, Y_train)
            dump(clf, MODELS_SVM + s + '.joblib')
            print("Done Training")

    def svm_training_with_adaBoost(self):
        
        kernel = ["linear","rbf","poly"]
        C = [i for i in range(1,1000,10)]
        degree = [1,2]
        hyperparameteres = dict(kernel=kernel, C=C, degree=degree)

        for s in self.subjects:
            ab_clf = AdaBoostClassifier(SVC(probability = True, gamma ='auto',decision_function_shape='ovo'))
            clf = GridSearchCV(ab_clf, hyperparameteres, scoring='f1', n_jobs=-1)
            clf.fit(self.X_train, self.Y_train)
            dump(clf, MODELS_SVM + s + '.joblib')
        

    def svm_training_with_Bagging(self):

        kernel = ["linear","rbf","poly"]
        C = [i for i in range(1,1000,10)]
        degree = [1,2]
        hyperparameteres = dict(kernel=kernel, C=C, degree=degree)

        for s in self.subjects:
            bagging_clf = BaggingClassifier(SVC(gamma ='auto',decision_function_shape='ovo'))
            clf = GridSearchCV(bagging_clf, hyperparameteres, scoring='f1', n_jobs=-1)
            clf.fit(self.X_train, self.Y_train)
            dump(clf, MODELS_SVM + s + '.joblib')

    

if __name__ == '__main__':
    SVM = SVM_Model()
    SVM.svm_training()
    print('SVM done training')