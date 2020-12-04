import os
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from Helper import get_train_data, read_data, get_results

DATA_SPLIT = 'data/split/'
MODELS_SVM = 'src/models/all_data/'
MODELS_SVM_UB = 'src/models/unbalanced_data/'



class SVM_Model:

    def __init__(self, balanced = False):
        _, self.subjects = read_data()

    def startTraining(self,reg = True, ada = False, Bagging = False):
        if(reg):
            self.svm_training()
        if(ada):
            self.svm_training_with_adaBoost()
        if(Bagging):
            self.svm_training_with_Bagging()
    
    def startTesting(self):
        model_names = ["Adaboost_SVM","Bagging_SVM","SVM"]
        for model in model_names:
            get_results(self.subjects, model,"SVM")


    def svm_training(self, all_data = True): 
        #SVM with gridsearch is still as optimal as bagging, bagging still optimizes even if its very low margins
        kernel = ["rbf","poly","sigmoid"]

        C = [.001, .01, .1, 1, 10, 100, 500, 1000]
        degree = [i for i in range(1,7)]
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
        for s in self.subjects:
            
            X_train, Y_train = get_train_data(s,True)
            svm_clf = SVC(probability = True, gamma ='auto',decision_function_shape='ovo')
            ada_clf = AdaBoostClassifier(svm_clf)
            ada_clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_SVM + s):
                os.makedirs(MODELS_SVM + s)
            dump(ada_clf, MODELS_SVM + s + '/Adaboost_SVM.joblib')
            print("finished:",s)
        

    def svm_training_with_Bagging(self, all_data = True):
        print("----------------- Performing Bagging Training ------------------")

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,True)
            sv_clf = SVC(probability = True, gamma ='auto', decision_function_shape='ovo')
            bag_clf = BaggingClassifier(sv_clf.best_estimator_)
            bag_clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_SVM + s):
                os.makedirs(MODELS_SVM + s)
            dump(bag_clf, MODELS_SVM + s + '/Bagging_SVM.joblib')
            print("Finished subject",s)

    

if __name__ == '__main__':
    
    SVM = SVM_Model()
    # SVM.startTraining(False,True,True)
    SVM.startTesting()
    
        
