#temporarily need 
from sklearn import metrics
###################
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump
from sklearn.model_selection import GridSearchCV
from logitboost import LogitBoost
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from Helper import directoryExist, get_train_data, read_data, get_results, get_test_data

DATA_SPLIT = 'data/split/'
MODELS_LOG = 'src/models/all_data/'
MODELS_LOG_UB = 'src/models/unbalanced_data/'



class LOG_Model:

    def __init__(self, balanced = False):
        _, self.subjects = read_data()

    def startTraining(self,reg = True, logit = False, ada = False, Bagging = False):
        if(reg):
            self.log_training()
        if(logit):
            self.log_training_with_LogitBoost()
        if(ada):
            self.log_training_with_Adaboost()
        if(Bagging):
            self.log_training_with_Bagging()
    
    def startTesting(self):
        model_names = ["Adaboost_LOG","LBoost_LOG", "Bagging_LOG"]
        for i in model_names:
            get_results(LOG.subjects,i,"LOG")


    def log_training(self, all_data = True): 
    #Calculates and saves the best hyperparameters for each subject's
    #Logistic Regression model
        penalty = ['l1', 'l2', 'elasticnet', 'none']
        C = np.logspace(-4, 4, 20)
        solver = ['lbfgs','newton-cg','liblinear','sag','saga']
        max_iter = [100, 1000, 2500, 5000]
        hyperparameters = dict(penalty=penalty, C=C, solver=solver, max_iter=max_iter)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            log_clf = LogisticRegression(tol=.001)
            grid_clf = GridSearchCV(log_clf, hyperparameters, scoring='f1', n_jobs=-1)
            grid_clf.fit(X_train, Y_train)
            directoryExist(MODELS_LOG + s)
            dump(grid_clf, MODELS_LOG + s + '/LOG.joblib')
            print("done",s)


    def log_training_with_Adaboost(self,all_data = True):
        
        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            ab_clf = AdaBoostClassifier(LogisticRegression(tol =.001),n_estimators= 100)
            ab_clf.fit(X_train, Y_train)
            directoryExist(MODELS_LOG + s)
            dump(ab_clf, MODELS_LOG + s + '/Adaboost_LOG.joblib')
            print("done",s)

        
    def log_training_with_LogitBoost(self, all_data = True):

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            lb_clf = LogitBoost(n_estimators= 200, bootstrap = True)
            lb_clf.fit(X_train, Y_train)
            directoryExist(MODELS_LOG + s)
            dump(lb_clf, MODELS_LOG + s + '/LBoost_LOG.joblib')
            print("done",s)
            

        

    def log_training_with_Bagging(self, all_data = True):

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            bagging_clf = BaggingClassifier(LogisticRegression(tol = .0001),n_estimators=15)
            bagging_clf.fit(X_train, Y_train)
            directoryExist(MODELS_LOG + s)
            dump(bagging_clf, MODELS_LOG + s + '/Bagging_LOG.joblib')
            print("done",s)


if __name__ == "__main__":

    LOG = LOG_Model()
    LOG.startTraining(True,False,False,False)
    LOG.startTesting()

   