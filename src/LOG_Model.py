#temporarily need 
from sklearn import metrics
###################
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from logitboost import LogitBoost
from sklearn.ensemble import BaggingClassifier
from Helper import directoryExist, get_train_data, read_data, get_results, get_test_data

DATA_SPLIT = 'data/split/'
MODELS_LOG = 'src/models/all_data/'
MODELS_LOG_UB = 'src/models/unbalanced_data/'



class LOG_Model:

    def __init__(self, balanced = False):
        _, self.subjects = read_data()

    def startTraining(self,reg = True, ada = False, Bagging = False):
        if(reg):
            self.log_training()
        if(ada):
            self.log_training_with_adaBoost()
        if(Bagging):
            self.log_training_with_Bagging()
    
    def startTesting(self):
        #TODO begin testing here
        model_names = ["LBoost_LOG"]
        for i in model_names:
            get_results(LOG.subjects,i,"LOG")
        print()


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



        
    def log_training_with_adaBoost(self, all_data = True):

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            lb_clf = LogitBoost(n_estimators= 200, bootstrap = True)
            lb_clf.fit(X_train, Y_train)
            directoryExist(MODELS_LOG + s)
            dump(lb_clf, MODELS_LOG + s + '/LBoost_LOG.joblib')

        

    def log_training_with_Bagging(self, all_data = True):

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            X_train = X_train.astype(np.float)
            Y_train = Y_train.astype(np.float)
            bagging_clf = BaggingClassifier(LogisticRegression(tol = .0001),n_estimators=15)
            bagging_clf.fit(X_train, Y_train)
            directoryExist(MODELS_LOG + s)
            dump(bagging_clf, MODELS_LOG + s + '/Bagging_LOG.joblib')


if __name__ == "__main__":

    LOG = LOG_Model()
    LOG.startTraining(False,True,False)
    LOG.startTesting()

    # X_test, Y_test = get_test_data("s037",True)
    # model = load_model("LOG","s037")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('LOG done training')
    # X_test, Y_test = get_test_data("s055",True)
    # model = load_model("LOG","s055")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('LOG done training')
    # X_test, Y_test = get_test_data("s050",True)
    # model = load_model("LOG","s030")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('LOG done training')