#temporarily need 
from sklearn import metrics
###################
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from Helper import get_test_data, get_train_data, read_data, load_model

DATA_SPLIT = 'data/split/'
MODELS_LOG = 'src/models/all_data/'
MODELS_LOG_UB = 'src/models/unbalanced_data/'



class LOG_Model:

    def __init__(self, balanced = False):
        _, self.subjects = read_data()

    def startTraining(self,reg = True, ada = False, Bagging = False):
        if(reg):
            log_training()
        if(ada):
            log_training_with_adaBoost()
        if(Bagging):
            log_training_with_Bagging()
    
    def startTesting(self):
        #TODO begin testing here
        print()


    def log_training(self, all_data = True): 
    #Calculates and saves the best hyperparameters for each subject's
    #Logistic Regression model
        #penalty = ['l1', 'l2', 'elasticnet', 'none']
        penalty = ['none']
        C = np.logspace(-4, 4, 20)
        solver = ['lbfgs','newton-cg','liblinear','sag','saga']
        max_iter = [100, 1000, 2500, 5000]
        hyperparameters = dict(penalty=penalty, C=C, solver=solver, max_iter=max_iter)

        # for s in self.subjects:
        #     X_train, Y_train = get_train_data(s,all_data)
        #     log_tune = LogisticRegression()
        #     clf = GridSearchCV(log_tune, hyperparameters, scoring='f1', n_jobs=-1)
        #     clf.fit(X_train, Y_train)
        #     if not os.path.isdir(MODELS_LOG + s):
        #         os.makedirs(MODELS_LOG + s)
        #     dump(clf, MODELS_LOG + s + '/LOG.joblib')

      
        X_train, Y_train = get_train_data(self.subjects[0],all_data)
        log_tune = LogisticRegression()
        clf = GridSearchCV(log_tune, hyperparameters, scoring='f1', n_jobs=-1)
        print("GGGGGGGGGGGGGG GGGGGGGGGGGGGG GGGGGGGGGGGGGG GGGGGGGGGGGGGG GGGGGGGGGGGGGG ")
        clf.fit(X_train, Y_train)
        print("GGGGGGGGGGGGGG GGGGGGGGGGGGGG GGGGGGGGGGGGGG GGGGGGGGGGGGGG GGGGGGGGGGGGGG ")
        if not os.path.isdir(MODELS_LOG + self.subjects[0]):
            os.makedirs(MODELS_LOG + self.subjects[0])
        dump(clf, MODELS_LOG + self.subjects[0] + '/LOG.joblib')


        
    def log_training_with_adaBoost(self, all_data = True):

        penalty = ['l1', 'l2', 'elasticnet', 'none']
        C = np.logspace(-4, 4, 20)
        solver = ['lbfgs','newton-cg','liblinear','sag','saga']
        max_iter = [100, 1000, 2500, 5000]
        hyperparameters = dict(penalty=penalty, C=C, solver=solver, max_iter=max_iter)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            b_clf = AdaBoostClassifier(LogisticRegression())
            clf = GridSearchCV(b_clf, hyperparameters, scoring='f1', n_jobs=-1)
            clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_LOG + s):
                os.makedirs(MODELS_LOG + s)
            dump(clf, MODELS_LOG + s + '/Adaboost_LOG.joblib')

        

    def log_training_with_Bagging(self, all_data = True):


        penalty = ['l1', 'l2', 'elasticnet', 'none']
        C = np.logspace(-4, 4, 20)
        solver = ['lbfgs','newton-cg','liblinear','sag','saga']
        max_iter = [100, 1000, 2500, 5000]
        hyperparameters = dict(penalty=penalty, C=C, solver=solver, max_iter=max_iter)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            bagging_clf = BaggingClassifier(LogisticRegression())
            clf = GridSearchCV(bagging_clf, hyperparameters, scoring='f1', n_jobs=-1)
            clf.fit(X_train, Y_train)
            if not os.path.isdir(MODELS_LOG + s):
                os.makedirs(MODELS_LOG + s)
            dump(clf, MODELS_LOG + s + '/Bagging_LOG.joblib')


if __name__ == "__main__":

    LOG = LOG_Model()
    LOG.log_training()

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