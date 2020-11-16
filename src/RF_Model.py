#temporarily need 
from sklearn import metrics
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from IPython.display import Image
from sklearn.tree import export_graphviz
from subprocess import check_call
from Helper import get_test_data, get_train_data, read_data, load_model, directoryExist


DATA_SPLIT = 'data/split/'
MODELS_RF = 'src/models/all_data/'
MODELS_RF_UB = 'src/models/unbalanced_data/'



class RF_Model:

    def __init__(self, balanced = False):
        _, self.subjects = read_data()

    def startTraining(self,reg = True, ada = False, Bagging = False):
        if(reg):
            self.rf_training()
        if(ada):
            self.rf_training_with_adaBoost()
        if(Bagging):
            rf_training_with_Bagging()
    
    def startTesting(self):
        #TODO begin testing here
        print()


    def rf_training(self, all_data = True): 
    #Generates a Random Forest regression model for each subject
        _, subjects = read_data()
   
        n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = list(range(50, 100, 10))
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
                    bootstrap=bootstrap)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            rf_clf = RandomForestClassifier()
            clf = GridSearchCV(rf_clf, hyperparameters, scoring='f1', n_jobs= -1)
            clf.fit(X_train, Y_train)
            directoryExist(MODELS_RF + s)
            if not os.path.isdir(MODELS_RF + s):
                os.makedirs(MODELS_RF + s)
            dump(clf, MODELS_RF + s + '/RF.joblib')
            print("Finished:",s[2:])


        
    def rf_training_with_adaBoost(self, all_data = True):

        _, subjects = read_data()
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = list(range(50, 100, 10))
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
                    bootstrap=bootstrap)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            rf_clf = RandomForestClassifier(n_estimators = 60, max_depth = 10, criterion =  "gini")
            #rf_clf = RandomForestRegressor(n_estimators=100)
            ada_clf = AdaBoostClassifier(rf_clf, n_estimators = 20, learning_rate = 1)
            ada_clf.fit(X_train,Y_train)
            directoryExist(MODELS_RF + s)
            if not os.path.isdir(MODELS_RF + s):
                os.makedirs(MODELS_RF + s)
            dump(ada_clf, MODELS_RF + s + '/Adaboost_RF.joblib')
            print(s,"Finished")
      

    def rf_training_with_Bagging(self, all_data = True):

        _, subjects = read_data()
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = list(range(50, 100, 10))
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
                    bootstrap=bootstrap)

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            bagging_clf = BaggingClassifier(RandomForestClassifier())
            clf = GridSearchCV(bagging_clf, hyperparameters, scoring='f1', n_jobs= -1)
            clf.fit(X_train, Y_train)
            directoryExist(MODELS_RF + s)
            dump(clf, MODELS_RF + s + '/Bagging_RF.joblib')


    #TODO finish implementing visual for each users tree
    
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
    # RF.startTraining(True,True)
    results = {}
    all_miss_rate = []
    all_f_alarm_rate = []
    all_pers = []
    all_recs = []
    all_f1s = []
    results['subjects'] = {}
    for s in RF.subjects:
        X_test, Y_test = get_test_data(s,True)
        model = load_model("RF",s)
        Y_pred = model.predict(X_test)

        

        s_results = {}
        n = len(Y_test)
        miss = 0
        f_a = 0
        total_pos = 0

        for i in range(n):

            if Y_test[i] == 1.0:
                total_pos += 1
                if Y_pred[i] == -1.0:
                    f_a += 1

            elif Y_test[i] == -1.0 and Y_pred[i] == 1.0:
                miss += 1

        miss_rate = miss / n
        false_alarm_rate = f_a / n
        precision = (total_pos - miss) / ((total_pos - miss) + miss)
        recall = (total_pos - miss) / ((total_pos - miss) + f_a)
        f1 = 2 * ((precision * recall) / (precision + recall))
        print("\n--------------------",s,"--------------------")
        s_results['false acceptance rate'] = miss_rate
        print('false acceptance rate',miss)
        all_miss_rate.append(miss_rate)
        s_results['false rejection rate'] = false_alarm_rate
        print('false rejection rate', false_alarm_rate)
        all_f_alarm_rate.append(false_alarm_rate)
        s_results['precision'] = precision
        print('precision',precision)
        all_pers.append(precision)
        s_results['recall'] = recall
        print('recall',recall)
        all_recs.append(recall)
        s_results['f1'] = f1
        print('f1',f1)
        all_f1s.append(f1)
        print("Accuracy:",metrics.accuracy_score(Y_pred,Y_test))
        
        
        results['subjects'][s] = s_results
        
    results['false acceptance rate mean'] = np.array(all_miss_rate).mean()
    results['false acceptance rate SD'] = np.array(all_miss_rate).std()
    results['false rejection rate mean'] = np.array(all_f_alarm_rate).mean()
    results['false rejection rate SD'] = np.array(all_f_alarm_rate).std()
    results['precision mean'] = np.array(all_pers).mean()
    results['precision SD'] = np.array(all_pers).std()
    results['recall mean'] = np.array(all_recs).mean()
    results['recall SD'] = np.array(all_recs).std()
    results['f1 mean'] = np.array(all_f1s).mean()
    results['f1 SD'] = np.array(all_f1s).std()
    print(results['false acceptance rate mean'])
    print(results['false acceptance rate SD'])
    print(results['false rejection rate mean'])
    print(results['false rejection rate SD'])
    print(results['precision mean'])
    print(results['precision SD'])
    print(results['f1 mean'])

    # X_test, Y_test = get_test_data("s037",True)
    # model = load_model("RF","s037")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('RF done training')
    # X_test, Y_test = get_test_data("s055",True)
    # model = load_model("RF","s055")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('RF done training')
    # X_test, Y_test = get_test_data("s050",True)
    # model = load_model("RF","s030")
    # Y_pred = model.predict(X_test)
    # print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
    # print('RF done training')
    