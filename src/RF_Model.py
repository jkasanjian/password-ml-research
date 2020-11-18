#temporarily need 
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from IPython.display import Image
from sklearn.tree import export_graphviz
from subprocess import check_call
from Helper import get_results, get_train_data, read_data, directoryExist, get_results


DATA_SPLIT = 'data/split/'
MODELS_RF = 'src/models/all_data/'
MODELS_RF_UB = 'src/models/unbalanced_data/'



class RF_Model:

    
    def __init__(self, balanced = False):
        _, self.subjects = read_data()

    
    def startTraining(self,reg = False, ada = False, Bagging = False):
        if(reg):
            self.rf_training()
        if(ada):
            self.rf_training_with_adaBoost()
        if(Bagging):
            self.rf_training_with_Bagging()
    
    
    def startTesting(self):
        model_names = ["RF","Adaboost_RF","Bagging_RF"]
        for i in model_names:
            get_results(RF.subjects,i,"RF")
    
    
    #Generates a Random Forest regression model for each subject
    def rf_training(self, all_data = True): 
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
            dump(clf, MODELS_RF + s + '/RF.joblib')
                


        
    def rf_training_with_adaBoost(self, all_data = True):

        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            rf_clf = RandomForestClassifier(n_estimators = 60, max_depth = 10, criterion =  "gini")
            ada_clf = AdaBoostClassifier(rf_clf, n_estimators = 20, learning_rate = 1)
            ada_clf.fit(X_train,Y_train)
            directoryExist(MODELS_RF + s)
            dump(ada_clf, MODELS_RF + s + '/Adaboost_RF.joblib')
            print(s,"Finished")
      

    def rf_training_with_Bagging(self, all_data = True):
        
        _, subjects = read_data()
        for s in self.subjects:
            X_train, Y_train = get_train_data(s,all_data)
            bagging_clf = BaggingClassifier(RandomForestClassifier(n_estimators = 60, max_depth = 10, criterion =  "gini"),n_estimators=10)
            bagging_clf.fit(X_train, Y_train)
            directoryExist(MODELS_RF + s)
            print("Finished:",s[2:])
            dump(bagging_clf, MODELS_RF + s + '/Bagging_RF.joblib')


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
    # RF.startTraining(True,False,False)
    RF.startTesting()


    
    