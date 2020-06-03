from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix

def SVM():
    print("---------------Running Training---------------")
    accuracies = []
    models = []
    #Training on different Cross Fold Validation sets
    for i in range(1,6):
        accuracies[i],models[i] = trainSVM(i)

    print("-------------All Estimator Scores-------------")
    for i in accuracies:
        print(i)
    
    print("----------------Best Estimator-----------------")
    print("Model:")
    print(models[accuracies.index(max(accuracies))])
    print("Score:",max(accuracies))

def trainSVM(cv):
    #Should go through 120 different iterations
    param_grid = {"kernel":["linear","rbf"], "C" : [1,5,25,125,725,100], "degrees": [1,2,3,4,5,6,7,8,9,10]}
    
    #Split and pass in data to train SVM
    grid_classifier = GridSearchCV(SVC(gamma = "scale", decision_function_shape="ovo" ), param_grid, cv)
    grid_classifier.fit(X_train,y_train)

    # Needed if we want to take note of statistical data of each model 
    # print("\nGrid scores on development set:")
    # print()
    # means = grid_classifier.cv_results_['mean_test_score']
    # stds = grid_classifier.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, grid_classifier.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #         % (mean, std * 2, params))

    print("\nBest parameters set found on development set:\n")

    print(grid_classifier.best_params_)
    y_pred = grid_classifier.predict(X_test)
    print(classification_report(y_test, y_pred,target_names= ["user", "intruder"]))
    
    return grid_classifier.best_estimator_.score, grid_classifier.best_estimator_