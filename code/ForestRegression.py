from main import *
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import graphviz 
import os
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import pydot
# os.system('dot -Tpng random.dot -o random.png')


def makeSamples():
    for i in range(1):
        X,y,z = getData()
        zz = np.ones((len(z),1))
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=.20, random_state = 242)
        Classifier = DecisionTreeClassifier(min_samples_split=10)
        r = RandomForestRegressor(n_jobs=-1, min_samples_split=2, max_depth= 32) 
        estimators = np.arange(1000, 1001, 1000)
        scores = []
        for n in estimators:
            r.set_params(n_estimators=n)
            r.fit(X_train, y_train)
            y_pred = r.predict(X_test)
            num = 0 
            for i in y_pred:
                if i > .51:
                    y_pred[num] = 1.0
                else: 
                    y_pred[num] = 0.0
                num +=1
            # print(y_pred)
            print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
            print(r.get_params)
            y_pred = r.predict(z)
            num = 0 
            for i in y_pred:
                if i > .50:
                    y_pred[num] = 1.0
                else: 
                    y_pred[num] = 0.0
                num +=1
            print(y_pred)
            print("Accuracy:", metrics.accuracy_score(zz,y_pred))
            if(metrics.accuracy_score(zz,y_pred) >= .98):
                print("Total data:",len(y_train))
                print("Positive data:",y_train.count(1.0)," Negative data:", len(y_train) - y_train.count(1.0))
                # print(s)
                print("\n")
            else:
                print("DIDN MAKE IT")
                print("Total data:",len(y_train))
                print("Positive data:",y_train.count(1.0)," Negative data:", len(y_train) - y_train.count(1.0))
                # print(s)
                print("\n")
    # plt.title("Effect of n_estimators")
    # plt.xlabel("n_estimator")
    # plt.ylabel("score")
    # plt.plot(estimators, scores)   
    # plt.show()
    Classifier.fit(X_train, y_train)
    y_pred = Classifier.predict(X_test,y_test)
    print("Prediction:",y_pred)
    print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
    dot_data = export_graphviz(Classifier, out_file='tree.dot',
                rounded = True, proportion = False, 
                precision = 2, filled = True)
    graph = graphviz.Source(dot_data) 
    # (graph,) = pydot.graph_from_dot_file('tree.dot')
    # graph.write_png('somefile.png')
    graph.render() 
    


def main():
    makeSamples()



main()