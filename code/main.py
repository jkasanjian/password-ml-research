import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import operator
from os import mkdir, remove
from joblib import dump, load
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from IPython.display import Image
from sklearn.tree import export_graphviz
from subprocess import check_call
import os


# DIRECTORIES
DATA_SOURCE = 'data/DSL-StrongPasswordData.csv'
DATA_JSON = 'data/password_data.json'
DATA_SPLIT_SK = 'data/split/sklearn/'

MODELS_LOGIT = 'models/logistic/'
MODELS_KNN = 'models/knn/'
MODELS_RF = 'models/randfor/'
MODELS_SVM = 'models/svm/'

USER_GRAPHS = 'data/graphs/'
TREE_GRAPHS = 'data/trees/'
RESULTS_ALONE = 'results/alone/'
RESULTS_COMBO = 'results/combo/'


# ---------------------------- DATA PROCESSING ----------------------------

def read_data():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)
    # features = data['subject']
    del data['subject']
    subjects = list(data.keys())
    
    return data, subjects

def get_features():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)

    return data['subject'][0]


def make_json():
    data = {}
    with open(DATA_SOURCE) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] not in data:
                trials = []
                trials.append(row[3:])
                data[row[0]] = trials 
            else:
                data[row[0]].append(row[3:])

    with open(DATA_JSON, 'w') as fp:
        json.dump(data, fp)



def create_signature_graphs():
    ''' Creates visualization of each subject ''' 
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)

    features = data['subject'][0]
    del data['subject']

    x = [i for i in range(len(features))]
    for subject in data:
        plt.figure(figsize=(16,8))
        for trial in data[subject]:
            plt.scatter(x, trial)
        plt.xticks(x, features, rotation='vertical', fontsize=10)
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.2)
        plt.xlim(-1,31)
        plt.grid(True)
        plt.xlabel('Features')
        plt.ylabel('Time')
        plt.title(subject)
        f_name = USER_GRAPHS + subject
        plt.savefig(f_name)
        plt.close()



def load_all_data(subject, data):
    '''
    Returns all data as x, y given a subject
    x,y - all 400 repetitions of subject, first 10 of each imposter
    '''
    user = data[subject]
    imposter = []
    for s in data:
        if s != subject:
            imposter.extend(data[s][:10])

    x = user + imposter
    y = [1 for i in range(len(user))] + [-1 for i in range(len(imposter))]

    return (np.array(x).astype(np.float64), np.array(y).astype(np.float64))



def split_data_sklearn():
    data, subjects = read_data()
    for s in subjects:
        mkdir(DATA_SPLIT_SK + s)

        x, y = load_all_data(s, data)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, train_size=0.5)
        np.save(DATA_SPLIT_SK + s + '/x_train.npy', x_train)
        np.save(DATA_SPLIT_SK + s + '/x_test.npy', x_test)
        np.save(DATA_SPLIT_SK + s + '/y_train.npy', y_train)
        np.save(DATA_SPLIT_SK + s + '/y_test.npy', y_test)



def load_data_sklearn(s):
    ''' Loads datasets for a subject s'''
    x_train = np.load(DATA_SPLIT_SK + s + '/x_train.npy')
    x_test = np.load(DATA_SPLIT_SK + s + '/x_test.npy')
    y_train = np.load(DATA_SPLIT_SK + s + '/y_train.npy')
    y_test = np.load(DATA_SPLIT_SK + s + '/y_test.npy')

    return x_train, x_test, y_train, y_test
    


def load_models(s):
    ''' Returns all models for a given subject s
        ADD NEW MODELS HERE TO RETURN STATEMENT '''
    knn_clf = load(MODELS_KNN + s + '.joblib') 
    log_clf = load(MODELS_LOGIT + s + '.joblib')
    rfr_clf = load(MODELS_RF + s +'.joblib')

    return knn_clf, log_clf, rfr_clf


def load_model(name, s):
    ''' Returns the corresponding model for a subject '''
    return load('models/' + name + '/' + s + '.joblib')


def get_params_rf(s):
    ''' Returns the calculate best hyperparameters for a subject's
        Random Forest model '''
    with open(MODELS_RF+'best_params.json') as json_file:
        params = json.load(json_file)
    return params[s]


def get_params_knn(s):
    ''' Returns the calculate best hyperparameters for a subject's
        Random Forest model '''
    with open(MODELS_KNN+'best_params.json') as json_file:
        params = json.load(json_file)
    return params[s]


def get_params_logreg(s):
    ''' Returns the calculate best hyperparameters for a subject's
        Random Forest model '''
    with open(MODELS_LOGIT+'best_params.json') as json_file:
        params = json.load(json_file)
    return params[s]


def get_params_svm(s):
    ''' Returns the calculate best hyperparameters for a subject's
        Random Forest model '''
    with open(MODELS_SVM+'best_params.json') as json_file:
        params = json.load(json_file)
    return params[s]


# ---------------------------- KNN ----------------------------

def knn_tuning():
    ''' Calculates and saves the best hyperparameteres for each subject's KNN model '''
    _, subjects = read_data()
    best_params = {}


    leaf_size = list(range(1,50))
    n_neighbors = list(range(1,30))
    p=[1,2]
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

    for s in subjects:
        x_train, _, y_train, _ = load_data_sklearn(s)

        knn_tune = KNeighborsClassifier()
        clf = GridSearchCV(knn_tune, hyperparameters, cv=10)
        clf.fit(x_train, y_train)
        
        best_params[s] = clf.best_params_
    
    with open(MODELS_KNN+'best_params.json', 'w') as fp:
        json.dump(best_params, fp)


def knn_classifier():
    ''' Generates a KNN model for each subject, stored in models/knn/ folder'''
    _, subjects = read_data()

    for s in subjects:
        x_train, _, y_train, _ = load_data_sklearn(s)
        params = get_params_knn(s)
        clf = KNeighborsClassifier(algorithm='brute', metric='minkowski')
        clf.set_params(**params)
        clf.fit(x_train, y_train)

        dump(clf, MODELS_KNN + s + '.joblib')



# ---------------------------- Logistic Regression ----------------------------

def log_reg_tuning():
    '''' Calculates and saves the best hyperparameteres for each subject's
    Logistic Regression model '''
    _, subjects = read_data()
    best_params = {}
    
    param_grid = [    
        {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
        'C' : np.logspace(-4, 4, 20),
        'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
        'max_iter' : [100, 1000, 2500, 5000]
        }
    ]

    for s in subjects:
        x_train, _, y_train, _ = load_data_sklearn(s)
        
        log_model = LogisticRegression()
        clf = GridSearchCV(log_model, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
        clf.fit(x_train, y_train)
        best_params[s] = clf.best_params_

    with open(MODELS_LOGIT+'best_params.json', 'w') as fp:
        json.dump(best_params, fp)



def log_reg_classifier():
    ''' Generates a Logistic regression model for each subject, 
        stored in models/logistic/ folder '''
    _, subjects = read_data()

    for s in subjects:
        x_train, _, y_train, _ = load_data_sklearn(s) 
        clf = LogisticRegression()
        params = get_params_logreg(s)
        clf.set_params(**params)
        clf.fit(x_train, y_train)

        dump(clf, MODELS_LOGIT + s + '.joblib')



# ---------------------------- Random Forest ----------------------------

def random_forest_classifier():
    ''' Generates a Random Forest classifier model for each 
        subject stored in models/rfc/ folder '''
    _, subjects = read_data()
    
    for s in subjects:
        x_train, _, y_train, _ = load_data_sklearn(s)
        params = get_params_rf(s)
        clf = RandomForestRegressor(**params)
        clf.fit(x_train, y_train)
        dump(clf, MODELS_RF + s + '.joblib')
        #visualize_tree(clf, s)



def random_forest_classifier_tuning():
    ''' Generates a Random Forest regression model for each subject'''
    _, subjects = read_data()
    best = {}

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = list(range(50, 100, 10))
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    param_dist = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    for s in subjects:
        x_train, x_test, y_train, y_test = load_data_sklearn(s) 
        # Create the random grid
        print(s)
        break
        rf = RandomForestClassifier()
        rf = GridSearchCV(estimator=rf, param_grid=param_dist, cv=3, scoring='precision_macro', n_jobs= -1)
        rf.fit(x_train, y_train)

        best[s] = rf.best_params_
        print('DONE', rf.best_params_)
        # break
        y_pred = rf.predict(x_test)
        print(metrics.classification_report(y_test, y_pred))
        break

    # with open(MODELS_RF+'best_params.json', 'w') as fp:
    #     json.dump(best, fp)



def visualize_tree(clf, s):
    ''' Creates a graphics diagram of a tree from the random forest '''
    estimator = clf.estimators_[5]

    export_graphviz(estimator, out_file=TREE_GRAPHS+s+'tree.dot', 
                feature_names = get_features(),
                class_names = ['genuine user', 'imposter'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

    check_call(['dot', '-Tpng', TREE_GRAPHS+s+'tree.dot', '-o', TREE_GRAPHS+s+'tree.png', '-Gdpi=600'])
    Image(filename=TREE_GRAPHS+s+'tree.png')
    remove(TREE_GRAPHS+s+'tree.dot')




# ---------------------------- SVM ----------------------------

def support_vector_tuning():
    ''' Calculates and saves the best hyperparameteres for each subject's SVM model '''
    _, subjects = read_data()
    best_params = {}
    param_grid = {"kernel":["linear","rbf","poly"], "C" : [i for i in range(1,1000,10)] , "degree": [1,2]}
    
    for s in subjects:
        x_train, _, y_train, _= load_data_sklearn(s)
        
        grid_classifier = GridSearchCV(SVC(gamma ="auto",decision_function_shape="ovo"), param_grid, scoring = 'recall_macro', cv = 5)
        grid_classifier.fit(x_train,y_train)

        best_params[s] = (grid_classifier.best_params_)

    with open(MODELS_SVM+'best_params.json', 'w') as fp:
        json.dump(best_params, fp)



def support_vector_classifier():
    ''' Generates a SVM classifier model for each sbuject
        stored in models/svm/ folder '''
    _, subjects = read_data()
    
    for s in subjects:
        x_train, _, y_train, _ = load_data_sklearn(s)

        params = get_params_svm(s)
        clf = SVC(gamma ="auto",decision_function_shape="ovo")
        clf.set_params(**params)
        clf.fit(x_train, y_train)

        dump(clf, MODELS_SVM + s + '.joblib')



# ---------------------------- EVALUATION ----------------------------

def evaluate_all_models_alone():
    ''' Evaluates all models alone and generates results files '''
    models = [('knn', False), ('logistic', False), ('randfor', True), ('svm', False)]
    for m in models:
        evaluate_model(m[0], m[1])



def evaluate_model(name, check_pred):
    ''' _Evaluates the model named by calculating the false rejection and false acceptance rates 
        Params: name - name of model, check_pred - True if model returns probability not label '''
    _, subjects = read_data()
    results = {}
    results['subjects'] = {}
    all_miss_rate = []
    all_f_alarm_rate = []
    all_accs = []
    all_pers = []
    all_recs = []

    for s in subjects:
        _, x_test, _, y_test = load_data_sklearn(s)

        s_results = {}
        n = len(y_test)
        miss = 0
        f_a = 0
        total_pos = 0

        clf = load_model(name, s)
        y_pred = clf.predict(x_test)

        for i in range(n):
            if check_pred:
                y_pred = np.where(y_pred >= 0.5, 1.0, -1.0)

            if y_test[i] == 1.0:
                total_pos += 1
                if y_pred[i] == -1.0:
                    f_a += 1

            elif y_test[i] == -1.0 and y_pred[i] == 1.0:
                miss += 1

        miss_rate = miss / n
        false_alarm_rate = f_a / n
        precision = (total_pos - miss) / ((total_pos - miss) + miss)
        recall = (total_pos - miss) / ((total_pos - miss) + f_a)
        accuracy = (n - miss - f_a) / n 

        s_results['false acceptance rate'] = miss_rate
        all_miss_rate.append(miss_rate)
        s_results['false rejection rate'] = false_alarm_rate
        all_f_alarm_rate.append(false_alarm_rate)
        s_results['precision'] = precision
        all_pers.append(precision)
        s_results['recall'] = recall
        all_recs.append(recall)
        s_results['accuracy'] = accuracy
        all_accs.append(accuracy)
        
        results['subjects'][s] = s_results
        
    results['false acceptance rate mean'] = np.array(all_miss_rate).mean()
    results['false acceptance rate SD'] = np.array(all_miss_rate).std()
    results['false rejection rate mean'] = np.array(all_f_alarm_rate).mean()
    results['false rejection rate SD'] = np.array(all_f_alarm_rate).std()
    results['precision mean'] = np.array(all_pers).mean()
    results['precision SD'] = np.array(all_pers).std()
    results['recall mean'] = np.array(all_recs).mean()
    results['recall SD'] = np.array(all_recs).std()
    results['accuracy mean'] = np.array(all_accs).mean()
    results['accuracy SD'] = np.array(all_accs).std()
    
    with open(RESULTS_ALONE + name + '.json', 'w') as fp:
        json.dump(results, fp)


def evaluate_majority_vote(models):
    ''' Evaluates the performance of using multiple classifiers and taking a majority vote'''
    _, subjects = read_data()
    num_models = len(models)
    win_count = [0, 0, 0]
    results = {}
    results['subjects'] = {}
    all_miss_rate = []
    all_f_alarm_rate = []
    all_accs = []
    all_pers = []
    all_recs = []
    fname = RESULTS_COMBO

    first = True
    for s in subjects:
        clfs = []
        for m in models:
            clfs.append(load_model(m[0], s))
            if first:
                fname += m[0] + '-'
        first = False
        
        _, x_test, _, y_test = load_data_sklearn(s)
        
        preds = []
        for i in range(num_models):
            pred = clfs[i].predict(x_test)
            if models[i][1]: # if check_pred is true
                pred = np.where(pred >= 0.5, 1.0, -1.0)
            preds.append(pred)
        
        s_results = {}
        n = len(y_test)
        miss = 0
        f_a = 0
        total_pos = 0
        
        for i in range(n):
            votes = []
            for p in preds:
                votes.append(p[i])

            if sum(votes) >= 1.0:
                classify = 1.0
            else:
                classify = -1.0
            
            if y_test[i] == 1.0:
                total_pos += 1
                if classify == -1.0:
                    f_a += 1

            elif y_test[i] == -1.0 and classify == 1.0:
                miss += 1

            # counting winners
            for j in range(len(votes)):
                if y_test[i] == votes[j]:
                    win_count[j] += 1


        miss_rate = miss / n
        false_alarm_rate = f_a / n
        precision = (total_pos - miss) / ((total_pos - miss) + miss)
        recall = (total_pos - miss) / ((total_pos - miss) + f_a)
        accuracy = (n - miss - f_a) / n 

        s_results['false acceptance rate'] = miss_rate
        all_miss_rate.append(miss_rate)
        s_results['false rejection rate'] = false_alarm_rate
        all_f_alarm_rate.append(false_alarm_rate)
        s_results['precision'] = precision
        all_pers.append(precision)
        s_results['recall'] = recall
        all_recs.append(recall)
        s_results['accuracy'] = accuracy
        all_accs.append(accuracy)
        
        results['subjects'][s] = s_results
        
    results['false acceptance rate mean'] = np.array(all_miss_rate).mean()
    results['false acceptance rate SD'] = np.array(all_miss_rate).std()
    results['false rejection rate mean'] = np.array(all_f_alarm_rate).mean()
    results['false rejection rate SD'] = np.array(all_f_alarm_rate).std()
    results['precision mean'] = np.array(all_pers).mean()
    results['precision SD'] = np.array(all_pers).std()
    results['recall mean'] = np.array(all_recs).mean()
    results['recall SD'] = np.array(all_recs).std()
    results['accuracy mean'] = np.array(all_accs).mean()
    results['accuracy SD'] = np.array(all_accs).std()
    results['win count'] = win_count
    
    fname = fname[:-1] + '.json'
    with open(fname, 'w') as fp:
        json.dump(results, fp)
            
            
    
def run_majority_votes():
    # models = 
    models = [  [('knn', False), ('logistic', False), ('randfor', True)],
                [('knn', False), ('logistic', False), ('svm', False)],
                [('knn', False), ('randfor', True), ('svm', False)],
                [('logistic', False), ('randfor', True), ('svm', True)] ]
    
    for m_set in models:
        evaluate_majority_vote(m_set)


def print_all_stats():
    # for f in listdir('results/alone/'):
    #     file = 'results/alone/' + f

    dash = '-' * 40
    for path, _, files in os.walk('results'):
        for name in files:
            file = os.path.join(path, name)
            if file[-5:] == '.json':
                with open(file) as json_file:
                    data = json.load(json_file)
                    print('\n\n\nResults for', name[:-5])
                    print(dash)

                    print('{:<30s}{:<2.8f}'.format('Precision mean:', 100*data['precision mean']))
                    print('{:<30s}{:<2.8f}'.format('Precision SD:', 100*data['precision SD']))
                    print()
                    print('{:<30s}{:<2.8f}'.format('Recall mean:', 100*data['recall mean']))
                    print('{:<30s}{:<2.8f}'.format('Recall SD:', 100*data['recall SD']))
                    print()
                    print('{:<30s}{:<2.8f}'.format('FAR mean:', 100*data['false acceptance rate mean']))
                    print('{:<30s}{:<2.8f}'.format('FAR SD:', 100*data['false acceptance rate SD']))
                    print()
                    print('{:<30s}{:<2.8f}'.format('FRR mean:', 100*data['false rejection rate mean']))
                    print('{:<30s}{:<2.8f}'.format('FRR SD:', 100*data['false rejection rate SD']))
                    
                    
                    
                    
                    
                   



if __name__ == '__main__':
    # evaluate_all_models_alone()
    # run_majority_votes()
    # print_all_stats()
    random_forest_classifier_tuning()