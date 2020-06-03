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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from IPython.display import Image
from sklearn.tree import export_graphviz
from subprocess import check_call


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



def test_models_alone():
    ''' Calculates the miss/false alarm rates for each model individually'''

    _, subjects = read_data()
   
    knn_results = {}
    knn_results['miss rate'] = []
    knn_results['false alarm rate'] = []

    log_results = {}
    log_results['miss rate'] = []
    log_results['false alarm rate'] = []

    rfr_results = {}
    rfr_results['miss rate'] = []
    rfr_results['false alarm rate'] = []

    for s in subjects:

        knn_clf, log_clf, rfr_clf = load_models(s)   # ADD NEW MODELS HERE

        _, x_test, _, y_test = load_data_sklearn(s)
        n = len(y_test)

        knn_pred = knn_clf.predict(x_test)
        miss_knn = 0
        f_alarm_knn = 0

        log_pred = log_clf.predict(x_test)
        miss_log = 0
        f_alarm_log = 0

        rfr_pred = rfr_clf.predict(x_test)
        miss_rfr = 0
        f_alarm_rfr = 0

        for i in range(n):

            if knn_pred[i] == 1.0 and y_test[i] == -1.0:
                miss_knn += 1
            elif knn_pred[i] == -1.0 and y_test[i] == 1.0:
                f_alarm_knn += 1

            if log_pred[i] == 1.0 and y_test[i] == -1.0:
                miss_log += 1
            elif log_pred[i] == -1.0 and y_test[i] == 1.0:
                f_alarm_log += 1
            
            if rfr_pred[i] == 1.0 and y_test[i] == -1.0:
                miss_rfr += 1
                
            elif rfr_pred[i] == -1.0 and y_test[i] == 1.0:
                f_alarm_rfr += 1
                
        
        
            
        knn_results['miss rate'].append(miss_knn/n)
        knn_results['false alarm rate'].append(f_alarm_knn/n)

        log_results['miss rate'].append(miss_log/n)
        log_results['false alarm rate'].append(f_alarm_log/n)

        rfr_results['miss rate'].append(miss_rfr/n)
        rfr_results['false alarm rate'].append(f_alarm_rfr/n)

    print('Pred:',log_pred[:30])
    print('Test:',y_test[:30])

    print('KNN Average miss rate:', sum(knn_results['miss rate'])/len(subjects))
    print('KNN Average false alarm rate:', sum(knn_results['false alarm rate'])/len(subjects))

    print('\nLogistic Average miss rate:', sum(log_results['miss rate'])/len(subjects))
    print('Logistic Average false alarm rate:', sum(log_results['false alarm rate'])/len(subjects))

    print('\nRandom Forest Average miss rate:', sum(rfr_results['miss rate'])/len(subjects))
    print('Random Forest Average false alarm rate:', sum(rfr_results['false alarm rate'])/len(subjects))
        





def knn_classifier():
    ''' Generates a KNN model for each subject, stored in models/knn/ folder'''
    _, subjects = read_data()
    k = 2   # yielded highest accuracy

    for s in subjects:
        x_train, _, y_train, _ = load_data_sklearn(s)
        clf = KNeighborsClassifier(algorithm='brute',
                                   metric='minkowski',
                                   n_neighbors=k)
        clf.fit(x_train, y_train)
    
        dump(clf, MODELS_KNN + s + '.joblib') 



def log_reg_classifier():
    ''' Generates a Logistic regression model for each subject, 
        stored in models/logistic/ folder '''
    _, subjects = read_data()

    for s in subjects:
        x_train, _, y_train, _ = load_data_sklearn(s) 
        clf = LogisticRegression()
        clf.fit(x_train, y_train)

        dump(clf, MODELS_LOGIT + s + '.joblib')


def random_forest_classifier():
    ''' Generates a Random Forest classifier model for each 
        subject stored in models/rfc/ folder '''
    _, subjects = read_data()
    
    for s in subjects:
        x_train, _, y_train, _ = load_data_sklearn(s)
        clf = RandomForestRegressor(n_estimators=800, min_samples_split=5, min_samples_leaf=1, 
                                    max_features='sqrt', max_depth=90, bootstrap=False)
        clf.fit(x_train, y_train)
        dump(clf, MODELS_RF + s + '.joblib')
        #visualize_tree(clf, s)


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


def random_forest_classifier_tuning():
    ''' Generates a Random Forest regression model for each subject'''
    _, subjects = read_data()
    best = {}

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    for s in subjects:
        x_train, _, y_train, _ = load_data_sklearn(s) 
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, 
                                    n_iter=100, cv=3, verbose=2, random_state=42, n_jobs= 1)
        rf_random.fit(x_train, y_train)

        best[s] = rf_random.best_params_
    
    with open(MODELS_RF+'best_params.json', 'w') as fp:
        json.dump(best, fp)
    # S[0] = {'n_estimators': 800, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 90, 'bootstrap': False}
    # Improved accuracy from 87% default to 90% with best



def knn_tuning():
    data, subjects = read_data()
    
    all_results = []
    accs = []
    for i in range(50):
        acc = {}
        for k in range(1, 5):
            
            x, y = load_all_data(subjects[i], data)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, train_size=0.5)
            # x_train, y_train, x_test, y_test = load_datasets_mix(subjects[3], data)

            clf = KNeighborsClassifier(algorithm='brute', metric='minkowski', n_neighbors=k)
            
            clf.fit(x_train, y_train)
            accuracy = clf.score(x_test, y_test)
            acc[k] = accuracy
        
        best_k = (max(acc.items(), key=operator.itemgetter(1))[0])
        accs.append(acc[best_k])
        all_results.append(best_k)
            
    print(mode(all_results))
    # CALCULATED k = 2 yields highest accuracy






if __name__ == '__main__':
    # knn_classifier()
    # log_reg_classifier()
    # random_forest_classifier_tuning()
    # random_forest_classifier()
    # test_models_alone()
    random_forest_classifier_tuning()