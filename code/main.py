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
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
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


def get_best_k(s):
    ''' Returns the calculated best k value for a given subject's KNN model'''
    with open(MODELS_KNN+'best_k.json') as json_file:
        best_k = json.load(json_file)
    
    return best_k[s]


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


# ---------------------------- KNN ----------------------------

def knn_tuning():
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

def support_vector_classifier():
    ''' Generates a Support Vector Machine classifier model for each 
        subject stored in models/svm/ folder '''
    _, subjects = read_data()
    best_params = {}
    
    for s in subjects:
        x_train, _, y_train, _= load_data_sklearn(subjects[0])
        
        #visualize_tree(clf, s)
        #param_grid = [{'kernel': ['rbf'], 'gamma': [1,10,100],'C': [1, 10, 100, 1000]}]

        param_grid = {"kernel":["linear","rbf","poly"], "C" : [i for i in range(1,1000,10)] , "degree": [1,2,3]}
        
        #Split and pass in data to train SVM
        grid_classifier = GridSearchCV(SVC(gamma = "auto",decision_function_shape="ovo"), param_grid, scoring = 'recall_macro', cv = 5)
        grid_classifier.fit(x_train,y_train)

        # Needed if we want to take note of statistical data of each model 
        # print("\nGrid scores on development set:")
        # print()
        # means = grid_classifier.cv_results_['mean_test_score']
        # stds = grid_classifier.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, grid_classifier.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #         % (mean, std * 2, params))

        best_params[s] = (grid_classifier.best_params_)
        
        # print(classification_report(y_test, y_pred,target_names= ["user", "intruder"]))
        dump(grid_classifier, MODELS_SVM + s + '.joblib')

    with open(MODELS_SVM+'best_params.json', 'w') as fp:
        json.dump(best_params, fp)

    # return grid_classifier.best_estimator_.score, grid_classifier.best_estimator_




# ---------------------------- EVALUATION ----------------------------

def evaluate_all_models_alone():
    ''' Evaluates all models alone and generates results files '''
    models = [('knn', False), ('logistic', False), ('randfor', True)]
    for m in models:
        evaluate_model(m[0], m[1])



def evaluate_model(name, check_pred):
    ''' _Evaluates the model named by calculating the false alarm and miss rates 
        Params: name - name of model, check_pred - True if model returns probability not label '''
    _, subjects = read_data()

    miss_rate = []
    f_alarm_rate = []
    accs = []
    results = {}

    for s in subjects:
        _, x_test, _, y_test = load_data_sklearn(s)
        n = len(y_test)
        miss = 0
        f_a = 0

        clf = load_model(name, s)
        y_pred = clf.predict(x_test)

        for i in range(n):
            if check_pred:
                if y_pred[i] >= 0.5 and y_test[i] == -1.0:
                    miss += 1
                elif y_pred[i] < 0.5 and y_test[i] == 1.0:
                    f_a += 1

            else:
                if y_pred[i] == 1.0 and y_test[i] == -1.0:
                    miss += 1
                elif y_pred[i] == -1.0 and y_test[i] == 1.0:
                    f_a += 1
        
        miss_rate.append(miss / n)
        f_alarm_rate.append(f_a / n)
        accs.append(clf.score(x_test, y_test))

    results['miss rate mean'] = np.array(miss_rate).mean()
    results['miss rate SD'] = np.array(miss_rate).std()
    results['false alarm rate mean'] = np.array(f_alarm_rate).mean()
    results['false alarm rate SD'] = np.array(f_alarm_rate).std()
    results['accuracy mean'] = np.array(accs).mean()
    results['accuracy SD'] = np.array(accs).std()

    with open(RESULTS_ALONE + name + '.json', 'w') as fp:
        json.dump(results, fp)



def evaluate_majority_vote(models):
    ''' Evaluates the performance of using multiple classifiers and taking a majority vote'''
    _, subjects = read_data()
    num_models = len(models)

    miss_rate = []
    f_alarm_rate = []
    accs = []
    results = {}
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
        
        n = len(y_test)
        miss = 0
        f_a = 0
        
        for i in range(n):
            votes = []
            for p in preds:
                votes.append(p[i])

            if sum(votes) >= 1.0:
                classify = 1.0
            else:
                classify = -1.0
            
            if classify == 1.0 and y_test[i] == -1.0:
                    miss += 1
            elif classify == -1.0 and y_test[i] == 1.0:
                f_a += 1

        miss_rate.append(miss / n)
        f_alarm_rate.append(f_a / n)
        accs.append((n - miss - f_a)/n)

    results['miss rate mean'] = np.array(miss_rate).mean()
    results['miss rate SD'] = np.array(miss_rate).std()
    results['false alarm rate mean'] = np.array(f_alarm_rate).mean()
    results['false alarm rate SD'] = np.array(f_alarm_rate).std()
    results['accuracy mean'] = np.array(accs).mean()
    results['accuracy SD'] = np.array(accs).std()
    
    fname = fname[:-1] + '.json'
    with open(fname, 'w') as fp:
        json.dump(results, fp)
            
            
    

def run_majority_votes():
    models = [('knn', False), ('logistic', False), ('randfor', True)]
    evaluate_majority_vote(models)




if __name__ == '__main__':
    # random_forest_classifier_tuning()
    # knn_tuning()
    # knn_classifier()
    # evaluate_model('knn', False)
    # support_vector_classifier()
    # evaluate_model('svm', False)
    # evaluate_all_models_alone()
    # log_reg_tuning()
    # log_reg_classifier()
    # evaluate_model('logistic', False)
    run_majority_votes()




# def SVM():
#     print("---------------Running Training---------------")
#     accuracies = []
#     models = []
#     #Training on different Cross Fold Validation sets
#     for i in range(1,6):
#         accuracies[i],models[i] = trainSVM(i)

#     print("-------------All Estimator Scores-------------")
#     for i in accuracies:
#         print(i)
    
#     print("----------------Best Estimator-----------------")
#     print("Model:")
#     print(models[accuracies.index(max(accuracies))])
#     print("Score:",max(accuracies))
