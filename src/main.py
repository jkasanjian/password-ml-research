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
DATA_SPLIT = 'data/split/'

MODELS_LOGIT = 'models/logistic/'
MODELS_KNN = 'models/knn/'
MODELS_RF = 'models/randfor/'
MODELS_SVM = 'models/svm/'

USER_GRAPHS = 'data/graphs/'
TREE_GRAPHS = 'data/trees/'
RESULTS_ALONE = 'results/alone/'
RESULTS_COMBO = 'results/combo/'




###########################################################################
# ---------------------------- DATA PROCESSING ----------------------------
###########################################################################

# def read_data():
#     with open(DATA_JSON) as json_file:
#         data = json.load(json_file)
#     # features = data['subject']
#     del data['subject']
#     subjects = list(data.keys())
#     return data, subjects


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
    x,y - all 400 repetitions of subject, all 20000 imposter sets
    '''
    x_user = data[subject]
    x_imposter = []
    for s in data:
        if s != subject:
            x_imposter.extend(data[s])

    y_user = [1 for i in range(len(x_user))]
    y_imposter = [-1 for i in range(len(x_imposter))]
    print('\n\nSubject', subject)
    print('number user:',len(x_user))
    print('number imposter:',len(x_imposter))
    return x_user, x_imposter, y_user, y_imposter
    

def split_data_wval():
    '''
    Splits the data into 60% train, 20% validate, 20% test
    For each set, we randomly select frn 
    '''
    data, subjects = read_data()
    path = DATA_SPLIT + 'w-val/'
    for s in subjects:
        mkdir(path + s)

        x_user_0, x_imposter_0, y_user_0, y_imposter_0 = load_all_data(s, data)
        x_user = x_user_0.copy()
        x_imposter = x_imposter_0.copy()
        y_user = y_user_0.copy()
        y_imposter = y_imposter_0.copy()
        n_u = len(x_user)
        n_i = len(x_imposter)

        num_train_u = int(0.60 * n_u)
        num_val_u = int(0.20 * n_u)
        num_test_u = int(0.20 * n_u)
        num_train_i = int(0.60 * n_i)
        num_val_i = int(0.20 * n_i)
        num_test_i = int(0.20 * n_i)
        x_train = []
        x_val = []
        x_test = []
        y_train = []
        y_val = []
        y_test = []

        for c in range(num_train_u):
            i = np.random.randint(0, len(x_user))
            x_train.append(x_user.pop(i))
            y_train.append(y_user.pop(i))
        for c in range(num_train_i):
            i = np.random.randint(0, len(x_imposter))
            x_train.append(x_imposter.pop(i))
            y_train.append(y_imposter.pop(i))

        for c in range(num_val_u):
            i = np.random.randint(0, len(x_user))
            x_val.append(x_user.pop(i))
            y_val.append(y_user.pop(i))
        for c in range(num_val_i):
            i = np.random.randint(0, len(x_imposter))
            x_val.append(x_imposter.pop(i))
            y_val.append(y_imposter.pop(i))

        for c in range(num_test_u):
            i = np.random.randint(0, len(x_user))
            x_test.append(x_user.pop(i))
            y_test.append(y_user.pop(i))
        for c in range(num_test_i):
            i = np.random.randint(0, len(x_imposter))
            x_test.append(x_imposter.pop(i))
            y_test.append(y_imposter.pop(i))
        
        x_train, y_train = shuffle(x_train, y_train)
        x_test, y_test = shuffle(x_test, y_test)
        x_val, y_val = shuffle(x_val, y_val)

        x_train = np.array(x_train).astype(np.float64)
        x_val = np.array(x_val).astype(np.float64)
        x_test = np.array(x_test).astype(np.float64)
        y_train = np.array(y_train).astype(np.float64)
        y_val = np.array(y_val).astype(np.float64)
        y_test = np.array(y_test).astype(np.float64)

        np.save(path + s + '/x_train.npy', x_train)
        np.save(path + s + '/x_val.npy', x_val)
        np.save(path + s + '/x_test.npy', x_test)
        np.save(path + s + '/y_train.npy', y_train)
        np.save(path + s + '/y_val.npy', y_val)
        np.save(path + s + '/y_test.npy', y_test)


def split_data_nval():
    ''' Splits the data into 70% train and 30% test '''
    data, subjects = read_data()
    path = DATA_SPLIT + 'n-val/'
    for s in subjects:
        mkdir(path + s)

        x_user_0, x_imposter_0, y_user_0, y_imposter_0 = load_all_data(s, data)
        x_user = x_user_0.copy()
        x_imposter = x_imposter_0.copy()
        y_user = y_user_0.copy()
        y_imposter = y_imposter_0.copy()
        n_u = len(x_user)
        n_i = len(x_imposter)
        print('number user:',n_u)
        print('number imposter:',n_i)
        num_train_u = int(0.70 * n_u)
        num_test_u = int(0.30 * n_u)
        num_train_i = int(0.70 * n_i)
        num_test_i = int(0.30 * n_i)
        x_train = []
        x_test = []
        y_train = []
        y_test = []

        for c in range(num_train_u):
            i = np.random.randint(0, len(x_user))
            x_train.append(x_user.pop(i))
            y_train.append(y_user.pop(i))
        for c in range(num_train_i):
            i = np.random.randint(0, len(x_imposter))
            x_train.append(x_imposter.pop(i))
            y_train.append(y_imposter.pop(i))

        for c in range(num_test_u):
            i = np.random.randint(0, len(x_user))
            x_test.append(x_user.pop(i))
            y_test.append(y_user.pop(i))
        for c in range(num_test_i):
            i = np.random.randint(0, len(x_imposter))
            x_test.append(x_imposter.pop(i))
            y_test.append(y_imposter.pop(i))
        
        x_train, y_train = shuffle(x_train, y_train)
        x_test, y_test = shuffle(x_test, y_test)

        x_train = np.array(x_train).astype(np.float64)
        x_test = np.array(x_test).astype(np.float64)
        y_train = np.array(y_train).astype(np.float64)
        y_test = np.array(y_test).astype(np.float64)

        np.save(path + s + '/x_train.npy', x_train)
        np.save(path + s + '/x_test.npy', x_test)
        np.save(path + s + '/y_train.npy', y_train)
        np.save(path + s + '/y_test.npy', y_test)


def load_data_wval(s):
    ''' Loads datasets for a subject s (train, test, val)'''
    path = DATA_SPLIT + 'w-val/'
    x_train = np.load(path + s + '/x_train.npy')
    x_val   = np.lpad(path + s + '/x_val.npy')
    x_test  = np.load(path + s + '/x_test.npy')
    y_train = np.load(path + s + '/y_train.npy')
    y_val   = np.load(path + s + '/y_val.npy')
    y_test  = np.load(path + s + '/y_test.npy')
    return x_train, x_val, x_test, y_train, y_val, y_test
    

# def load_data_nval(s):
#     ''' Loads datasets for a subject s (train, test)'''
#     path = DATA_SPLIT + 'n-val/'
#     x_train = np.load(path + s + '/x_train.npy')
#     x_test  = np.load(path + s + '/x_test.npy')
#     y_train = np.load(path + s + '/y_train.npy')
#     y_test  = np.load(path + s + '/y_test.npy')
#     return x_train, x_test, y_train, y_test


# def load_model(name, s):
#     ''' Returns the corresponding model for a subject '''
#     return load('models/' + name + '/' + s + '.joblib')



def shuffle(x, y):
    ''' Shuffles the order of x/y pairs in the two arrays'''
    for i in range(5 * len(x)):
        i_1 = np.random.randint(0, len(x))
        i_2 = np.random.randint(0, len(x))
        temp_x = x[i_1]
        x[i_1] = x[i_2]
        x[i_2] = temp_x
        temp_y = y[i_1]
        y[i_1] = y[i_2]
        y[i_2] = temp_y
    
    return x, y

###############################################################
# ---------------------------- KNN ----------------------------
###############################################################

# def knn_training():
#     ''' Calculates and saves the best hyperparameteres for each subject's KNN model '''
#     _, subjects = read_data()

#     leaf_size = list(range(1,50))
#     n_neighbors = list(range(1,30))
#     p = [1,2]
#     hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

#     for s in subjects:
#         x_train, _, y_train, _ = load_data_nval(s)

#         knn_tune = KNeighborsClassifier(algorithm='brute', metric='minkowski')
#         clf = GridSearchCV(knn_tune, hyperparameters, scoring='f1', n_jobs=-1)
#         clf.fit(x_train, y_train)

#         dump(clf, MODELS_KNN + s + '.joblib')
        



###############################################################################
# ---------------------------- Logistic Regression ----------------------------
###############################################################################

# def log_reg_training():
#     '''' Calculates and saves the best hyperparameteres for each subject's
#     Logistic Regression model '''
#     _, subjects = read_data()
    
#     penalty = ['l1', 'l2', 'elasticnet', 'none']
#     C = np.logspace(-4, 4, 20)
#     solver = ['lbfgs','newton-cg','liblinear','sag','saga']
#     max_iter = [100, 1000, 2500, 5000]
#     hyperparameteres = dict(penalty=penalty, C=C, solver=solver, max_iter=max_iter)

#     for s in subjects:
#         x_train, _, y_train, _ = load_data_nval(s)
        
#         log_tune = LogisticRegression()
#         clf = GridSearchCV(log_tune, hyperparameteres, scoring='f1', n_jobs=-1)
#         clf.fit(x_train, y_train)
       
#         dump(clf, MODELS_LOGIT + s + '.joblib')



#########################################################################
# ---------------------------- Random Forest ----------------------------
#########################################################################

# def random_forest_training():
#     ''' Generates a Random Forest regression model for each subject'''
#     _, subjects = read_data()
   
#     n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
#     max_features = ['auto', 'sqrt']
#     max_depth = list(range(50, 100, 10))
#     max_depth.append(None)
#     min_samples_split = [2, 5, 10]
#     min_samples_leaf = [1, 2, 4]
#     bootstrap = [True, False]
#     hyperparameteres = dict(
#                 n_estimators=n_estimators,
#                 max_features=max_features,
#                 max_depth=max_depth,
#                 min_samples_split=min_samples_split,
#                 min_samples_leaf=min_samples_leaf,
#                 bootstrap=bootstrap)

#     for s in subjects:
#         x_train, _, y_train, _ = load_data_nval(s) 

#         rf_tune = RandomForestClassifier()
#         clf = GridSearchCV(rf_tune, hyperparameteres, scoring='f1', n_jobs= -1)
#         clf.fit(x_train, y_train)
        
#         dump(clf, MODELS_RF + s + '.joblib')
#         # vizualize_tree(clf, s)


# def visualize_tree(clf, s):
#     ''' Creates a graphics diagram of a tree from the random forest '''
#     estimator = clf.estimators_[5]

#     export_graphviz(estimator, out_file=TREE_GRAPHS+s+'tree.dot', 
#                 feature_names = get_features(),
#                 class_names = ['genuine user', 'imposter'],
#                 rounded = True, proportion = False, 
#                 precision = 2, filled = True)

#     check_call(['dot', '-Tpng', TREE_GRAPHS+s+'tree.dot', '-o', TREE_GRAPHS+s+'tree.png', '-Gdpi=600'])
#     Image(filename=TREE_GRAPHS+s+'tree.png')
#     remove(TREE_GRAPHS+s+'tree.dot')



###############################################################
# ---------------------------- SVM ----------------------------
###############################################################


# def svm_training():
#     ''' Calculates and saves the best hyperparameteres for each subject's SVM model '''
#     _, subjects = read_data()

#     kernel = ["linear","rbf","poly"]
#     C = [i for i in range(1,1000,10)]
#     degree = [1,2]

#     hyperparameteres = dict(kernel=kernel, C=C, degree=degree)
    
#     for s in subjects:
#         x_train, _, y_train, _= load_data_nval(s)

#         svm_tune = SVC(gamma ='auto',decision_function_shape='ovo')
#         clf = GridSearchCV(svm_tune, hyperparameteres, scoring='f1', n_jobs=-1)
#         clf.fit(x_train, y_train)

#         dump(clf, MODELS_SVM + s + '.joblib')



######################################################################
# ---------------------------- EVALUATION ----------------------------
######################################################################

def evaluate_all_models_alone():
    ''' Evaluates all models alone and generates results files '''
    models = ('knn', 'logistic', 'randfor', 'svm')
    for m in models:
        evaluate_model(m)

    
def run_majority_votes():
    models = [  ['knn', 'logistic', 'randfor'],
                ['knn', 'logistic', 'svm'],
                ['knn', 'randfor', 'svm'],
                ['logistic', 'randfor', 'svm'] ]
    
    for m_set in models:
        evaluate_majority_vote(m_set)

def evaluate_model(name):
    ''' _Evaluates the model named by calculating the false rejection and false acceptance rates 
        Params: name - name of model '''
    _, subjects = read_data()
    results = {}
    results['subjects'] = {}
    all_miss_rate = []
    all_f_alarm_rate = []
    all_pers = []
    all_recs = []
    all_f1s = []

    for s in subjects:
        _, x_test, _, y_test = load_data_nval(s)

        s_results = {}
        n = len(y_test)
        miss = 0
        f_a = 0
        total_pos = 0

        clf = load_model(name, s)
        y_pred = clf.predict(x_test)

        for i in range(n):

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
        f1 = 2 * ((precision * recall) / (precision + recall))

        s_results['false acceptance rate'] = miss_rate
        all_miss_rate.append(miss_rate)
        s_results['false rejection rate'] = false_alarm_rate
        all_f_alarm_rate.append(false_alarm_rate)
        s_results['precision'] = precision
        all_pers.append(precision)
        s_results['recall'] = recall
        all_recs.append(recall)
        s_results['f1'] = f1
        all_f1s.append(f1)
        
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
    all_pers = []
    all_recs = []
    all_f1s = []
    fname = RESULTS_COMBO

    first = True
    for s in subjects:
        clfs = []
        for m in models:
            clfs.append(load_model(m, s))
            if first:
                fname += m + '-' 
        first = False
        
        _, x_test, _, y_test = load_data_nval(s)
        
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
        f1 = 2 * ((precision * recall) / (precision + recall))

        s_results['false acceptance rate'] = miss_rate
        all_miss_rate.append(miss_rate)
        s_results['false rejection rate'] = false_alarm_rate
        all_f_alarm_rate.append(false_alarm_rate)
        s_results['precision'] = precision
        all_pers.append(precision)
        s_results['recall'] = recall
        all_recs.append(recall)
        s_results['f1'] = f1
        all_f1s.append(f1)
        
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
    results['win count'] = win_count
    
    fname = fname[:-1] + '.json'
    with open(fname, 'w') as fp:
        json.dump(results, fp)
            
            
def print_all_stats():
    dash = '-' * 40
    for path, _, files in os.walk('results'):
        for name in files:
            file = os.path.join(path, name)
            if file[-5:] == '.json':
                with open(file) as json_file:
                    data = json.load(json_file)
                    print('\n\n\nResults for', name[:-5])
                    print(dash)

                    print('{:<30s}{:<2.8f}'.format('F1 mean:', 100*data['f1 mean']))
                    print('{:<30s}{:<2.8f}'.format('F1 SD:', 100*data['f1 SD']))
                    print()
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
                    




################################################################
# ---------------------------- MAIN ----------------------------
################################################################
if __name__ == '__main__':
    # split_data_wval()
    # split_data_nval()

    knn_training()
    print('KNN done training')
    log_reg_training()
    print('Logistic regression done training')
    random_forest_training()
    print('Random forest done training')
    svm_training()
    print('SVM done training')

    evaluate_all_models_alone()
    run_majority_votes()
    print_all_stats()
    