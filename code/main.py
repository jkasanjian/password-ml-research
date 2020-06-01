import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import operator
from os import mkdir
from joblib import dump, load
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



DATA_SOURCE = 'data/DSL-StrongPasswordData.csv'
DATA_JSON = 'data/password_data.json'
DATA_SPLIT_SK = 'data/split/sklearn/'
MODELS_LOGIT = 'models/logistic/'
MODELS_KNN = 'models/knn/'
USER_GRAPHS = 'data/graphs/'
RESULTS = 'results/'


def read_data():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)
    # features = data['subject']
    del data['subject']
    subjects = list(data.keys())
    
    return data, subjects


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



def load_datasets_pos(subject, data):
    ''' Loads train and test datasets for a given subject 
        using only POSITIVE data for train
        train: first 200 repititions of user
        test_user: last 200 repititions of user
    ''' 
    train_user = data[subject][:200]
    test_user = data[subject][200:]
    test_imposter = []
    for s in data:
        if s != subject:
            test_imposter.extend(data[s][:5])

    x_train = train_user
    y_train = [1 for i in range(len(x_train))]
    x_test = test_user + test_imposter
    y_test = [1 for i in range(len(test_user))] + [-1 for i in range(len(test_imposter))]

    # x_train, y_train = shuffle(x_train, y_train)
    # x_test, y_test = shuffle(x_test, y_test)

    return (np.array(x_train).astype(np.float64), np.array(y_train).astype(np.float64), 
    np.array(x_test).astype(np.float64), np.array(y_test).astype(np.float64)) 



def load_datasets_mix(subject, data):
    ''' Loads train and test datasets for a given subject 
        using POSITIVE AND NEGATIVE data for user
        train: first 200 repititions of user, last 5 from each imposter
        test_user: last 200 repititions of user, first 5 from each imposter
    ''' 
    train_user = data[subject][:200]
    train_imposter = []
    test_user = data[subject][200:]
    test_imposter = []
    for s in data:
        if s != subject:
            train_imposter.extend(data[s][-5:])
            test_imposter.extend(data[s][:5])

    x_train = train_user + train_imposter
    y_train = [1 for i in range(len(train_user))] + [-1 for i in range(len(train_imposter))]
    x_test = test_user + test_imposter
    y_test = [1 for i in range(len(test_user))] + [-1 for i in range(len(test_imposter))]
    # x_train, y_train = shuffle(x_train, y_train)
    # x_test, y_test = shuffle(x_test, y_test)
    return (np.array(x_train).astype(np.float64), np.array(y_train).astype(np.float64), 
    np.array(x_test).astype(np.float64), np.array(y_test).astype(np.float64))



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
        stored in models/logistic/ folder'''
    _, subjects = read_data()

    for s in subjects:
        x_train, _, y_train, _ = load_data_sklearn(s) 
        clf = LogisticRegression()
        clf.fit(x_train, y_train)

        dump(clf, MODELS_LOGIT + s + '.joblib')




'''
# --------------------TESTING BELOW--------------------

def test_knn():
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
    print(sum(accs)/len(accs))
    # CALCULATED K = 2 yields highest accuracy


def test_logistic_regression():
    data, subjects = read_data()
    x, y = load_all_data(subjects[8], data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, train_size=0.5)
    # x_train, y_train, x_test, y_test = load_datasets_mix(subjects[8], data)

    print(x_train.shape)
    print(x_test.shape)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    print('Accuracy LOGIT:',accuracy)

# --------------------TESTING ABOVE--------------------
'''





if __name__ == '__main__':
    # test_knn()
    # test_logistic_regression()
    knn_classifier()
    log_reg_classifier()
