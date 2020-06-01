import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import operator

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



DATA_SOURCE = 'data/DSL-StrongPasswordData.csv'
DATA_JSON = 'data/password_data.json'
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
    '''
    user = data[subject]
    imposter = []
    for s in data:
        if s != subject:
            imposter.extend(data[s])

    x = user + imposter
    y = [1 for i in range(len(user))] + [-1 for i in range(len(imposter))]

    return np.array(x).astype(np.float64), np.array(y).astype(np.float64)



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


def signature_graph():
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



def test_knn():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)

    # features = data['subject']

    del data['subject']
    subjects = list(data.keys())
    n = 1

    x, y = load_all_data(subjects[6], data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.02, train_size=.02)
    print('all data:',x.shape)
    print('train data:',x_train.shape)
    print('test data:',x_test.shape)
    clf = KNeighborsClassifier(algorithm='brute',
                                         metric='mahalanobis',
                                         metric_params={'V': np.cov(x_train)},
                                         n_neighbors=n)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    print('Accuracy (sklearn):',accuracy)


    '''
    MANUAL DATA SPLITTING not very accurate (not sure why)
    Training with all positive data, 0.4-0.5 accuracy
    Training with mixed datam 0.6-0.8 accuracy (shown below)

    x_train, y_train, x_test, y_test = load_datasets_mix(subjects[8], data)
    clf = KNeighborsClassifier(algorithm='brute',
                                         metric='mahalanobis',
                                         metric_params={'V': np.cov(x_train)},
                                         n_neighbors=n)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    print('Accuracy (pos data):',accuracy)
    '''


def test_logistic_regression():
    data, subjects = read_data()
    x, y = load_all_data(subjects[6], data)

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.02, train_size=.02)
    # again, more accurate with automatic splitting
    x_train, y_train, x_test, y_test = load_datasets_mix(subjects[0], data)
    print('all data:',x.shape)
    print('train data:',x_train.shape)
    print('test data:',x_test.shape)

    print('first x', x_train[0])
    print('first y', y_train[0])

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    print('Accuracy:',accuracy)




def KNN_classifier():
    
    data, subjects = read_data()
    k = 2   # yielded highest accuracy
    results = {}

    for s in subjects:
        x, y = load_all_data(s, data)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.02, train_size=0.02)    
        clf = KNeighborsClassifier(algorithm='brute',
                                   metric='mahalanobis',
                                   metric_params={'V': np.cov(x_train)},
                                   n_neighbors=k)
        clf.fit(x_train, y_train)
        accuracy = clf.score(x_test, y_test)
        results[s] = accuracy

    with open(RESULTS + 'KNN_automatic_split', 'w') as fp:
        json.dump(results, fp)



def log_reg_classifier():
    
    data, subjects = read_data()
    results = {}

    for s in subjects:
        x, y = load_all_data(s, data)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.02, train_size=0.02)    
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        accuracy = clf.score(x_test, y_test)
        results[s] = accuracy

    with open(RESULTS + 'Logistic_automatic_split', 'w') as fp:
        json.dump(results, fp)





if __name__ == '__main__':
    # KNN_classifier()
    # log_reg_classifier()
    test_logistic_regression()

    pass