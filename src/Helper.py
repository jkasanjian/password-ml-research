import numpy as np
import json
import os
from AUROC import AUROC
from sklearn import metrics
from joblib import dump, load


DATA_JSON = 'data/password_data.json'

def read_data():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)
    # features = data['subject']
    del data['subject']
    subjects = list(data.keys())
    return data, subjects

def directoryExist(name):
    if not os.path.isdir(name):
        os.makedirs(name)


def load_model(name, s):
    ''' Returns the corresponding model for a subject '''
    return load('src/models/all_data/' + s + "/" + name + ".joblib")


def get_test_data(subject, is_balanced):
    ''' Returns the testing data partition for the given subject.
        is_balanced is a boolean field. if true, returns test data
        that is classed-balanced. if false, returns test data with
        the same class proportions as the entire dataset ''' 

    if is_balanced:
        path = 'data/partitions/balanced_data/'
    else:
        path = 'data/partitions/all_data/'
    x_test = np.load(path + subject + '/x_test.npy')
    y_test = np.load(path + subject + '/y_test.npy')

    return x_test, y_test


def get_train_data(subject, is_balanced):
    ''' Returns the training data partition for the given subject.
        is_balanced is a boolean field. if true, returns train data
        that is classed-balanced. if false, returns train data with
        the same class proportions as the entire dataset ''' 

    if is_balanced:
        path = 'data/partitions/balanced_data/'
    else:
        path = 'data/partitions/all_data/'
    x_train = np.load(path + subject + '/x_train.npy')
    y_train = np.load(path + subject + '/y_train.npy')

    return x_train, y_train

def get_results(subjects, model_name, model_type):
    
    results = {}
    all_miss_rate = []
    all_f_alarm_rate = []
    all_pers = []
    all_recs = []
    all_f1s = []
    results['subjects'] = {}

    for s in subjects:
        X_test, Y_test = get_test_data(s,True)
        model = load_model(model_name,s)
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
        path = "src/models/all_data/" + s +"/"+model_type+"_PNG/"
        directoryExist(path)
        charts = AUROC(model,s,path+model_name,X_test,Y_test)
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
    print("\n\n\n\n\n\n\n--------------------Final Scores--------------------")
    print('false acceptance rate mean:',results['false acceptance rate mean'])
    print('false acceptance rate SD:',results['false acceptance rate SD'])
    print('false rejection rate mean:',results['false rejection rate mean'])
    print('false rejection rate SD:',results['false rejection rate SD'])
    print('Precision mean:',results['precision mean'])
    print('Precision SD',results['precision SD'])
    print('F1 mean',results['f1 mean'],"\n\n\n")


if __name__ == "__main__":
    ''' Main method '''
    # test models