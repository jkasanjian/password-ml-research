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
    all_AUROC_scores = []
    all_pers = []
    all_recs = []
    all_f1s = []
    results['subjects'] = {}

    for s in subjects:
        #### Creates directory for PNG files if there is not one already
        path = "src/models/all_data/" + s +"/"+model_type+"_PNG/"
        directoryExist(path)
        #### 

        X_test, Y_test = get_test_data(s,True)

        model = load_model(model_name,s)
        X_test = X_test.astype(np.float)
        Y_test = Y_test.astype(np.float)
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
        charts = AUROC(model,path+model_name,X_test,Y_test)
        auc_score = charts.getAUC()
        s_results['auc'] = auc_score
        all_AUROC_scores.append(auc_score)
        results['subjects'][s] = s_results

        # printUserResults(s,miss_rate,false_alarm_rate, precision, recall, f1, auc_score)
        print("Accuracy:",metrics.accuracy_score(Y_pred,Y_test))
        
        
    results['false acceptance rate mean'] = np.array(all_miss_rate).mean()
    results['false acceptance rate SD'] = np.array(all_miss_rate).std()
    results['false rejection rate mean'] = np.array(all_f_alarm_rate).mean()
    results['false rejection rate SD'] = np.array(all_f_alarm_rate).std()
    results['precision mean'] = np.array(all_pers).mean()
    results['precision SD'] = np.array(all_pers).std()
    results['recall mean'] = np.array(all_recs).mean()
    results['recall SD'] = np.array(all_recs).std()
    results['auc mean'] = np.array(all_AUROC_scores).mean()
    results['auc SD'] = np.array(all_AUROC_scores).std()
    results['f1 mean'] = np.array(all_f1s).mean()
    results['f1 SD'] = np.array(all_f1s).std()
    printResults(results)


def printResults(results):
    print("\n\n\n\n\n\n\n--------------------Final Scores--------------------")
    print('False acceptance rate mean:',results['false acceptance rate mean'])
    print('False acceptance rate SD:',results['false acceptance rate SD'])
    print('False rejection rate mean:',results['false rejection rate mean'])
    print('False rejection rate SD:',results['false rejection rate SD'])
    print('Precision mean:',results['precision mean'])
    print('Precision SD',results['precision SD'])
    print('Recall mean:',results['recall mean'])
    print('Recall SD',results['recall SD'])
    print('AUC mean',results['auc mean'])
    print('AUC SD',results['auc SD'])
    print('F1 mean',results['f1 mean'])
    print('F1 SD',results['f1 SD'],"\n\n\n")


def printUserResults(s,miss_rate,false_alarm_rate,precision,recall,f1,auc_score):
    print("\n--------------------",s,"--------------------")
    print('False acceptance rate',miss_rate)
    print('False rejection rate', false_alarm_rate)
    print('Precision',precision)
    print('Recall',recall)
    print('F1',f1)
    print('Area under the curve',auc_score)

if __name__ == "__main__":
    ''' Main method '''
    # test models