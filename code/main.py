import csv
import json
import numpy as np


DATA_SOURCE = 'data/DSL-StrongPasswordData.csv'
DATA_JSON = 'data/password_data.json'


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

    del data['subject']
    with open(DATA_JSON, 'w') as fp:
        json.dump(data, fp)


def load_datasets(subject, data):
    ''' Loads train and test datasets for a given subject 
        train: first 200 repititions of user
        test_user: last 200 repititions of user
    ''' 
    train = data[subject][:350]
    test_user = data[subject][350:]
    test_imposter = []
    for s in data:
        if s != subject:
            test_imposter.extend(data[s][:5])

    train = convert_to_float(train)
    test_user = convert_to_float(test_user)
    test_imposter = convert_to_float(test_imposter)

    return train, test_user, test_imposter    


def convert_to_float(rows):
    for i in range(len(rows)):
        rows[i] = [float(r) for r in rows[i]]
    return rows



def visualize_data():
    ''' Creates visualization of each subject ''' 
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)

    for subject in data:    
        pass    # subject is the string key, data[subject] is list of trial rows (400 each)




def test():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)

    subjects = list(data.keys())
    train, test, test_imposter = load_datasets(subjects[0], data)
    # train = train[:20] + test_imposter[:40] + train[20:60] + test_imposter[40:50]
    test_imposter = np.array(test_imposter)
    train = np.array(train)
    test = np.array(test)
    print(test_imposter.shape)
    return train,test,test_imposter
    
    

def getData():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)

    subjects = list(data.keys())
    train, test, test_imposter = load_datasets(subjects[0], data)
    X = np.array(train + test_imposter)
    y = list()
    z = np.array(test)
    # for i in range(len(train) + len(test) + len(test_imposter)):
    #     if(i < 400):
    #         y.append(1.0)
    #     else: 
    #         y.append(0.0)
    for i in range(len(train) + len(test_imposter)):
        if(i < 350):
            y.append(1.0)
        else: 
            y.append(0.0)
    return X,y,z

if __name__ == '__main__':

    test()