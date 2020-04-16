import csv
import json
import matplotlib.pyplot as plt
import numpy as np



DATA_SOURCE = 'data/DSL-StrongPasswordData.csv'
DATA_JSON = 'data/password_data.json'
USER_GRAPHS = 'data/graphs/'


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


def load_datasets(subject, data):
    ''' Loads train and test datasets for a given subject 
        train: first 200 repititions of user
        test_user: last 200 repititions of user
    ''' 
    train = data[subject][:200]
    test_user = data[subject][200:]
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



def test():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)

    subjects = data.keys()

    train, test_user, test_imposter = load_datasets(subjects[0], data)
    




if __name__ == '__main__':
    signature_graph()


