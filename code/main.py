import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_SOURCE = 'data/DSL-StrongPasswordData.csv'
DATA_JSON = 'data/password_data.json'
x=[]
y=[]


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



def test():
    with open(DATA_JSON) as json_file:
        data = json.load(json_file)

    subjects = data.keys()

    train, test_user, test_imposter = load_datasets(subjects[0], data)
    

def signature_graph():
<<<<<<< HEAD
=======

    with open('data/individual-user-datum/s002.csv', 'r') as data:
        plots= csv.reader(data, delimiter=',')
        is_title_row = True
        row_count = 0
        data_list = []
        y_axis = []

        for row in plots:

            # Row is just column names
            if(is_title_row):
                print(row)
                title_list = row
                is_title_row = False
                row_count += 1
                print("\n\nActual data:")
            else:
                data_list.append(row[3:])
                row_count += 1
           
        
        for i in range(len(title_list)):
            print(data_list[i])
            for j in range(len(data_list[0])):
                z = 0 # Debug val

            
   
    # plt.plot(x,y, marker='o')

    # plt.title('s002 Typing Signature')
>>>>>>> 4280efc927449c83c8712fd8e9b9f81c69bae5f9

    #x,y = np.loadtxt('data/individual-user-datum/s002.csv', unpack = True, delimiter = ',')
    my_data = np.genfromtxt('data/individual-user-datum/s002.csv', delimiter=',')
    print(my_data)
    
    #plt.plot(x,y)
    #plt.show()
    
    # with open('data/individual-user-datum/s002.csv', 'r') as data:
    #     plots= csv.reader(data, delimiter=',')
    #     is_title_row = True
    #     row_count = 0
    #     for row in plots:

    #         # Row is just column names
    #         if(is_title_row):
    #             print(row)
    #             title_list = row
    #             is_title_row = False
    #             row_count += 1
    #             print("\n\nActual data:")
    #         else:
    #             print(row[3:]) # Cut off first 3 columns
    #             row_count += 1



if __name__ == '__main__':
    signature_graph()