import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def signature_graph():

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