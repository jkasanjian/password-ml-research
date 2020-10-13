from os import path, mkdir
import numpy as np


def get_test_data(subject, is_balanced):
    ''' Returns the testing data partition for the given subject.
        is_balanced is a boolean field. if true, returns test data
        that is classed-balanced. if false, returns test data with
        the same class proportions as the entire dataset ''' 

    if is_balanced:
        path = 'data/partitions/balanced/'
    else:
        path = 'data/partitions/all/'
    x_test = np.load(path + subject + '/x_test.npy')
    y_test = np.load(path + subject + '/y_test.npy')

    return x_test, y_test







if __name__ == "__main__":
    ''' Main method '''
    # test models