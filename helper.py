import pickle
import numpy as np


# Сургалтын явцыг харуулах
def print_progress(count, total, bar_len=60, loss='', accuracy=''):
    percent = round(100.0 * count / float(total), 1)
    filled_len = int(round(bar_len * count / float(total)))
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('\r[{0}] {1}% {2}/{3} loss*: {4:.3f} accuracy*: {5:.3f}%\r'.format(bar, percent, count, total, loss, accuracy), flush=True, end='\r')


# Neural Network файлаас унших
def nn_load(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


# Neural Network файлд хадгалах
def nn_save(filename, nn):
    with open(filename, 'wb') as output:
        pickle.dump(nn, output, pickle.HIGHEST_PROTOCOL)


# One Hot Encoding
def one_hot_encoding(classes, labels):
    res = []
    for (i, v) in enumerate(labels):
        temp = np.zeros(len(classes))
        temp[v] = 1
        res.append(temp)
    return res
