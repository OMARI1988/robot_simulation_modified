from matplotlib import pyplot as plt
from matplotlib import cm as CM
from matplotlib import mlab as ML
from pylab import *
import numpy as np
import pickle

if __name__ == '__main__':
    f = open('/home/omari/Dropbox/robot_modified/EN/hypotheses/grammar_parsing_results2.txt', 'r')
    X = [0]
    Y1 = [0]
    Y2 = [0]
    count = 1
    for line in f:
        x = line.split('\n')[0].split('=')[0].split(',')
        y = line.split('\n')[0].split('=')[1].split(',')
        X.append(int(x[0]))
        Y1.append(int(y[0]))
        Y2.append(int(y[1]))
    #
    # X.append(int(x[0]))
    # Y1.append(0)
    # Y2.append(0)
    print Y1
    print Y2
    Y3 = []
    for i in range(len(Y2)-2):
        # print i,Y1[i]
        Y3.append(float(Y2[i]-Y1[i])*100.0/float(929-Y1[i]))
    print Y3


    fig = plt.figure()
    # fig.suptitle('A tale of 2 subplots')
    ax = fig.add_subplot(1, 2, 1)
    # l = ax.fill(X, Y2, 'r', label='Testing on all the dataset')
    ax.grid(True)
    ax.set_ylabel('Damped oscillation')
    plt.plot(X, Y2, 'r', linewidth=3,label='Testing on all the dataset')
    plt.plot(X, Y1, 'b', linewidth=3, label='Testing on a subset of the dataset')
    legend = plt.legend(loc='lower right', shadow=True, fontsize='x-large')
    xlabel('Scene number', fontsize='x-large')
    ylabel('Correctly parsed sentences number', fontsize=25)

    ax = fig.add_subplot(1, 2, 2)
    plt.plot(X[0:-2], Y3,  'g',linewidth=3, label='Percentage of correctly parsed new sentences')
    legend = plt.legend(loc='lower right', shadow=True, fontsize=25)
    # title('Incremental Language parsing in English language')
    xlabel('Scene number', fontsize=25)
    ylabel('Correctly parsed new sentence %', fontsize=25)
    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20)
    plt.grid(True)

    savefig('/home/omari/Desktop/language_parsing.png')
    plt.show()
