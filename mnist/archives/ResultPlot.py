import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rff
import csv
import rff

def trials_agg(trials, folder, prefix, suffix):
    content = list()
    for idx in trials:
        sourcename = folder + prefix + str(idx) + suffix + '.csv'
        with open(sourcename,'r') as sourcefile:
            datareader = csv.reader(sourcefile)
            for row in datareader:
                content.append(row)
    targetname = folder + prefix + suffix + '.csv'
    with open(targetname,'w',newline='') as targetfile:
        datawriter = csv.writer(targetfile)
        datawriter.writerows(content)
    return 1

def read_trials(filename):
    rate_list = list()
    with open(filename,'r') as datafile:
        datareader = csv.reader(datafile)
        trials = 0
        for row in datareader:
            trials = trials + 1
            rate = list()
            for idx,value in enumerate(row):
                rate.append(float(value))
            rate_list.append(rate)
    return rate_list, trials

def rate_plot(samplesize,opt_filename,unif_filename,image_filename):
    opt_rate_list, opt_trials = read_trials(opt_filename)
    unif_rate_list, unif_trials = read_trials(unif_filename)
    opt_rate_list = np.array(opt_rate_list)
    unif_rate_list = np.array(unif_rate_list)
    opt_mean = np.sum(opt_rate_list,axis=0) / opt_trials
    unif_mean = np.sum(unif_rate_list,axis=0) / unif_trials
    opt_err_rate = -opt_mean + 0.9
    unif_err_rate = -unif_mean + 0.9
    opt_std = np.std(opt_rate_list,axis=0)
    unif_std = np.std(unif_rate_list,axis=0)

    fig = plt.figure()
    plt.errorbar(samplesize,unif_err_rate,unif_std,fmt='g:x',fillstyle='none',label='unif')
    plt.errorbar(samplesize,opt_err_rate,opt_std,fmt='b--s',fillstyle='none',label='opt')
    plt.yticks(np.arange(0,0.2,0.05))
    plt.legend(loc=1)
    plt.xlabel('$\log(m)$')
    plt.ylabel('error rate')
    plt.savefig(image_filename)
    plt.close(fig)

def main():
    ### combine opt_best_score results from different trials
    #trials = range(1,11,1)
    #folder = 'result/'
    #prefix = 'opt_best_score trial '
    #suffix = ' '
    #trials_agg(trials,folder,prefix,suffix)

    ### combine unif_best_score results from different trials
    #trials = range(1,11,1)
    #folder = 'result/'
    #prefix = 'unif_best_score trial '
    #suffix = ' '
    #trials_agg(trials,folder,prefix,suffix)

    ### plot the learning rate
    #samplesize = np.arange(2,6,0.5)
    #opt_filename = 'result/opt_best_score trial  .csv'
    #unif_filename = 'result/unif_best_score trial  .csv'
    #image_filename = 'image/learningrate.eps'
    #rate_plot(samplesize,opt_filename,unif_filename,image_filename)

    ### plot sample points
    X = np.loadtxt('data/ideal_Xtest.txt')
    Y = np.loadtxt('data/ideal_Ytest.txt')
    ratio = 50 / len(X)
    rff.plot_circle(X,Y,ratio)

if __name__ == '__main__':
    main()
