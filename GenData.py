import csv
import numpy as np
import rff
### set up data parameters
def main():
    data_para = {'gap':0.2,'label_prob':0.9,'samplesize':10**6,'testsize':0.2}
    gap = data_para['gap']
    label_prob = data_para['label_prob']
    samplesize = data_para['samplesize']
    test_size = data_para['testsize']

    ### generate and write train and test dataset

    X,Y = rff.unit_circle_ideal(gap,label_prob,samplesize)
    X_train_p,X_test,Y_train_p,Y_test = train_test_split(X,Y,
                                                         test_size = test_size,
                                                         random_state=0)
    np.savetxt('data/ideal_Xtrain.txt',X_train_p)
    np.savetxt('data/ideal_Xtest.txt',X_test)
    np.savetxt('data/ideal_Ytrain.txt',Y_train_p)
    np.savetxt('data/ideal_Ytest.txt',Y_test)
    with open('data/ideal_parameter.csv','w',newline='') as csvfile:
        fieldnames = list(data_para.keys())
        datawriter = csv.DictWriter(csvfile,fieldnames=fieldnames)
        datawriter.writeheader()
        datawriter.writerow(data_para)

if __name__ == '__main__':
    main()
