import csv
import numpy as np
import datagen
from sklearn.model_selection import train_test_split
### set up data parameters
def main():
    data_para = {'dim':2,'gap':0.2,'label_prob':0.9,'samplesize':10**6,'testsize':0.2}
    dim = data_para['dim']
    gap = data_para['gap']
    label_prob = data_para['label_prob']
    samplesize = data_para['samplesize']
    test_size = data_para['testsize']

    ### generate and write train and test dataset

    X,Y = datagen.unit_ball_ideal(dim,gap,label_prob,samplesize)
    X_train_p,X_test,Y_train_p,Y_test = train_test_split(X,Y,
                                                         test_size = test_size,
                                                         random_state=0)
    np.savetxt('data/ideal_Xtrain_'+str(dim)+'.txt',X_train_p)
    np.savetxt('data/ideal_Xtest_'+str(dim)+'.txt',X_test)
    np.savetxt('data/ideal_Ytrain_'+str(dim)+'.txt',Y_train_p)
    np.savetxt('data/ideal_Ytest_'+str(dim)+'.txt',Y_test)
    with open('data/ideal_parameter_'+str(dim)+'.csv','w',newline='') as csvfile:
        fieldnames = list(data_para.keys())
        datawriter = csv.DictWriter(csvfile,fieldnames=fieldnames)
        datawriter.writeheader()
        datawriter.writerow(data_para)

if __name__ == '__main__':
    main()
