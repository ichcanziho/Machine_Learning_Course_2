from vic_parallel import runVic,makePartitions
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import time
# how many partitions do you want to create
numberOfPartitions = 3
# how categorical classes do you want to create
numberOfClasses = 2
# name of the history output's name
outputName = 'Test'
# name of the new data with categorical values (it keeps the best result for you)
dataName = 'BestPartitionTest'
# what is the minimum number for each partition
min_cut = 500
# take care about this number when you modify the number of classes, this number must be less than the len of your data
max_cut = 4000
# the number of folds to use un cross-validation
numberOfKfolds = 4
# the number of cpus to run in parallel the cross-validation
numberOfCPUS = 5
# you can add a prefix in the number of partition, of you don't want to use a initial cut you can remove the element
# use prefix #2349
#prefix = [[2636]]
# not use prefix
prefix = []

# to add a new classifier, append to the next list, each classifier can have their own parameters
# the following classifiers belongs to sklearn
classifiers = [RandomForestClassifier(max_features='sqrt', criterion='entropy', n_jobs=4),
               SVC(gamma='auto'), GaussianNB(), LinearDiscriminantAnalysis(), MLPClassifier()]

if __name__ == '__main__':
    # load the dataset and puts the name of the output class
    partition = makePartitions('Data_15s_30r.csv', 'score_change')
    # remove unnecessary columns
    partition.cleanData(['fingerprint', 'minutia'])
    startTime = time.time()
    runVic(partition, numberOfPartitions, numberOfClasses,
           min_cut, max_cut,classifiers,numberOfKfolds,numberOfCPUS,
           outputName,dataName, prefix=prefix)

    endTime = time.time()
    elpaseTime = endTime - startTime
    print("--------------------")
    print("Finish, total time:",elpaseTime)
    print("--------------------")
