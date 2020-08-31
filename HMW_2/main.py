from vic_parallel import runVic,makePartitions
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# how many partitions do you want to create
numberOfPartitions = 1
# how categorical classes do you want to create
numberOfClasses = 2
# what is the minimum number for each partition
min_cut = 500
# take care about this number when you modify the number of classes, this number must be less than the len of your data
# for example: if your data has 5000 rows and you want 3 classes, each class could have 4000 elements, and that will produce
# an error, to corret this put max_cut to 2000 for example, in the worst case the classes would have 2000 + 2000 + 1000, the last
# class always is te complement of the others. When the number of classes is 2, you can choose whatever you want.
max_cut = 4000
# the number of folds to use un cross-validation
numberOfKfolds = 3
# the number of cpus to run in parallel the cross-validation
numberOfCPUS = 5
# you can add a prefix in the number of partition, of you don't want to use a initial cut you can remove the element
# use prefix
prefix = [[2636]]
# not use prefix
# prefix = []


classifiers = [RandomForestClassifier(max_features='sqrt', criterion='entropy', n_jobs=2),
               SVC(gamma='auto'), GaussianNB(), LinearDiscriminantAnalysis(), MLPClassifier()]

if __name__ == '__main__':
    partition = makePartitions('Data_15s_30r.csv', 'score_change')
    partition.cleanData(['fingerprint', 'minutia'])
    runVic(partition, numberOfPartitions, numberOfClasses,
           min_cut, max_cut,classifiers,numberOfKfolds,numberOfCPUS, prefix=prefix)
