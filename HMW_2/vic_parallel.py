from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
import pandas as pd
from sklearn.svm import SVC
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import random


class VIC:

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def roc_auc_score_multiclass(self, y_true, y_pred, average="macro"):
        classes = set(y_true)
        roc_auc_dict = {}
        for current_class in classes:
            other_class = [x for x in classes if x != current_class]
            new_actual_class = [0 if x in other_class else 1 for x in y_true]
            new_pred_class = [0 if x in other_class else 1 for x in y_pred]
            roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
            roc_auc_dict[current_class] = roc_auc
        avg = sum(roc_auc_dict.values()) / len(roc_auc_dict)
        return avg

    def train(self, train_index, test_index, model, X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model.fit(X_train, y_train)
        Y_predict = model.predict(X_test)
        aucV = self.roc_auc_score_multiclass(y_test, Y_predict)
        return aucV

    def vic(self, data, kFolds=5, CPUS=5):

        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1]
        results = []
        for model in self.classifiers:
            skf = StratifiedKFold(n_splits=kFolds, random_state=30, shuffle=True)
            start = time.time()
            name = model.__class__.__name__
            out = Parallel(n_jobs=CPUS)(  # pre_dispatch='1.5*n_jobs'
                delayed(self.train)(train_index, test_index, model, X, Y) for train_index, test_index in
                skf.split(X, Y))
            end = time.time()
            elapse = end - start
            aucAverage = sum(out) / len(out)
            results.append([aucAverage, name])
            print("model {}, auc average {} time running {}".format(name, aucAverage, elapse))
        results.sort(reverse=True)
        best = results[0]
        print("Current result:", best)
        return best


class makePartitions():

    def __init__(self, data, y):
        self.data = pd.read_csv('data/' + data)
        self.data = self.data.fillna(self.data.mean())
        self.realOutput = self.data[y].tolist()
        self.y_label = y

    def cleanData(self, cols):
        self.data.drop(cols, axis=1, inplace=True)

    def mapOutputs(self, cuts):

        y = self.realOutput
        classes = list(range(0, len(cuts) + 1))
        classes = classes[::-1]
        cuts.append(len(y))
        y_sorted = y.copy()
        y_sorted.sort()
        cuts = cuts[::-1]
        pivots = [y_sorted[n - 1] for n in cuts]
        outpus = [5] * len(y)
        for pivot, label in zip(pivots, classes):
            i = 0
            for value in y:
                if value <= pivot:
                    outpus[i] = label
                i += 1

        self.updateData(outpus)
        return self.data

    def updateData(self, new_y):
        self.data.drop([self.y_label], axis=1, inplace=True)
        self.data[self.y_label] = new_y

    def makeCuts(self, n_cuts, n_classes, min_cut=500, max_cut=4000):
        n_classes = n_classes - 1
        cuts = []
        i = 0
        while True:
            cut = []
            for n in range(n_classes):
                r = random.randint(min_cut, max_cut)
                cut.append(r)
            cut.sort()
            if cut not in cuts:
                cuts.append(cut)
                i += 1
            if i == n_cuts:
                break
        return cuts


def main(partition, numberOfPartitions, numberOfClasses, min_cut, max_cut,prefix=[]):
    cuts = prefix
    cuts += partition.makeCuts(n_cuts=numberOfPartitions, n_classes=numberOfClasses, min_cut=min_cut, max_cut=max_cut)

    print(cuts)
    iteration = 1
    # 2103 [2081], [2780] 2636 CHIDO
    best = [0, "free"]
    history = []
    for cut in cuts:
        print('####################################################')
        print("####### Partition {} iteration {}/{}  ############".format(cut, iteration, len(cuts)))
        print('####################################################')
        newData = partition.mapOutputs(cut)
        model = VIC(classifiers)
        current = model.vic(newData, kFolds=numberOfKfolds, CPUS=numberOfCPUS)
        if current[0] > best[0]:
            best = current
            newData.to_csv('dataOutput/bestPartition.csv', index=False)
        print("------------------")
        print("Best overall:", best)
        print("------------------")
        out = current+[cut]
        history.append(out)
        iteration += 1

    history_df = pd.DataFrame(history, columns=['AUC', 'Classifier','Distribution'])
    history_df.to_csv('dataOutput/history.csv')
    print(history_df)


numberOfPartitions = 1
numberOfClasses = 2
min_cut = 500
max_cut = 4000
numberOfKfolds = 3
numberOfCPUS = 5
prefix = [[2636]]
classifiers = [RandomForestClassifier(max_features='sqrt', criterion='entropy', n_jobs=2),
               SVC(gamma='auto'), GaussianNB(), LinearDiscriminantAnalysis(), MLPClassifier()]

if __name__ == '__main__':
    partition = makePartitions('Data_15s_30r.csv', 'score_change')
    partition.cleanData(['fingerprint', 'minutia'])
    main(partition, numberOfPartitions, numberOfClasses, min_cut, max_cut,prefix=prefix)
