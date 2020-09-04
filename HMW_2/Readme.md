# Validity-Index


Implementation of set of classifiers in Python and Weka

- Author: Gabriel Ichcanziho Pérez Landa
- Date of creation: September 01st, 2020
- Email: ichcanziho@outlook.com




### Installation

This repostory requires [Pip](https://docs.python.org/3/installing/index.html) to install the requirements.

To run the program you must have some libraries, you can install it using the next command:

```sh
$ pip install -r requirements.txt
```

## Editable parameters

You can modify the following parameters inside main.py to suit your needs

```sh
# how many partitions do you want to create
numberOfPartitions = 10
# how categorical classes do you want to create
numberOfClasses = 2
# name of the history output's name
outputName = 'History'
# name of the new data with categorical values (it keeps the best result for you)
dataName = 'BestPartition'
# what is the minimum number for each partition
min_cut = 500
# take care about this number when you modify the number of classes, this number must be less than the len of your data
max_cut = 4000
# the number of folds to use on cross-validation
numberOfKfolds = 10
# the number of cpus to run in parallel the cross-validation
numberOfCPUS = 5
```
## Implemented classifiers

- RandomForestClassifier()
- SVC()
- GaussianNB()
- LinearDiscriminantAnalysis()
- MLPClassifier()
- DecisionTreeClassifier()
- LogisticRegression()
- LogisticRegression()
- AdaBoostClassifier()

## Add or remove a classifier

To add a new classifier you must import their library and put it on main.py file, for example, let's import GaussianProcessClassifier

### Before
```sh
from vic_parallel import runVic,makePartitions
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import time
```

### After
```sh
from vic_parallel import runVic,makePartitions
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import time
# add a new classifier
from sklearn.gaussian_process import GaussianProcessClassifier
```

Then we just modify the list of current classifiers:

### Before
```sh
classifiers = [RandomForestClassifier(max_features='sqrt', criterion='entropy', n_jobs=4),
               SVC(gamma='auto'), GaussianNB(), LinearDiscriminantAnalysis(solver='eigen',shrinkage = 'auto'),
               MLPClassifier(activation = 'logistic',learning_rate = 'invscaling'),
               DecisionTreeClassifier(max_depth=60),LogisticRegression(),AdaBoostClassifier()]
```
### After
```sh
classifiers = [GaussianProcessClassifier(),RandomForestClassifier(max_features='sqrt', criterion='entropy', n_jobs=4),
               SVC(gamma='auto'), GaussianNB(), LinearDiscriminantAnalysis(solver='eigen',shrinkage = 'auto'),
               MLPClassifier(activation = 'logistic',learning_rate = 'invscaling'),
               DecisionTreeClassifier(max_depth=60),LogisticRegression(),AdaBoostClassifier()]
```

To remove a classifier you only need to erase it from the classifiers list.

## Run the program

To generate the best partition of your dataset you need to run the next command:

```sh
$ python main.py
```

Additionaly you can make different graphs using:

```sh
$ python makeGraphs.py
```

## References

J. Rodríguez, M. A. Medina-Pérez, A. E. Gutierrez-Rodríguez, R. Monroy, H. Terashima-Marín, "Cluster validation using an ensemble of supervised classifiers," Knowledge-Based Systems, vol. 145, pp. 134-144, 2018. (Q1, IF: 5.101) 


**Free Software, Hell Yeah!**
