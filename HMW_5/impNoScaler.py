from pathlib import Path
import time
from tqdm import tqdm
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#Classifiers
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import brminer
#parallel process
from joblib import Parallel, delayed

def getFolders(path):
    objects = [obj.name for obj in Path(path).iterdir()]
    return objects

def getOrderFolders():
    folders = pd.read_csv("order.csv")
    folders = list(folders.folder)
    return folders

def getFiles(path):
    objects = [obj.name for obj in Path(path).iterdir()]
    return objects

def useScaler(allData,scaler="no"):
    ohe = OneHotEncoder(sparse=True)
    AllDataWihoutClass = allData.iloc[:, :-1]
    AllDataWihoutClassOnlyNominals = AllDataWihoutClass.select_dtypes(include=['object'])
    AllDataWihoutClassNoNominals = AllDataWihoutClass.select_dtypes(exclude=['object'])

    encAllDataWihoutClassNominals = ohe.fit_transform(AllDataWihoutClassOnlyNominals)
    encAllDataWihoutClassNominalsToPanda = pd.DataFrame(encAllDataWihoutClassNominals.toarray())

    if AllDataWihoutClassOnlyNominals.shape[1] > 0:
        codAllDataAgain = pd.concat([encAllDataWihoutClassNominalsToPanda, AllDataWihoutClassNoNominals],
                                    ignore_index=True, sort=False, axis=1)
    else:
        codAllDataAgain = AllDataWihoutClass
        # Seperating the target variable
    X_train = codAllDataAgain  # [:objInTrain]

    X_test = codAllDataAgain  # [objInTrain:]
    y_test = allData.values[0:, -1]
    y_test = list(y_test)
    y_test = [element if element != 'Class' else 'negative' for element in y_test ]

    #from sklearn.preprocessing import LabelEncoder
    #label_encoder = LabelEncoder()
    #y = label_encoder.fit_transform(y_test)

    y_train = y_test

    if scaler == "minmax":
        mm_scaler = MinMaxScaler()
        X_train_minmax = pd.DataFrame(mm_scaler.fit_transform(X_train[X_train.columns]), index=X_train.index,
                                      columns=X_train.columns)
        X_test_minmax = pd.DataFrame(mm_scaler.transform(X_test[X_test.columns]), index=X_test.index,
                                     columns=X_test.columns)
        return X_train_minmax,X_test_minmax,y_train,y_test
    elif scaler == "std":
        std_scaler = StandardScaler()
        X_train_std = pd.DataFrame(std_scaler.fit_transform(X_train[X_train.columns]), index=X_train.index,
                                   columns=X_train.columns)
        X_test_std = pd.DataFrame(std_scaler.transform(X_test[X_test.columns]), index=X_test.index,
                                  columns=X_test.columns)
        return X_train_std,X_test_std,y_train,y_test
    elif scaler == "minmax_std":
        mm_scaler = MinMaxScaler()
        std_scaler = StandardScaler()
        X_train_minmax = pd.DataFrame(mm_scaler.fit_transform(X_train[X_train.columns]), index=X_train.index,
                                      columns=X_train.columns)
        X_test_minmax = pd.DataFrame(mm_scaler.transform(X_test[X_test.columns]), index=X_test.index,
                                     columns=X_test.columns)
        X_train_minmax_std = pd.DataFrame(std_scaler.fit_transform(X_train_minmax[X_train_minmax.columns]),
                                          index=X_train_minmax.index, columns=X_train_minmax.columns)
        X_test_minmax_std = pd.DataFrame(std_scaler.transform(X_test_minmax[X_test_minmax.columns]),
                                         index=X_test_minmax.index, columns=X_test_minmax.columns)
        return X_train_minmax_std,X_test_minmax_std,y_train,y_test
    else:
        return X_train,X_test,y_train,y_test

# Function to make predictions
def prediction(X_test, clf_object,typeModel):
    #if typeModel != 'BRMiner':
    #    y_pred = clf_object.predict(X_test)
    #else:
    y_pred = clf_object.score_samples(X_test)
    return y_pred

def getResults(root,scaler,clf_classif,typeModel,start_count=0,counts=-1):
    names = ['train', 'test']
    arr_auc = []
    folders = getOrderFolders()

    if counts != -1:
        folders = folders[start_count:start_count+counts]
    total_folders = len(folders)
    for iteration, folder in enumerate(folders):
        print("starting",folder)
        start = time.time()

        csv_files = {}

        for name, file in enumerate(getFiles(f"{root}/{folder}")):
            frame = pd.read_csv(f"{root}/{folder}/{file}", header=None)
            csv_files[names[name]] = frame

        merged = pd.concat([csv_files['train'], csv_files['test']], ignore_index=True, sort=False, axis=0)

        csv_files['merged'] = merged
        # split datasets
        X_train, X_test, y_train, y_test = useScaler(csv_files['merged'], scaler)

        # Performing training
        clf_classif.fit(X_train, y_train)

        # Operational Phase
        y_pred_classif = prediction(X_test, clf_classif,typeModel)

        auc = metrics.roc_auc_score(y_test, y_pred_classif)
        arr_auc.append(1 - auc if auc < 0.5 else auc)
        print(f"{round(time.time()-start,3)} Sec------{typeModel}--------{iteration + 1}/{total_folders} Name: {folder}----------")
        print(arr_auc)

    aver_auc = sum(arr_auc) / len(arr_auc)
    print('FINAL RESULTS:' + typeModel +'Average:'+str(aver_auc) +'\n History!! ' + str(arr_auc))
    return [typeModel]+arr_auc+[aver_auc]

def saveResults(results,start_counts,counts,name="results.csv"):
    head = ['model']
    folders = getOrderFolders()
    if counts != -1:
        head += folders[start_counts:start_counts+counts]
    else:
        head+=folders
    head += ["Average"]
    dataOutput = pd.DataFrame(results, columns=head)
    dataOutput.to_csv(name, index=False)

root = "Unsupervised_Anamaly_Detection_csv"
scaler="no" # others [minmax,std,minmax_std]
CPUS = 4 # the number of CPUS to use in parallel
start=0 # initial dataset
counts=95 # number of dataset to evaluate

models = [GaussianMixture(),IsolationForest(),
          OneClassSVM()]#,brminer.BRM()]#

# the name that will be saved in the result csv file
typeModels = ['GaussianMixture','IsolationForest',
              'OneClassSVM']#,'BRMiner']#

results = Parallel(n_jobs=CPUS)(delayed(getResults)(root, scaler, model,typeModel,start,counts) for model,typeModel in zip(models,typeModels))
saveResults(results,start,counts,name=f'results/95/results_{scaler}_scaler_{start}_{start+counts}.csv')