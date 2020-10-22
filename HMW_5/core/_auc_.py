import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score

# Function importing Dataset
def importdata(trainFile, testFile):
    train = pd.read_csv(trainFile, sep=',', header=None)
    test = pd.read_csv(testFile, sep=',', header=None)
    return train, test

# Function to split target from data
def splitdataset(train, test,scaler='no'):
    ohe = OneHotEncoder(sparse=True)
    objInTrain = len(train)
    allData = pd.concat([train, test], ignore_index=True, sort=False, axis=0)
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
    X_train = codAllDataAgain[:objInTrain]
    y_train = train.values[:, -1]

    X_test = codAllDataAgain[objInTrain:]
    y_test = test.values[:, -1]

    if scaler == 'no':
        return X_train, X_test, y_train, y_test
    elif scaler == 'minmax':
        mm_scaler = MinMaxScaler()
        X_train_minmax = pd.DataFrame(mm_scaler.fit_transform(X_train[X_train.columns]), index=X_train.index,
                                      columns=X_train.columns)
        X_test_minmax = pd.DataFrame(mm_scaler.transform(X_test[X_test.columns]), index=X_test.index,
                                     columns=X_test.columns)
        return X_train_minmax, X_test_minmax,y_train, y_test

    elif scaler == 'std':
        std_scaler = StandardScaler()
        X_train_std = pd.DataFrame(std_scaler.fit_transform(X_train[X_train.columns]), index=X_train.index,
                                   columns=X_train.columns)
        X_test_std = pd.DataFrame(std_scaler.transform(X_test[X_test.columns]), index=X_test.index, columns=X_test.columns)

        return X_train_std, X_test_std,y_train, y_test
    elif scaler == 'both':
        mm_scaler = MinMaxScaler()
        X_train_minmax = pd.DataFrame(mm_scaler.fit_transform(X_train[X_train.columns]), index=X_train.index,
                                      columns=X_train.columns)
        X_test_minmax = pd.DataFrame(mm_scaler.transform(X_test[X_test.columns]), index=X_test.index,
                                     columns=X_test.columns)
        std_scaler = StandardScaler()
        X_train_minmax_std = pd.DataFrame(std_scaler.fit_transform(X_train_minmax[X_train_minmax.columns]),
                                          index=X_train_minmax.index, columns=X_train_minmax.columns)
        X_test_minmax_std = pd.DataFrame(std_scaler.transform(X_test_minmax[X_test_minmax.columns]),
                                         index=X_test_minmax.index, columns=X_test_minmax.columns)

        return X_train_minmax_std, X_test_minmax_std,y_train, y_test

# Function to make predictions
def prediction(X_test, model, model_name, other_models):
    if model_name in other_models:
        y_pred = model.decision_function(X_test)
    else:
        y_pred = model.score_samples(X_test)

    return y_pred

def getAUC(model,model_name,trainFile,testFile,scaler,other_models):
    train, test = importdata(trainFile, testFile)
    X_train, X_test, y_train, y_test= splitdataset(train, test, scaler)
    if model_name in other_models:
        model.fit(X_train)
    else:
        model.fit(X_train, y_train)
    y_pred = prediction(X_test,model, model_name,other_models)
    auc = roc_auc_score(y_test, y_pred)
    return auc

def getOrderFolders():
    folders = pd.read_csv("core/order.csv")
    folders = list(folders.folder)
    return folders

def saveResults(results,start_counts,counts,name="results.csv"):
    head = ['model']
    folders = getOrderFolders()
    if counts != -1:
        folders = folders[start_counts:start_counts+counts]
    folders.reverse()
    head += folders

    head += ["Average"]
    dataOutput = pd.DataFrame(results, columns=head)
    dataOutput.to_csv(name, index=False)





