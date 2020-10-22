import pandas as pd
from pathlib import Path

def getFiles(path):
    objects = [obj.name for obj in Path(path).iterdir()]
    return objects

def mergeFilesFromFolder(path):
    files = getFiles(path)
    test_train=[]
    for file in files:
        new_path = path+"/"+file
        frame = pd.read_csv(new_path)
        test_train.append(frame)

    base = pd.concat([test_train[0], test_train[1]])
    base.to_csv(path+'/merged.csv',index=False)

def merge(path = '../Unsupervised_Anamaly_Detection_csv'):
    folders = getFiles(path)
    for folder in folders:
        #if folder == 'new-thyroid1':
        new_path = path+"/"+folder
        mergeFilesFromFolder(new_path)

if __name__ == '__main__':
    merge()







