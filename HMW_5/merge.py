import pandas as pd
from pathlib import Path

def getFiles(path):
    objects = [obj.name for obj in Path(path).iterdir()]
    return objects

def mergeTraTst(path):
    files = getFiles(path)
    if 'merged.csv' in files:
        files.remove('merged.csv')
    a = pd.read_csv(f"{path}/{files[0]}")
    b = pd.read_csv(f"{path}/{files[0]}")
    base = pd.concat([a, b],ignore_index=True, sort=False, axis=0)
    base.to_csv(path + '/merged.csv', index=False,header=False)

def merge(path = 'Unsupervised_Anamaly_Detection_csv'):
    folders = getFiles(path)
    for folder in folders:
        mergeTraTst(path+"/"+folder)

if __name__ == '__main__':
    merge()