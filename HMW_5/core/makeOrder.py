from pathlib import Path
import pandas as pd
from tqdm import tqdm

def getFiles(path):
    objects = [obj.name for obj in Path(path).iterdir()]
    return objects
names = ['train', 'test']
root = "../Unsupervised_Anamaly_Detection_csv"
arr_auc = []
folders = getFiles(root)
total_folders = len(folders)
bar = tqdm(total = total_folders,position=0,leave=False)
totals = []
for iteration, folder in enumerate(folders):
    merged = pd.read_csv(f'{root}/{folder}/merged.csv')
    totals.append([len(merged),folder])
    bar.update(1)
bar.update(0)
bar.close()
totals.sort()
for i,folder in enumerate(totals):
    print(i,folder)
order = pd.DataFrame(totals,columns=["len","folder"])
order.to_csv("order.csv",index=False)