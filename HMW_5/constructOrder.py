from pathlib import Path
import pandas as pd
from tqdm import tqdm
def getFolders(path):
    objects = [obj.name for obj in Path(path).iterdir()]
    return objects

def getFiles(path):
    objects = [obj.name for obj in Path(path).iterdir()]
    return objects

names = ['train', 'test']
root = "Unsupervised_Anamaly_Detection_csv"
arr_auc = []

folders = getFolders(root)

total_folders = len(folders)
bar = tqdm(total = total_folders,position=0,leave=False)

totals = []
for iteration, folder in enumerate(folders):
    pair = []
    #print(f"--------------{iteration + 1}/{total_folders} Name: {folder}----------")
    csv_files = {}

    for name, file in enumerate(getFiles(f"{root}/{folder}")):
        frame = pd.read_csv(f"{root}/{folder}/{file}", header=None)
        csv_files[names[name]] = frame

    merged = pd.concat([csv_files['train'], csv_files['test']], ignore_index=True, sort=False, axis=0)
    #print(folder,len(merged))
    totals.append([len(merged),folder])
    #totals.append(pair)
    bar.update(1)
bar.update(0)
bar.close()

totals.sort()

for i,folder in enumerate(totals):
    print(i,folder)

order = pd.DataFrame(totals,columns=["len","folder"])
order.to_csv("order.csv",index=False)