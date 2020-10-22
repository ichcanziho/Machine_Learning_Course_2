from core._auc_ import getAUC,getOrderFolders,saveResults

from pyod.models.so_gaal import SO_GAAL

'''
from pyod.models.mcd import MCD
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from brminer import BRM
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.vae import VAE
from pyod.models.sos import SOS
from pyod.models.pca import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
'''
import time
from joblib import Parallel, delayed

def getResults(root,scaler,model,model_name,start_count=0,counts=-1,other_models=('SOS')):
    errors =0
    arr_auc = []
    folders = getOrderFolders()
    if counts != -1:
        folders = folders[start_count:start_count+counts]
    total_folders = len(folders)
    folders.reverse()
    for iteration,folder in enumerate(folders):
        print("starting", folder)
        start = time.time()
        new_path = f"{root}/{folder}/merged.csv"
        try:
            auc = getAUC(model,model_name,new_path,new_path,scaler,other_models)
            arr_auc.append(1 - auc if auc < 0.5 else auc)
            print(
                f"{round(time.time() - start, 3)} Sec------{model_name}--------{iteration + 1}/{total_folders} Name: {folder}----------")
            print(arr_auc)
        except errors:
            print(errors)
            print("i am",model_name,"i have a problem with",folder)
            errors+=1


    aver_auc = sum(arr_auc) / len(arr_auc)
    print('FINAL RESULTS:' + model_name + 'Average:' + str(aver_auc) + '\n History!! ' + str(arr_auc),"\n errors:",errors)
    return [model_name] + arr_auc + [aver_auc]

scaler = 'minmax'

start=0
counts=90
CPUS = 3

root = 'Unsupervised_Anamaly_Detection_csv'

other_models = ['ABOD', 'KNN','VAE','SOS','PCA','CBLOF','COF', 'LSCP','SO_GAAL']
name = "SO_GAAL"
models = {name:SO_GAAL()}
results = Parallel(n_jobs=CPUS)\
    (delayed(getResults)
     (root, scaler, model,model_name,start,counts,other_models) for model_name,model in models.items())

saveResults(results,start,counts,name=f'results/{name}_{scaler}_scaler_{start}_{start+counts}.csv')