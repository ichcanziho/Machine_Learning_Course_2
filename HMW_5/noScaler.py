from core._auc_ import getAUC,getOrderFolders,saveResults
from brminer import BRM
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.cof import COF
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.feature_bagging import FeatureBagging
import time
from joblib import Parallel, delayed

def getResults(root,scaler,model,model_name,start_count=0,counts=-1,other_models=('SOS')):
    errors =0
    arr_auc = []
    arr_ave = []
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
            auc,ave = getAUC(model,model_name,new_path,new_path,scaler,other_models)
            arr_auc.append(1 - auc if auc < 0.5 else auc)
            arr_ave.append(1 - ave if ave < 0.5 else ave)
            print(
                f"{round(time.time() - start, 3)} Sec------{model_name}--------{iteration + 1}/{total_folders} Name: {folder}----------")
            print(arr_auc)
        except errors:
            print(errors)
            print("i am",model_name,"i have a problem with",folder)
            errors+=1

    aver_auc = sum(arr_auc) / len(arr_auc)
    aver_ave = sum(arr_ave) / len(arr_ave)
    print('FINAL RESULTS:' + model_name + 'Average:' + str(aver_auc) + '\n History!! ' + str(arr_auc),"\n errors:",errors)
    auc_results = [model_name] + arr_auc + [aver_auc]
    ave_results = [model_name] + arr_ave + [aver_ave]
    return auc_results,ave_results

scaler = 'no'

start=0
counts=90
CPUS = 4

root = 'Unsupervised_Anamaly_Detection_csv'


other_models = ['AvgKNN','LargestKNN','MedKNN','PCA','COF',
                'LODA','LOF','HBOS','MCD','AvgBagging','MaxBagging']

name = "5_Models"

models = {'BRM':BRM(),
          'GM':GaussianMixture(),
          'IF': IsolationForest(),
          'OCSVM':OneClassSVM(),
          'EE':EllipticEnvelope(),
          'AvgKNN':KNN(method='mean'),
          'LargestKNN':KNN(method='largest'),
          'MedKNN':KNN(method='median'),
          'PCA':PCA(),
          'COF':COF(),
          'LODA':LODA(),
          'LOF':LOF(),
          'HBOS':HBOS(),
          'MCD':MCD(),
          'AvgBagging':FeatureBagging(combination='average'),
          'MaxBagging':FeatureBagging(combination='max')
          }

results = Parallel(n_jobs=CPUS)\
    (delayed(getResults)
     (root, scaler, model,model_name,start,counts,other_models) for model_name,model in models.items())

saveResults(results,start,counts,name=f'results/{name}_{scaler}_scaler_{start}_{start+counts}')
