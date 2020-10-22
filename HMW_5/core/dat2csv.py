import os
from pathlib import Path

def csvConverter(root,folder,path):
    head = ''
    start=0
    cut=0
    newLines = []
    with open(root+"/"+folder+"/"+path, 'r') as input_file:
        lines = input_file.readlines()
        for line in lines:
            line = line.rstrip('\n')
            newLines.append(line.replace(" ",""))
            if "@inputs" in line:
                words = line.replace("@inputs ","")
                words = words.replace(" ", "")
                words+=",Class"
                head = words
            start+=1
            if "@data" in line:
                cut=start
    newLines = newLines[cut:]
    name = path.replace(".dat","")
    if not (os.path.isdir('../Unsupervised_Anamaly_Detection_csv')):
        os.mkdir('../Unsupervised_Anamaly_Detection_csv')
    if not(os.path.isdir('../Unsupervised_Anamaly_Detection_csv/'+folder)):
        os.mkdir('../Unsupervised_Anamaly_Detection_csv/'+folder)

    file = open('../Unsupervised_Anamaly_Detection_csv/'+folder+"/"+name+'.csv','w')
    file.write(head+"\n")
    for text in newLines:
        file.write(text+"\n")
    file.close()

def getDocuments(path):
    objects = [obj.name for obj in Path(path).iterdir()]
    return objects

root = "../Unsupervised_Anomaly_Detection"

folders = getDocuments(root)

for folder in folders:
    print(folder)
    files = getDocuments(root+"/"+folder)
    for file in files:
        print(root,folder,file)
        csvConverter(root,folder, file)



