import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('dataOutput/history50c3.csv')
numberOfpartitions = 10
numberOfClusters = 3
outputName_1 = 'bestRoc50c3'
outputName_2 = 'bestDistribution50c3'


def graphAUCiteration():
    labels = list(range(1,len(data)+1))
    aucs = list(data['AUC'])
    aucs = [round(auc,3) for auc in aucs]
    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, aucs, width,color="b")
    bestPos = aucs.index(max(aucs))
    worstPos = aucs.index(min(aucs))
    rects1[bestPos].set_color('g')
    #rects1[bestPos].set_label("Best")
    rects1[worstPos].set_color('r')
    #rects1[worstPos].set_label("Worst")

    ax.set_ylabel('AUC score')
    ax.set_xlabel("Iterations")
    ax.set_title('Best area under the curve for each iteration')
    ax.set_xticks(x)
    plt.xticks(fontsize=8, rotation=90)
    ax.set_xticklabels(labels)
    #ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel([rects1[bestPos],rects1[worstPos]])
    fig.tight_layout()
    plt.savefig('dataOutput/images/{}.png'.format(outputName_1))
    plt.savefig('dataOutput/images/{}.pdf'.format(outputName_1))

    plt.show()

def graph2columns():
    data.sort_values(by=['AUC'],inplace=True, ascending=False)
    bestAucs = list(data['AUC'])[:numberOfpartitions]
    dist = list(data['Distribution'])[:numberOfpartitions]
    labels = [round(auc,3) for auc in bestAucs]
    matrix = []
    for row in dist:
        row = row.replace('[',"")
        row = row.replace(']',"")
        row = row.replace(' ',"")
        d = row.split(",")
        d = map(int, d)
        d = list(d)
        matrix.append(d)
    for i in range(len(matrix)):
        dif = matrix[i][1]-matrix[i][0]
        matrix[i][1] = dif
    new=[]
    for i in range(len(matrix[0])):
        row = []
        for element in matrix:
            row.append(element[i])
        new.append(row)
    x = np.arange(len(labels))  # the label locations
    width = 0.20  # the width of the bars
    fig, ax = plt.subplots()
    if len(matrix[0]) == 2:
        rects1 = ax.bar(x - width/2, new[0], width, label='Cluster 1')
        rects1 = ax.bar(x + 2*width/2, new[1], width, label='Cluster 2')

    ax.set_ylabel('Number of elements per cluster')
    ax.set_title('Distribution of the best results')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('AUC score')
    ax.legend()
    fig.tight_layout()
    plt.savefig('dataOutput/images/{}.png'.format(outputName_2))
    plt.savefig('dataOutput/images/{}.pdf'.format(outputName_2))
    plt.show()

def graph3columns():
    data.sort_values(by=['AUC'],inplace=True, ascending=False)
    bestAucs = list(data['AUC'])[:numberOfpartitions]
    dist = list(data['Distribution'])[:numberOfpartitions]
    labels = [round(auc,3) for auc in bestAucs]
    matrix = []
    for row in dist:
        row = row.replace('[',"")
        row = row.replace(']',"")
        row = row.replace(' ',"")
        d = row.split(",")
        d = map(int, d)
        d = list(d)
        matrix.append(d)

    for i in range(len(matrix)):
        dif = matrix[i][2]-matrix[i][1]
        matrix[i][2] = dif
        dif = matrix[i][1]-matrix[i][0]
        matrix[i][1] = dif

    new=[]
    for i in range(len(matrix[0])):
        row = []
        for element in matrix:
            row.append(element[i])
        new.append(row)
    x = np.arange(len(labels))  # the label locations
    width = 0.20  # the width of the bars
    fig, ax = plt.subplots()

    ax.bar(x - width/2, new[0], width, label='Cluster 1')
    ax.bar(x + 2*width/2, new[1], width, label='Cluster 2')
    ax.bar(x + 5 * width / 2, new[2], width, label='Cluster 3')

    ax.set_ylabel('Number of elements per cluster')
    ax.set_title('Distribution of the best results')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('AUC score')
    ax.legend()
    fig.tight_layout()
    plt.savefig('dataOutput/images/{}.png'.format(outputName_2))
    plt.savefig('dataOutput/images/{}.pdf'.format(outputName_2))
    plt.show()

graphAUCiteration()

if numberOfClusters == 2:
    graph2columns()
else:
    graph3columns()