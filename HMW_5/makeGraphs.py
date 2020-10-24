import Orange
from autorank import autorank, plot_stats, create_report, latex_table
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def saveCD(data,name='test',title='title'):
    models = list(data.model)
    data = data.drop(columns=['model', 'Average'])
    datasets = list(data.columns)
    values = data.values
    values = values.T
    data = pd.DataFrame(values, columns=models)
    result = autorank(data, alpha=0.05, verbose=False)
    print(result)
    critical_distance = result.cd
    rankdf = result.rankdf
    avranks = rankdf.meanrank
    ranks = list(avranks.values)
    names = list(avranks.index)
    names = names[:30]
    avranks = ranks[:30]
    Orange.evaluation.graph_ranks(avranks, names, cd=critical_distance, width=10, textspace=1.5, labels=True)
    plt.suptitle(title)
    plt.savefig('results/imgs/eps/'+name+".eps", format="eps")
    plt.savefig('results/imgs/png/' + name + ".png", format="png")
    plt.show()
    plt.close()

def get_box_plot_data(labels, bp):
    rows_list = []
    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        rows_list.append(dict1)
    return pd.DataFrame(rows_list)

def plotBoxes(results,data_name,title,metric):
    base = results.copy()
    models = list(results.model)
    results = results.drop(columns=['model', 'Average'])
    results = results.values
    results = results.T
    bp = plt.boxplot(results)
    info = get_box_plot_data(models, bp)
    plt.close()
    base['median'] = list(info['median'])
    base = base.sort_values(by=['median'])
    models = list(base.model)
    base = base.drop(columns=['model', 'Average','median'])
    base = base.values
    base = base.T
    ax = sns.boxplot(data=base)
    ax.set_ylabel(metric)  # average precision
    ax.set_title(title)
    ax.set_xticklabels(models, rotation=90)
    ax.set_xlabel("Models")
    plt.savefig(f'results/imgs/png/boxplot_{data_name}.png', bbox_inches='tight')
    plt.savefig(f'results/imgs/eps/boxplot_{data_name}.eps', bbox_inches='tight')
    plt.close()



results = pd.read_csv('results/std/30_Models_AVE_STD_SCALER.csv')

plotBoxes(results,'ave std scale',
          'Average precision results with standard scaling',
          'Average precision')
saveCD(results,'cd ave std scale',
       'Cd diagram - AUC with standard scaling')
