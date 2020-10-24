import pandas as pd
#results = pd.read_csv("90 columns/results_no_scaler_0_90.csv")
def csvFormater(results):
    model_names = list(results.model)
    results = results.drop(columns=['model','Average'])
    dataset_names = list(results.columns)
    total_datasets = len(dataset_names)
    output = pd.DataFrame({})
    classifier_names = []
    datasets_names=[]
    accuracys=[]
    for i in range(len(model_names)):
        names = [model_names[i]]*total_datasets
        classifier_names += names
        datasets_names += dataset_names
        frame = list(results.iloc[i])
        accuracys += frame
    output['classifier_name'] = classifier_names
    output['dataset_name']=datasets_names
    output['accuracy'] = accuracys
    #print(output)
    return output
    #output.to_csv("models.csv",index=False)