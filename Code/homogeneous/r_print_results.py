from numpy.lib.function_base import average
import pandas as pd
import os
import numpy as np
from sklearn.metrics import  accuracy_score, f1_score
from IPython.display import display
from config import *

workDir = r'../results/'
main_classifier = ['Bolivia', 'Ecuador', 'Colombia', 'Chile']
countries = ['Ecuador', 'Bolivia', 'Colombia', 'Chile']
datasets = ['train', 'val', 'test']

for main_c in main_classifier:
    country_dir = r'{}/{}/{}/'.format(workDir, main_c, DATA_OUT)
    result_dict = {'accuracy': {}, 'f1': {}}
    for country in countries:
        if main_c == country:    continue
        result_dict[country] = {}
        result_dict[country] = {}
        true_lbl, pred_lbl = [], []
        for dt in datasets:        
            dataDF = pd.read_csv(os.path.join(country_dir, '{}_{}_predictions.csv'.format(country, dt)), dtype = {'user_id': str})
            true_lbl.append(dataDF.true_label.values)
            pred_lbl.append(dataDF.predicted_label.values)
       
        true_lbl = np.concatenate(true_lbl)
        pred_lbl = np.concatenate(pred_lbl)
        result_dict[country]['accuracy'] = accuracy_score(true_lbl, pred_lbl)
        result_dict[country]['f1'] = f1_score(true_lbl, pred_lbl, average = 'macro')
        maj = true_lbl.mean()
        result_dict[country]['majority'] = maj if maj > 0.5 else 1 - maj
       
    resultsDF = pd.DataFrame.from_dict(result_dict, orient = 'index')
    print(f' ############ {main_c} Classifier ############')
    display(resultsDF)



