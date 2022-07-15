import pandas as pd
import os
import os.path
import numpy as np
import pyreadstat
import math
from scipy.stats import fisher_exact
import seaborn as sns 
import matplotlib.pyplot as plt 
import statsmodels.stats.multitest as multi
    
path = os.path.dirname(os.path.realpath(__file__)) + '/'

file = path + 'NAME.csv'
data = pd.read_csv(file)

targets = []
moods = []

for target in targets:
    Mintarget = math.floor(data[target].min())
    Maxtarget = math.ceil(data[target].max())
    for mood in moods:
        Minmood = math.floor(data[mood].min())
        Maxmood = math.ceil(data[mood].max())
        df = pd.DataFrame(0, index=range(Maxtarget, Mintarget-1, -1), columns=range(Minmood, Maxmood+1))
        for i in range(Maxtarget, Mintarget-1, -1):
            for j in range(Minmood, Maxmood+1):
                c1 = len(data[(data[target] >= i) & (data[mood] >= j)])
                c2 = len(data[(data[target] >= i) & (data[mood] < j)])
                c3 = len(data[(data[target] < i) & (data[mood] >= j)])
                c4 = len(data[(data[target] < i) & (data[mood] < j)])

                table = np.array([[c1, c2], [c3, c4]])
                OR, P = fisher_exact(table)
                df.loc[i, j] = -math.log10(P)
                
        fig,ax = plt.subplots(figsize=(12,7))
        plt.xlabel(mood, fontsize = 25)
        plt.ylabel(target, fontsize = 25)
        ax.set_xticks([])
        ax.set_yticks([])
        res = sns.heatmap(df,fmt ='.1%',cmap ='RdYlGn',cbar=True ,linewidths=0.10,ax=ax)
        plt.xlabel(mood, fontsize = 24)
        plt.ylabel(target, fontsize = 24)
        plt.savefig(path +'p_values_' + target + '_' + mood + '.png')