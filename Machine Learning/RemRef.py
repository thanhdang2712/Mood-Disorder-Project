import pandas
import os
import os.path
import numpy as np
import pyreadstat


file = os.path.dirname(os.path.realpath(__file__)) + '/SPAQ.sav'

data = pandas.read_spss(file)

genes1way = ['PER2', 'PER3A', 'PER3B', 'PER3C', 'CLOCK', 'CRY1', 'CRY2', 'VNTR']
allGenes = ['PER2AA', 'PER2GG', 'PER2AG', 'PER3ACC', 'PER3AGG', 'PER3ACG',
       'PER3BGG', 'PER3BAA', 'PER3BAG', 'PER3CGG', 'PER3CTT', 'PER3CGT',
       'CLOCKTT', 'CLOCKCC', 'CLOCKTC', 'CRY1CC', 'CRY1GG', 'CRY1CG', 'CRY2GG',
       'CRY2AA', 'CRY2AG', 'VNTRAA', 'VNTRBB', 'VNTRAB']

ref=[]
for gene in genes1way:
    variations = pandas.Series([i for i in allGenes if i.startswith(gene)])
    high = data[variations].value_counts().idxmax()
    indx = high.index(1)
    ref.append(variations[indx])

genelen = len(ref)

for i in range(genelen-1):
    for j in range(i+1, genelen):
        gene1 = ref[i]
        gene2 = ref[j]
        if(gene1[:len(gene1)-2]=='CLOCK'):
            gene1='CLOCK3111'+gene1[-2:]
        if(gene2[:len(gene2)-2]=='CLOCK'):
            gene2='CLOCK3111'+gene2[-2:]
        combine = gene1[:len(gene1)-2] + gene2[:len(gene2)-2] + gene1[-2:] + gene2[-2:]
        ref.append(combine)

data.drop(columns = ref, inplace=True)

arr = data.nunique()

for col in data.columns:
    if data[col].nunique()==1:
        data.drop(columns = [col], inplace=True)

pyreadstat.write_sav(data, file)