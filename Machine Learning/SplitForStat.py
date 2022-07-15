import pandas as pd
import os
import os.path
import numpy as np
import pyreadstat


hpath = os.path.dirname(os.path.realpath(__file__)) + '/'

moods = ['BDI', 'STAI', 'SLEEP','SPAQ']
clifeatures = ['AGE', 'MEQ']
SocioStat = ['SOCIOSTATUS_1.0', 'SOCIOSTATUS_2.0']
genes = ['PER2', 'PER3A', 'PER3B', 'PER3C', 'CLOCK', 'CRY1', 'CRY2', 'VNTR']

for mood in moods:
    gdata = pd.read_spss(hpath + mood + ".sav")

    mdata = gdata[gdata['GENDER'] == 0]
    mdata.drop(columns=['GENDER'], inplace=True)

    fdata = gdata[gdata['GENDER'] == 1]
    fdata.drop(columns=['GENDER'], inplace=True)

    gmoodcol = gdata[mood]
    gdata.drop([mood], axis=1, inplace=True)

    mmoodcol = mdata[mood]
    mdata.drop([mood], axis=1, inplace=True)

    fmoodcol = fdata[mood]
    fdata.drop([mood], axis=1, inplace=True)

    for ways in ['1','2']:
        for typ in ['cli', 'nocli']:
            for meth in ['combined','split']:
                for gender in ['male', 'female', 'gendered']:
                    genenum = 0
                    clinum = 0
                    fpath = hpath + 'splits/' + mood + ways + typ + '/' + meth + '/' + gender + '/'
                    if not os.path.exists(fpath):
                        os.makedirs(fpath)          
     
                    if gender == 'male': 
                        tempdata = mdata.copy()
                        moodcol = mmoodcol.copy()
                    elif gender == 'female': 
                        tempdata = fdata.copy()
                        moodcol = fmoodcol.copy()
                    else:
                        tempdata = gdata.copy()
                        moodcol = gmoodcol.copy()

                    if typ == 'nocli': tempdata.drop(clifeatures, axis=1, inplace=True)

                    for col in tempdata.columns:
                        if tempdata[col].nunique()==1:
                            tempdata.drop(columns = [col], inplace=True)

                    tempdata = tempdata.reset_index(drop=True)
                    moodcol = moodcol.reset_index(drop=True)

                    if meth == 'split':
                        for gene in genes:
                            group = []
                            for colname in tempdata.columns:
                                col = colname[:-2]
                                if col==gene: 
                                    group.append(colname)
                            if group:
                                pyreadstat.write_sav(pd.concat([tempdata.loc[:,group], moodcol], axis=1), fpath + 'genegroup' + str(genenum) + '.sav')
                                genenum = genenum + 1
                                tempdata.drop(group, axis=1, inplace=True)

                        if gender == 'gendered':
                            pyreadstat.write_sav(pd.concat([tempdata.loc[:,['GENDER']], moodcol], axis=1), fpath + 'clinicalfeature' + str(clinum) + '.sav')
                            clinum = clinum + 1
                            tempdata.drop('GENDER', axis=1, inplace=True)
                        
                        if typ == 'cli':
                            for cli in clifeatures:
                                pyreadstat.write_sav(pd.concat([tempdata.loc[:,[cli]], moodcol], axis=1), fpath + 'clinicalfeature' + str(clinum) + '.sav')
                                clinum = clinum + 1
                                tempdata.drop(cli, axis=1, inplace=True)
                            pyreadstat.write_sav(pd.concat([tempdata.loc[:,SocioStat], moodcol], axis=1), fpath + 'clinicalfeature' + str(clinum) + '.sav')
                            clinum = clinum + 1
                            tempdata.drop(SocioStat, axis=1, inplace=True)
                        
                        if ways == '2':
                            for colname in tempdata.columns:
                                col = colname[:-4]
                                group = [i for i in tempdata.columns if i.startswith(col)]
                                if group:
                                    pyreadstat.write_sav(pd.concat([tempdata.loc[:,group], moodcol], axis=1), fpath + 'genegroup' + str(genenum) + '.sav')
                                    genenum = genenum + 1
                                    tempdata.drop(group, axis=1, inplace=True)
                    else: 
                        if ways == '2': pyreadstat.write_sav(pd.concat([tempdata, moodcol], axis=1), fpath + 'combined.sav')
                        else:
                            for gene in genes:
                                group = [i for i in tempdata.columns if i.startswith(gene) and len(i)>len(gene)+2]
                                tempdata.drop(group, axis=1, inplace=True)
                                pyreadstat.write_sav(pd.concat([tempdata, moodcol], axis=1), fpath + 'combined.sav')                               