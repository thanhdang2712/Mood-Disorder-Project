import pandas as pd
import glob, os
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))+'/'
path = dir + 'tablefolders/'
print(path)
for folder in glob.glob(path+'*'):
    fselect = pd.read_csv(folder + '/FeatureSelectionSummary.csv').drop('AllFeaturesFEATURES.csv', axis=1)
    df = pd.DataFrame(columns = fselect.columns)
    for col in fselect.columns.drop('feature'):
        df[col]=0
        topfeat = fselect.loc[fselect[col] >= 95, 'feature'] 
        for feat in topfeat:
            if not feat in df['feature'].values:
                newrow = [feat] + ([0]*(df.shape[1]-1))
                df.loc[len(df.index)] = newrow
            df.loc[df['feature']==feat, [col]] = 1
    
    for st in glob.glob(folder + '/RegStat/*'):
        print(st)
        file = st.split('/')[-1]
        file = file.split('_')[0]
        statsfile = pd.read_csv(st)
        df[file]=0
        sigfeat = statsfile.loc[statsfile['Adjusted_P'] <= 0.1, 'Variable_Name'] 
        
        for feat in sigfeat:
            if not feat == '(Intercept)':
                if not feat in df['feature'].values:
                    newrow = [feat] + ([0]*(df.shape[1]-1))
                    df.loc[len(df.index)] = newrow
                df.loc[df['feature']==feat, file] = 1 
    df.drop(['jmi30', 'infogain30', 'mrmr30', 'reliefF30'], inplace=True, axis=1)
    drop =[]
    sort=[]
    for i in range(len(df)):
        if (df.iloc[i, 1:]).sum()<3:
            drop.append(i)
        else:
            sort.append((df.iloc[i, 1:]).sum())


    df.drop(drop, inplace=True)
    df['sort']=sort
    df.reset_index(drop=True, inplace=True)

    df.sort_values(by=['sort'], ignore_index=True, inplace=True, ascending=False)
    df.drop('sort', inplace=True, axis=1)
    #df.set_index('feature', inplace=True)
    df.to_csv(folder+'.csv')

    data = pd.read_spss(dir + 'DATA/' + folder.split('/')[-1]+ '.sav', usecols=np.append(df['feature'].to_numpy(), ['BDI', 'STAI', 'SPAQ', 'SLEEP']))
    data.to_csv(folder+'_data.csv')


# dir = os.path.dirname(os.path.realpath(__file__))+'/'
# path = dir + 'Epid/'
# df = pd.DataFrame()
# df['feature_ ']=0
# for folder in ['sth', 'prot', 'paras', 'helm']:
#     fselect = pd.read_csv(path + folder + '/FeatureSelectionSummary.csv').drop('AllFeaturesFEATURES.csv', axis=1)
#     for col in fselect.columns.drop('feature'):
#         df[col + '_' + folder]=0
#         topfeat = fselect.loc[fselect[col] >= 95, 'feature'] 
#         for feat in topfeat:
#             if not feat in df['feature_ '].values:
#                 newrow = [feat] + ([0]*(df.shape[1]-1))
#                 df.loc[len(df.index)] = newrow
#             df.loc[df['feature_ ']==feat, [col + '_' + folder]] = 1

#     statsfile = pd.read_csv(path + folder + '/Compiled_P.csv')
#     df['uni_' + folder]=0
#     sigfeat = statsfile.loc[statsfile['P_Value_uni'] <= 0.05, 'Variable_Name'] 
    
#     for feat in sigfeat:
#         if not feat == '(Intercept)':
#             if not feat in df['feature_ '].values:
#                 newrow = [feat] + ([0]*(df.shape[1]-1))
#                 df.loc[len(df.index)] = newrow
#             df.loc[df['feature_ ']==feat, 'uni_' + folder] = 1 


#     df['multi_' + folder]=0
#     sigfeat = statsfile.loc[statsfile['P_Value_multi'] <= 0.05, 'Variable_Name'] 
    
#     for feat in sigfeat:
#         if not feat == '(Intercept)':
#             if not feat in df['feature_ '].values:
#                 newrow = [feat] + ([0]*(df.shape[1]-1))
#                 df.loc[len(df.index)] = newrow
#             df.loc[df['feature_ ']==feat, 'multi_' + folder] = 1 



# drop =[]
# sort=[]
# for i in range(len(df)):
#     if (df.iloc[i, 1:]).sum()==1:
#         drop.append(i)
#     else:
#         sort.append((df.iloc[i, 1:]).sum())


# df.drop(drop, inplace=True)
# df['sort']=sort
# df.reset_index(drop=True, inplace=True)

# df.sort_values(by=['sort'], ignore_index=True, inplace=True, ascending=False)
# df.drop('sort', inplace=True, axis=1)
# #df.set_index('feature', inplace=True)
# df.columns = pd.MultiIndex.from_tuples([(c.split('_')[1], c.split('_')[0]) for c in df.columns])
# df.to_csv(dir + 'Epid/Ticktable.csv')