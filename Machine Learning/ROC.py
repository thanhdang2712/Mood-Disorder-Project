import pandas as pd
import os
import numpy as np
from sklearn.impute import KNNImputer
import data_preprocess as dp
import uni_multiStats as stats 
from joblib import Parallel, delayed
import pickle
from sklearn.linear_model import SGDClassifier, LogisticRegression
import featureselection as fselect
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

def normal_run(runs_path, path, methods, runs,n_features, n_seed):
    execute_feature_selection(path, runs, methods[1], n_features)
    with open(path+methods[1]+".txt","rb") as fp:
        features = pickle.load(fp)
    
    filename = methods[0] + "_" + methods[1] + ".csv"
    execute_a_run(runs,features, methods[0], filename, runs_path, n_seed)

def execute_feature_selection(path, runs, method, n_features):
    result = Parallel(n_jobs=-1)(delayed(execute_feature_selection_a_run)(run, method, n_features) for  run in runs)
    with open(path+ method+".txt","wb") as  fp:
        pickle.dump(result,fp)
    return result

def execute_feature_selection_a_run(run, method, n_features):
    X_train,  y_train = run[0], run[2]
    arr =[]
    if(method == 'AllFeatures'):
        arr.append(range(n_features))
    else:
        arr.append(fselect.run_feature_selection(method, X_train, y_train))
    return arr

def execute_a_run(runs,features, estimator, filename, path, n_seed):
    for  i  in range(len(runs)//n_seed):
        selection = features[i][0]
        if i==0:
            create_roc_init(runs,selection, estimator, filename, path, n_seed)
        else: 
            concat_roc(runs,selection, estimator, filename, path, i, n_seed)

def create_roc_init(runs,selection, estimator, filename, path, n_seed):
    for i in range(n_seed):
        run = runs[i]
        X_train, X_test = run[0], run[1]
        y_train, y_test = run[2], run[3]
        X_train, X_test = X_train[:,selection], X_test[:,selection]
        res = classify(estimator, X_train, X_test, y_train, y_test)
        res = np.concatenate(([y_test], [res]), axis=0).transpose()
        res = pd.DataFrame(res, columns=['testy_0', 'predict_0'])
        if i==0:
            res.to_csv(path + filename, index=False)
        else:
            df = pd.read_csv(path + filename)
            df = pd.concat([df, res], ignore_index=True)
            df.to_csv(path + filename, index=False)

def concat_roc(runs,selection, estimator, filename, path, idx, n_seed):
    df = pd.DataFrame(columns=['predict_'+str(idx)])

    for i in range(n_seed):
        run = runs[i]
        X_train, X_test = run[0], run[1]
        y_train, y_test = run[2], run[3]
        X_train, X_test = X_train[:,selection], X_test[:,selection]
        res = classify(estimator, X_train, X_test, y_train, y_test)
        res = np.concatenate(([y_test], [res]), axis=0).transpose()
        res = pd.DataFrame(res, columns=['testy_'+str(idx), 'predict_'+str(idx)])
        df = pd.concat([df, res], ignore_index=True)

    roc_file = pd.read_csv(path + filename)
    roc_file = pd.concat([roc_file, df], axis = 1)
    roc_file.to_csv(path + filename, index=False)

def classify(estimator, X_train, X_test, y_train, y_test):
    if estimator == 'svm':
        return SVM(X_train, X_test, y_train, y_test)
    elif estimator == 'naive_bayes':
        return naive_bayes(X_train, X_test, y_train, y_test)
    elif estimator == 'rdforest':
        return rdforest(X_train, X_test, y_train, y_test)
    elif estimator == 'knn':
        return KNN(X_train, X_test, y_train, y_test)
    elif estimator == 'elasticnet':
        return elasticnet(X_train, X_test, y_train, y_test)
    elif estimator =='xgboost':
        return xgboost(X_train, X_test, y_train, y_test)
    elif estimator =='logreg':
        return logistic_regression(X_train, X_test, y_train, y_test)

def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=10000, solver='lbfgs')
    model.fit(X_train, y_train)
    lr_probs = model.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    return lr_probs

def elasticnet(X_train,X_test,y_train,y_test):
    model = SGDClassifier(loss = 'log',alpha= 0.001,penalty = 'l1', l1_ratio=0.1, random_state=0)
    model.fit(X_train, y_train)
    lr_probs = model.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    return lr_probs

def xgboost(X_train,X_test,y_train,y_test):
    model = XGBClassifier(booster='gbtree', use_label_encoder =False, max_depth=6, n_estimators = 450, learning_rate=0.05)
    model.fit(X_train, y_train)
    lr_probs = model.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    return lr_probs

def naive_bayes(X_train,X_test,y_train,y_test):
    model = BernoulliNB()
    model.fit(X_train, y_train)
    lr_probs = model.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    return lr_probs

def rdforest(X_train,X_test,y_train,y_test):
    model = RandomForestClassifier(random_state=0, n_jobs=-1, max_depth=10, n_estimators = 500)
    model.fit(X_train, y_train)
    lr_probs = model.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    return lr_probs

def SVM(X_train,X_test,y_train,y_test):
    model = SVC(C=0.1, random_state=0, degree=2, gamma =1, max_iter=100000, kernel = 'poly')
    clf = CalibratedClassifierCV(model)
    clf.fit(X_train, y_train)
    lr_probs = clf.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    return lr_probs

def KNN(X_train,X_test,y_train,y_test):
    model = KNeighborsClassifier(n_neighbors=16, n_jobs=-1)
    model.fit(X_train, y_train)
    lr_probs = model.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    return lr_probs

def graph(target_path, methods, color_dictionary, n_seed):
    thresholds = np.linspace(0, 1, num=20)
    ns_x = [0, 1]
    ns_y = ns_x
    f = open(target_path+'AUC.txt','w+')
    pyplot.plot(ns_x, ns_y, linestyle='--')
    for m in methods:
        if m[0] == 'svm':
            label = 'SVM'
        elif m[0] == 'naive_bayes':
            label = 'NB'
        elif m[0] == 'rdforest':
            label = 'RF'
        elif m[0] == 'knn':
            label = 'KNN'
        elif m[0] == 'elasticnet':
            label = 'EN'
        elif m[0] =='xgboost':
            label = 'XGB'
        elif m[0] =='logreg':
            label = 'LR'
        file = pd.read_csv(target_path + "runs/" + m[0] + "_" + m[1] + ".csv")
        df = pd.DataFrame()
        for i in range(0, n_seed):
            testy = file['testy_'+str(i)]
            lr_probs = file['predict_'+str(i)]
            [lr_fpr, lr_tpr] = roc_curve(testy.to_numpy(), lr_probs.to_numpy(), thresholds)
            df["lr_fpr_"+str(i)] = lr_fpr
            df["lr_tpr_"+str(i)] = lr_tpr


        df['final_lr_tpr'] = df.loc[:, df.columns.str.startswith("lr_tpr")].mean(axis=1)
        df['final_lr_fpr'] = df.loc[:, df.columns.str.startswith("lr_fpr")].mean(axis=1)
        df.to_csv(target_path + "roc_values_"+ m[0] + "_" + m[1] +".csv", index=False)
        final_lr_tpr = df['final_lr_tpr']
        final_lr_fpr = df['final_lr_fpr']
        f.write(label + ': '+str(auc(final_lr_fpr, final_lr_tpr))+')\n')
        pyplot.plot(final_lr_fpr, final_lr_tpr, label= label + ' (AUC: '+str(round(auc(final_lr_fpr, final_lr_tpr), 2))+')', color=color_dictionary[m[0]])

        pyplot.xlabel('False Positive Rate', fontsize=18)
        pyplot.ylabel('True Positive Rate', fontsize=18)
        pyplot.xticks([0,1])
        pyplot.yticks([0,1])
        pyplot.tick_params(labelsize=14)
        pyplot.legend()
    f.close()
    pyplot.tight_layout()
    pyplot.savefig(target_path + "roc_curve.png")

def roc_curve(y_true, y_prob, thresholds):
    fpr = []
    tpr = []

    for threshold in thresholds:

        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]


datafile=""
target = ""

type="SMOTE"

directory_path = os.path.dirname(os.path.realpath(__file__))+'/'
target_path = directory_path+'roc/'+datafile+"_"+target+"_"+ type +"/"
runs_path = target_path + "runs/"
if not os.path.exists(runs_path):
        os.makedirs(runs_path)

data = pd.read_spss(directory_path+"data/"+datafile+".sav")

datacopy = data.copy(deep=True)
datacopy.replace('', np.nan,regex=True, inplace=True)
datacopy = dp.remove_invalid_data(datacopy, target)

columns_org = datacopy.columns
[datacopy, continuous] = dp.modify_data(datacopy)
columns_dummified = datacopy.columns
n_features = datacopy.shape[1]-1

if type == "SMOTE":
    [datacopy, MinMax] = dp.scale(datacopy, continuous)
    y = datacopy[target]
    X = datacopy.drop(columns=[target])
    
    imp_dum = KNNImputer(n_neighbors = 5)
    X.iloc[:] = imp_dum.fit_transform(X)

    X = stats.derive_class(X, columns_org.drop(continuous))

    over = SMOTE(sampling_strategy=1, random_state=5)
    X_res, y_res = over.fit_resample(X, y)

    datacopy = pd.concat([X_res, y_res], axis=1)
    datacopy = dp.rescale(datacopy, continuous, MinMax)

n_seed = 10
splits = 10
runs = stats.runSKFold(n_seed,splits,data=datacopy,target=target, columns_org=columns_org, continuous=continuous, columns_dummified=columns_dummified)

methods = [["logreg", "AllFeatures"], ["rdforest", "AllFeatures"], ["xgboost", "AllFeatures"], ["svm", "AllFeatures"]]
for m in methods:
    normal_run(runs_path, target_path, m, runs, n_features, n_seed)

color_dictionary = {'svm': 'b', 'naive_bayes':'g', 'rdforest':'r', 'knn':'c', 'elasticnet':'m', 'xgboost':'k', 'logreg':'y'}

graph(target_path, methods, color_dictionary, n_seed)