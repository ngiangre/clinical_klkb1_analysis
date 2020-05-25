import numpy as np
import pandas as pd
import pickle
import time
import random
import os
from prediction_functions import *

metric = 'roc_auc'
cv_split = 10
n_jobs = 10
nboot=200
test_size = 0.15
treat='PGD'
classification_metrics = ['roc_auc','precision','recall']

type_='clinical_01_within_notwithcohorts'
basename = type_+'_features_pgd_prediction_'
dir_ = '../../data/'
cohort = 'integrated'

t0_all=time.time()

try:
    os.mkdir(dir_+cohort+'_pgd_predictions/')
except:
    print(dir_+cohort+'_pgd_predictions/'+' exists')


X_all_proteins = pd.read_csv(dir_+cohort+'_X_raw_all_proteins.csv',index_col=0)
proteins_no_immunoglobulins = pickle.load(open('../../data/proteins_no_immunoglobulins.pkl','rb'))
X_all_proteins = X_all_proteins.loc[:,proteins_no_immunoglobulins]

X_all_clinical = pd.read_csv(dir_+cohort+'_X_clinical_and_cohort_covariates.csv',index_col=0)
Y = pd.read_csv(dir_+cohort+'_pgd_y.csv',index_col=0,header=None)

idmap_sub = pd.read_csv('../../data/protein_gene_map_full.csv')[['Protein','Gene_name']].dropna()

cov_df = X_all_clinical.loc[:,['Cohort_Columbia','Cohort_Cedar']].copy().astype(int)
all_cov_df = cov_df.copy()
all_cov_df.loc[:,'Cohort_Paris'] = (
    (all_cov_df['Cohort_Columbia'] + 
     all_cov_df['Cohort_Cedar'])==0).astype(int)


params = {'Y' : Y, 'cv_split' : cv_split, 
		  'metrics' : classification_metrics, 'n_jobs' : 1, 
		  'test_size' : test_size,
		  'retrained_models' : True, 'patient_level_predictions' : True}

features = [[x] for x in np.setdiff1d(X_all_clinical.columns.values,all_cov_df.columns.values)]

def get_performance(lst):
    perf = (pd.
            concat(lst,keys=range(len(lst))).
            reset_index(level=1,drop=True).
            rename_axis('bootstrap').
            reset_index()
           )
    return perf

def model_feature_importances(boot_mods):
    dfs = []
    X = params['X'].copy()
    X.loc[:,'Intercept'] = 0
    for i in range(len(boot_mods)):
        for j in boot_mods[i].keys():
            mod = boot_mods[i][j]
            coef = []
            try:
                coef.extend([i for i in mod.feature_importances_])
            except:
                coef.extend([i for i in mod.coef_[0]])
            coef.extend(mod.intercept_)
            fs = []
            fs.extend(X.columns.values)
            df = pd.DataFrame({
                'Feature' : fs,
                'Gene_name' : (X.T.
                               join(idmap_sub.
                                    set_index('Protein'),how='left').
                               Gene_name.values),
                'Importance' : coef,
                'Model' : j,
                'Bootstrap' : i
            })
            dfs.append(df)
    return pd.concat(dfs,sort=True)

def patient_predictions(lst):
        dat = \
        (pd.
         concat(
             lst
         ).
         reset_index().
         rename(columns={0 : 'Sample'}).
         set_index('Sample').
         join(all_cov_df).
         reset_index().
         melt(id_vars=['Sample','bootstrap','model','y_true','y_pred','y_proba'],
              var_name='cohort',value_name='mem')
        )
        dat.cohort = dat.cohort.str.split('_').apply(lambda x : x[1])
        dat = dat[dat.mem==1].drop('mem',1).reset_index(drop=True)
        return dat


for feature in features:

    jfeature = '_'.join(feature)
    out_dir = dir_+cohort+'_pgd_predictions/'+basename+jfeature.replace('/','_')+'_prediction_'
    print(out_dir)
    
    X_all = X_all_proteins.join(X_all_clinical)
    if type(feature)==str:
        X = X_all[[feature]]
    if type(feature)==list:
        X = X_all[feature]
    params.update(
        {'X' : X,
         'models' : l1_logit_model.copy()
        }
    )

    #bootstrap
    print('bootstrap...')
    t0=time.time()
    lst = bootstrap_of_fcn(func=train_test_val_top_fold_01_within,
                           params=params,n_jobs=n_jobs,nboot=nboot)

    perf = get_performance([lst[i][0] for i in range(nboot)])
    perf.to_csv(out_dir+'metric_bootstrap_train_test_val.csv')
    
    boot_mods = [lst[i][1] for i in range(nboot)]
    pickle.dump(boot_mods,open(out_dir+'metric_bootstrap_train_test_val_models.pkl','wb'))
                            
    fimp = model_feature_importances(boot_mods)
    fimp.to_csv(out_dir+'metric_bootstrap_train_test_val_feature_importances.csv')

    ppreds = patient_predictions([lst[i][2] for i in range(nboot)])
    ppreds.to_csv(out_dir+'metric_bootstrap_train_test_val_patient_level_data.csv')
    
    #permute
    print('permute...')

    plst = bootstrap_of_fcn(func=permuted_train_test_val_top_fold_01_within,
                           params=params,n_jobs=n_jobs,nboot=nboot)

    pperf = get_performance([plst[i][0] for i in range(nboot)])
    pperf.to_csv(out_dir+'metric_permute_train_test_val.csv')
    
    pboot_mods = [plst[i][1] for i in range(nboot)]
    pickle.dump(pboot_mods,open(out_dir+'metric_permute_train_test_val_models.pkl','wb'))
                            
    pfimp = model_feature_importances(pboot_mods)
    pfimp.to_csv(out_dir+'metric_permute_train_test_val_feature_importances.csv')

    pppreds = patient_predictions([plst[i][2] for i in range(nboot)])
    pppreds.to_csv(out_dir+'metric_permute_train_test_val_patient_level_data.csv')
    
    t1 = time.time()
    print(np.round( (t1 - t0) / 60, 2 ) ) 
    
t1_all = time.time()
print(np.round( (t1_all - t0_all) / 60, 2 ) ) 
