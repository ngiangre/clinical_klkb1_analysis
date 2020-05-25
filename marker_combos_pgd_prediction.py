import numpy as np
import pandas as pd
import pickle
import time
import random
import os
from prediction_functions import *

metric = 'roc_auc'
cv_split = 10
n_jobs = 20
nboot=200
test_size = 0.15
treat='PGD'

classification_metrics = ['roc_auc']

basename = 'raw_01_within_notwithcohorts_clinicalclinical_proteinclinical_proteinprotein_and_clinical_and_protein_features_small_combos_pgd_prediction_'
dir_ = '../../data/'
cohort = 'integrated'
out_dir = dir_+cohort+'_pgd_predictions/'+basename

X_all_proteins = pd.read_csv(dir_+cohort+'_X_raw_all_proteins.csv',index_col=0)
X_all_clinical = pd.read_csv(dir_+cohort+'_X_clinical_and_cohort_covariates.csv',index_col=0)
Y = pd.read_csv(dir_+cohort+'_pgd_y.csv',index_col=0,header=None)

cov_df = X_all_clinical.loc[:,['Cohort_Columbia','Cohort_Cedar']].copy().astype(int)
all_cov_df = cov_df.copy()
all_cov_df.loc[:,'Cohort_Paris'] = (
    (all_cov_df['Cohort_Columbia'] + 
     all_cov_df['Cohort_Cedar'])==0).astype(int)

idmap_sub = pd.read_csv('../../data/protein_gene_map_full.csv')[['Protein','Gene_name']].dropna()

query='mean_validation_roc_auc>0.5 &'+ \
           ' (odds_lwr>1 | odds_upr<1) & '+ \
           '(permuted_odds_lwr<1 & permuted_odds_upr>1) &'+ \
           'importance_bonferroni<0.001 & (importance_bonferroni>=importance_p_value)'
predictive_proteins =  \
(pd.
 read_csv('../../data/protein_raw_01_within_notwithcohorts_mccv_performance_significance_and_feature_odds_df.csv',
          index_col=0).
 query(query).
 feature.
 unique()
)
predictive_clinicals =  \
(pd.
 read_csv('../../data/clinical_01_within_notwithcohorts_mccv_performance_significance_and_feature_odds_df.csv',
          index_col=0).
 query(query).
 feature.
 unique()
)

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

params = {'Y' : Y, 'cv_split' : cv_split, 
		  'metrics' : classification_metrics, 'n_jobs' : 1, 
		  'test_size' : test_size,
		  'retrained_models' : True, 'patient_level_predictions' : True}

import itertools
clin_combos = [[list(i) for i in itertools.combinations(
    np.intersect1d(
        predictive_clinicals,
        X_all_clinical.columns.values),r)
               ] for r in np.arange(1,2)]
prot_combos = [[list(i) for i in itertools.combinations(
    np.intersect1d(
        predictive_proteins,
        X_all_proteins.columns.values),r)
               ] for r in np.arange(1,2)]

all_clin_1 = list(np.concatenate(list(itertools.chain(*clin_combos))))
print(len(all_clin_1))

all_prot_1 = list(np.concatenate(list(itertools.chain(*prot_combos))))
print(len(all_prot_1))

all_clin_1_and_prot_1 = list(
    itertools.chain(*[all_clin_1,all_prot_1])
)
print(len(all_clin_1_and_prot_1))

all_clin_1_prot_1 = list(
    itertools.chain(*
                    [[list(itertools.chain(*[[x],[y]])) for x in all_prot_1] for y in all_clin_1]
                   )
)
print(len(all_clin_1_prot_1))

all_clin_1_prot_1_and_clin_1_and_prot_1 = list(
    itertools.chain(*[all_clin_1,all_prot_1,all_clin_1_prot_1])
)
print(len(all_clin_1_prot_1_and_clin_1_and_prot_1))

all_clin_2 = [list(i) for i in itertools.combinations(all_clin_1,2)]
print(len(all_clin_2))

all_prot_2 = [list(i) for i in itertools.combinations(all_prot_1,2)]
print(len(all_prot_2))

all_clin_1_prot_1_and_prot_2 = list(
    itertools.chain(*[all_clin_1_prot_1,all_prot_2])
)
len(all_clin_1_prot_1_and_prot_2)

all_clin_2_and_clin_1_prot_1_and_prot_2 = list(
    itertools.chain(*[all_clin_2,all_clin_1_prot_1,all_prot_2])
)
len(all_clin_2_and_clin_1_prot_1_and_prot_2)

all_clin_2_and_clin_1_prot_1_and_prot_2_and_clin_1_and_prot_1 = list(
    itertools.chain(*[all_clin_2,all_clin_1_prot_1,all_prot_2,all_clin_1,all_prot_1])
)
print(len(all_clin_2_and_clin_1_prot_1_and_prot_2_and_clin_1_and_prot_1))

fimps_dfs = []
perf_dfs = []
ppreds_dfs = []
perm_fimps_dfs = []
perm_perf_dfs = []
perm_ppreds_dfs = []
feature_set = {}
for i,features in enumerate(all_clin_2_and_clin_1_prot_1_and_prot_2_and_clin_1_and_prot_1):
        X_all = X_all_proteins.join(X_all_clinical)
        if type(features)==str:
            X = X_all[[features]]
        if type(features)==list:
            X = X_all[features]
        feature_set[str(i)] = X.columns.tolist()
        params.update({'X' : X,'models' : l1_logit_model.copy()})
        lst = bootstrap_of_fcn(func=train_test_val_top_fold_01_within,
                       params=params,n_jobs=n_jobs,nboot=nboot)
        perf = get_performance([lst[i][0] for i in range(len(lst))])
        perf['set'] = str(i)
        perf_dfs.append(perf)
        fimps = model_feature_importances([lst[i][1] for i in range(len(lst))])
        fimps['set'] = str(i)
        fimps_dfs.append(fimps)
        ppreds = patient_predictions([lst[i][2] for i in range(len(lst))])
        ppreds['set'] = str(i)
        ppreds_dfs.append(ppreds)
        
        lst = bootstrap_of_fcn(func=permuted_train_test_val_top_fold_01_within,
               params=params,n_jobs=n_jobs,nboot=nboot)
        perm_perf = get_performance([lst[i][0] for i in range(len(lst))])
        perm_perf['set'] = str(i)
        perm_perf_dfs.append(perm_perf)
        perm_fimps = model_feature_importances([lst[i][1] for i in range(len(lst))])
        perm_fimps['set'] = str(i)
        perm_fimps_dfs.append(perm_fimps)
        perm_ppreds = patient_predictions([lst[i][2] for i in range(len(lst))])
        perm_ppreds['set'] = str(i)
        perm_ppreds_dfs.append(perm_ppreds)
        
perf_df = (pd.concat(perf_dfs).
           groupby(['set'])['validation_roc_auc'].
           describe(percentiles=[0.025,0.975]).
           loc[:,['2.5%','mean','97.5%']].
           sort_values('2.5%',ascending=False).
           reset_index()
          )
fimps_df = (pd.concat(fimps_dfs).
            groupby(['set','Feature'])['Importance'].
            describe(percentiles=[0.025,0.975]).
            loc[:,['2.5%','mean','97.5%']].
            sort_values('2.5%',ascending=False).
            reset_index()
          )
ppreds_df = (pd.concat(ppreds_dfs))

perm_perf_df = (pd.concat(perm_perf_dfs).
           groupby(['set'])['validation_roc_auc'].
           describe(percentiles=[0.025,0.975]).
           loc[:,['2.5%','mean','97.5%']].
           sort_values('2.5%',ascending=False).
           reset_index()
          )
perm_fimps_df = (pd.concat(perm_fimps_dfs).
            groupby(['set','Feature'])['Importance'].
            describe(percentiles=[0.025,0.975]).
            loc[:,['2.5%','mean','97.5%']].
            sort_values('2.5%',ascending=False).
            reset_index()
          )
perm_ppreds_df = (pd.concat(perm_ppreds_dfs))


perf_df.to_csv(out_dir+'agg_performance.csv')
fimps_df.to_csv(out_dir+'agg_feature_importances.csv')
ppreds_df.to_csv(out_dir+'agg_patient_level_data.csv')
perm_perf_df.to_csv(out_dir+'agg_permuted_performance.csv')
perm_fimps_df.to_csv(out_dir+'agg_permuted_feature_importances.csv')
perm_ppreds_df.to_csv(out_dir+'agg_permuted_patient_level_data.csv')
pickle.dump(feature_set,open(out_dir+'feature_set_dictionary.pkl','wb'))
