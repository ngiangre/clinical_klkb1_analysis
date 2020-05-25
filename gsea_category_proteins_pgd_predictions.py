import numpy as np
import pandas as pd
import pickle
import time
import os
from prediction_functions import *


cv_split = 10
n_jobs = 50
nboot=200
test_size = 0.15
treat='PGD'

classification_metrics = ['roc_auc','recall','precision','f1']

type_='ledge_protein'
basename = type_+'_features_pgd_prediction_'
dir_ = '../../data/'

cohort = 'integrated'

t0=time.time()

try:
	os.mkdir(dir_+cohort+'_pgd_predictions/gsea_categories/')
except:
	print(dir_+cohort+'_pgd_predictions/gsea_categories/'+' exists')


X = pd.read_csv(dir_+cohort+'_X_raw_all_proteins.csv',index_col=0)
proteins_immunoglobulins = pickle.load(open('../../data/proteins_immunoglobulins.pkl','rb'))
X = X.loc[:,proteins_immunoglobulins]

rich = pd.read_csv('../../data/integrated_bootstrap_conditional_protein_logit'+
                   '_mean_prerank_report_all_categories_tall.csv',index_col=0)

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

terms = rich[['Term']].drop_duplicates().Term.unique()

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


for term in terms:
        ledge_genes = rich.query('Term==@term').ledge_gene.unique()
        ledge_features = (idmap_sub[idmap_sub.Gene_name.isin(ledge_genes)].
                    groupby('Gene_name').
                    first().
                    reset_index().
                    Protein.
                    unique()
                    )
        features = np.intersect1d(X.columns.values,ledge_features)
        
        params.update({'X' : X[features],'models' : l1_logit_model.copy()})

        out_dir = dir_+cohort+'_pgd_predictions/gsea_categories/'+basename+term.replace('/','_')+'_proteins_prediction_'

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

        print(t1-t0)
