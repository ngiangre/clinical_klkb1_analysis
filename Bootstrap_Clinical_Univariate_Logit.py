
import numpy as np
import pandas as pd
import pickle

num_cores = 4
import time

nboot=1000
from sklearn.utils import shuffle, resample

from sklearn import linear_model

C=1
seed = 42
tol=1e-3

model = {"Logistic Regression" : 
linear_model.LogisticRegression(
	C=C,
	penalty='l1',
	solver="liblinear",
	random_state=seed,
	n_jobs=1,
	max_iter=500,
	fit_intercept=True,
	warm_start=True
	)
		 }

import os

print('Creating folders...')

dir_ = 'bootstrap_clinical_logit/'
try:
	os.mkdir("../../data/"+dir_)
except:
	print(dir_+' exists')

data_dir="../../data/"+dir_

for cohort in ['integrated']:
	try:
		os.mkdir(data_dir+cohort+'/')
	except:
		print(dir_+cohort+'/'+' exists')


def balanced_resample(Y,seed=42):
	"""
	Given a binary pandas series, resample after balancing for equal size of classes
	"""
	
	Y.sort_values(inplace=True)
	num_to_sample = Y.value_counts().min()
	
	dfs = []
	for grp in Y.unique():
		y = Y[Y==grp].head(num_to_sample)
		dfs.append(resample(y,random_state=seed))  
	
	return pd.concat(dfs)

def pull_logit_coefficients(fit):
	
	return fit.coef_[0][0]

def coef_to_prob(coef):
	
	odds = np.exp(coef)
	prob = odds/(1 + odds)
	
	return prob

def coef_to_odds(coef):
	
	odds = np.exp(coef)
	
	return odds

def permute(Y,seed=42):
	"""
	shuffle sample values

	Parameters:
	----------

	Y : pandas series
		Index of samples and values are their class labels
	seed : int
		Random seed for shuffling

	Returns:
	------

	arr_shuffle: pandas series
		A shuffled Y
	"""
	arr = shuffle(Y.values,random_state=seed)
	arr_shuffle = (pd.Series(arr.reshape(1,-1)[0],index=Y.index))
	return arr_shuffle

def permuted_prediction(X,Y,model,seed=42):
	"""
	Given a feature matrix and binary class series, 
	balance then resample y (depends on balanced_resample),
	predict and grab logistic regression coefficients,
	convert and return probability.
	
	"""
	X = X.loc[Y.index]
	Y_shuffle = permute(Y,seed=seed)
	X = X.loc[Y_shuffle.index]
	Y_balanced = resample(Y_shuffle,random_state=seed)
	X_balanced = X.loc[Y_balanced.index]
	
	fit = model.fit(X_balanced,Y_balanced)
	
	coef = pull_logit_coefficients(fit)
	
	return coef_to_odds(coef)

def prediction(X,Y,model,seed=42):
	"""
	Given a feature matrix and binary class series, 
	balance then resample y (depends on balanced_resample),
	predict and grab logistic regression coefficients,
	convert and return probability.
	
	"""
	Y_balanced = resample(Y,random_state=seed)
	X_balanced = X.loc[Y_balanced.index]
	
	fit = model.fit(X_balanced,Y_balanced)
	
	coef = pull_logit_coefficients(fit)
	
	return coef_to_odds(coef)

def balanced_prediction(X,Y,model,seed=42):
	"""
	Given a feature matrix and binary class series, 
	balance then resample y (depends on balanced_resample),
	predict and grab logistic regression coefficients,
	convert and return probability.
	
	"""
	Y_balanced = balanced_resample(Y,seed=seed)
	X_balanced = X.loc[Y_balanced.index]
	
	fit = model.fit(X_balanced,Y_balanced)
	
	coef = pull_logit_coefficients(fit)
	
	return coef_to_odds(coef)

def bootstrap_prediction_transformations(odds_boot,var='variable'):
	df = pd.DataFrame([
		[key for key in odds_boot.keys()],
		[np.median(odds_boot[key]) for key in odds_boot.keys()]
	],
		index=[var,'bootstrap_median']
	).T
	sorted_df = df.sort_values(['bootstrap_median'],ascending=[False])	
	output = (pd.DataFrame.from_dict(odds_boot).
			  reset_index().rename(columns={'index' : 'bootstrap'}).
			  melt(id_vars='bootstrap',var_name=var,value_name='odds').
			  set_index(var).
			  join(sorted_df.set_index(var))
			 )
	odds_wcov_boot = output.reset_index().copy()
	variables = odds_wcov_boot[var].unique()
	err = {}
	for p in variables:
		q = '{} == "{}"'.format(var,p)
		lwr = odds_wcov_boot.query(q).odds.quantile(.025)
		mean = odds_wcov_boot.query(q).odds.mean()
		median = odds_wcov_boot.query(q).odds.quantile(.5)
		upr =odds_wcov_boot.query(q).odds.quantile(.975)
		err[p] = [lwr,mean,median,upr]
	err_df = pd.DataFrame(err,index=['lwr','mean','median','upr']).T.rename_axis(var)
	return output, err_df

from joblib import Parallel, delayed

def bootstrap_of_fcn(func=None,params={},n_jobs=4,nboot=2):	
	if func==None:
		return "Need fcn to bootstrap"
	
	parallel = Parallel(n_jobs=n_jobs)
	return parallel(
		delayed(func)(
			seed=k,**params)
		for k in range(nboot))

print('Bootstrap analyses...')

#### PGD ~ clinical variables
print('#### PGD ~ clinical features')

cohort = "integrated"
X_orig = pd.read_csv("../../data/integrated_sample_groups_imputed_data_raw.csv",index_col=0)
mapd = pd.read_csv("../../data/sample_groups_dtype_map.csv",index_col=0)
num_vars = list(np.intersect1d(mapd.query('dtype=="float"')["var"].values,X_orig.columns))
X_num = X_orig.loc[:,num_vars]
cat_vars = list(np.intersect1d(mapd.query('dtype=="string"')["var"].values,X_orig.columns))
pivots = []
for c in cat_vars:
	s = X_orig[[c]]
	s['mem'] = 1
	pivoted = s.pivot(columns=c,values='mem').fillna(0)
	pivoted.columns = [c+'_'+str(x) for x in pivoted.columns]
	pivots.append(pivoted)

X_cat = pd.concat(pivots,axis=1)
X = pd.concat([X_num,X_cat],axis=1,sort=True)
Y = X_orig.loc[:,'PGD'].map({'Y' : 1, 'N' : 0})

vars_ = [x for x in X.columns if (('set' not in x) & ('tmt_tag' not in x) & ('PGD' not in x) & (not x.endswith('_N')) & (not x.endswith('_M')) & ('Cohort' not in x))]

cov = 'Cohort'
cov_df = X_orig[[cov]]
cov_df['mem'] = 1
cov_df = cov_df.pivot(columns=cov,values='mem').fillna(0)

joined = pd.concat([X,cov_df],axis=1).set_index('Sample')
joined.to_csv('../../data/integrated_sample_groups_imputed_data_raw_all_categories_binarized.csv')


cohort = "integrated"
X_orig = pd.read_csv("../../data/integrated_sample_groups_imputed_data_raw_unique_patients.csv",index_col=0)
mapd = pd.read_csv("../../data/sample_groups_dtype_map.csv",index_col=0)
num_vars = list(np.intersect1d(mapd.query('dtype=="float"')["var"].values,X_orig.columns))
X_num = X_orig.loc[:,num_vars]
cat_vars = list(np.intersect1d(mapd.query('dtype=="string"')["var"].values,X_orig.columns))
pivots = []
for c in cat_vars:
	s = X_orig[[c]]
	s['mem'] = 1
	pivoted = s.pivot(columns=c,values='mem').fillna(0)
	pivoted.columns = [c+'_'+str(x) for x in pivoted.columns]
	pivots.append(pivoted)

X_cat = pd.concat(pivots,axis=1)
X = pd.concat([X_num,X_cat],axis=1,sort=True)
Y = X_orig.loc[:,'PGD'].map({'Y' : 1, 'N' : 0})

vars_ = [x for x in X.columns if (('set' not in x) & ('tmt_tag' not in x) & ('PGD' not in x) & (not x.endswith('_N')) & (not x.endswith('_M')) & ('Cohort' not in x))]

cov = 'Cohort'
cov_df = X_orig[[cov]]
cov_df['mem'] = 1
cov_df = (cov_df.
          pivot(columns=cov,values='mem').
          fillna(0).
          drop('Paris',axis=1))

joined = pd.concat([X_orig.loc[:,'Sample_no_tmt_tag'],X,cov_df],axis=1).set_index('Sample_no_tmt_tag')
joined.to_csv('../../data/integrated_sample_groups_imputed_data_raw_all_categories_binarized_unique_patients.csv')

#bootstrap

boots = []
for var in vars_:
	params = {
	'X' : pd.concat([X[[var]],cov_df],axis=1),
	'Y' : Y,
	'model' : model['Logistic Regression']
	}
	lst = bootstrap_of_fcn(func=prediction,params=params,n_jobs=num_cores,nboot=nboot)
	boots.append(lst)

probs_boot_w2cov = {}
for i, var in enumerate(vars_):
	probs_boot_w2cov[var] = boots[i]

output, err_df = bootstrap_prediction_transformations(probs_boot_w2cov)

output.to_csv(data_dir+cohort+'_logit_bootstrap_pgd_~_clinical_features.csv')
err_df.to_csv(data_dir+cohort+'_logit_bootstrap_pgd_~_clinical_features_lwr_mean_median_upr.csv')

#permuted

boots = []
for var in vars_:
	params = {
	'X' : pd.concat([X[[var]],cov_df],axis=1),
	'Y' : Y,
	'model' : model['Logistic Regression']
	}
	lst = bootstrap_of_fcn(func=permuted_prediction,params=params,n_jobs=num_cores,nboot=nboot)
	boots.append(lst)

probs_boot_w2cov = {}
for i, var in enumerate(vars_):
	probs_boot_w2cov[var] = boots[i]

output, err_df = bootstrap_prediction_transformations(probs_boot_w2cov)

output.to_csv(data_dir+cohort+'_logit_permuted_bootstrap_pgd_~_clinical_features.csv')
err_df.to_csv(data_dir+cohort+'_logit_permuted_bootstrap_pgd_~_clinical_features_lwr_mean_median_upr.csv')

