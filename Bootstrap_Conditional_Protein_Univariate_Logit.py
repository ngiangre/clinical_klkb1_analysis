import numpy as np
import pandas as pd
import pickle

num_cores = 4
import multiprocessing as mp
import time

nboot=200
from sklearn.utils import resample

from sklearn import linear_model

C=1
seed = 42
tol=1e-3

model = {"Logistic Regression" : 
linear_model.LogisticRegression(
	C=C,
	penalty='l1',
	solver="liblinear",
	random_state=seed
	)
		 }

import os

print('Creating folders...')

dir_ = 'bootstrap_conditional_protein_logit/'
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

def prediction(X,Y,model,seed=42):
	"""
	Given a feature matrix and binary class series, 
	balance then resample y (depends on balanced_resample),
	predict and grab logistic regression coefficients,
	convert and return probability.
	
	"""
	Y_balanced = resample(Y,random_state=seed)
	X_balanced = X.loc[Y_balanced.index]

	fit = model.fit(X_balanced,Y_balanced.values)

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
	return parallel(delayed(func)(seed=k,**params) for k in range(nboot))

print('Bootstrap analyses...')

##### PGD ~ protein + cohort - paris
print('##### PGD ~ protein + cohort - paris')
cohort = 'integrated'

Y = pd.read_csv('../../data/'+cohort+'_pgd_y.csv',index_col=0,header=None)
X = pd.read_csv('../../data/'+cohort+'_sample_data_raw.csv',index_col=0).T.apply(lambda x : (x - np.mean(x))/np.std(x),axis=0)
cohort_covs = pd.read_csv('../../data/integrated_cohort_covariates_minus_paris.csv',index_col=0,header=None)
proteins = pickle.load(open('../../data/proteins_immunoglobulins.pkl','rb'))

boots = []
for prot in proteins:
	params = {
	'X' : X[[prot]].join(cohort_covs),
	'Y' : Y,
	'model' : model['Logistic Regression']}
	lst = bootstrap_of_fcn(func=prediction,params=params,n_jobs=num_cores,nboot=nboot)
	boots.append(lst)

odds_boot = {}
for i,prot in enumerate(proteins):
	odds_boot[prot] = boots[i]

output, err_df = bootstrap_prediction_transformations(odds_boot)

output.to_csv(data_dir+cohort+'/logit_bootstrap_pgd_~_protein_+_cohort_-_paris.csv')
err_df.to_csv(data_dir+cohort+'/logit_bootstrap_pgd_~_protein_+_cohort_-_paris_lwr_mean_median_upr.csv')
