# clinical\_klkb1\_analysis

[![DOI](https://zenodo.org/badge/266758068.svg)](https://zenodo.org/badge/latestdoi/266758068)

Code repository (meant to be viewed and not executed) for 'Plasma kallikrein predicts primary graft dysfunction after heart transplant' by Giangreco et al.

The clinical characteristics table-one tables in the paper were created using *tableone.R* 

The paper results include prediction tasks with clinical, protein, and both markers. 

For population association analyses, we have the *Bootstrap\_Clinical\_Multivariate\_Logit.py* and *Bootstrap\_Clinical\_Uniivariate\_Logit.py* scripts as well as the *Bootstrap\_Conditional\_Protein\_Multivariate\_Logit.py* and *Bootstrap\_Conditional\_Protein\_Univariate\_Logit.py*
scripts. 

For prediction analyses, the monte carlo cross validation methods are found in *prediction\_functions.py*. Univariate MCCVs were estimated in *individual\_protein\_pgd\_predictions.py* and *individual\_clinical\_pgd\_predictions.py*. Bivariate MCCVs were estimated in *marker\_combos\_pgd\_prediction.py*. Multivariate MCCvs with gene sets were estimated in *gsea\_category\_proteins\_pgd\_predictions*. 

The results and figures were generated within *paper\_figure\_code.ipynb*. Further figure collation was done using [Biorender](https://biorender.com/).
