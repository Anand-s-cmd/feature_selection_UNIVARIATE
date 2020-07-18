""" UNIVARIATE FEATURE SELECTION """

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data=pd.read_csv(r"C:\\EXCELR\NOTES WRITTEN\\SOLVING_ASSIGNMENTS\\Logistic Regression\\bank_data.csv")
data=data.abs()

data.shape
x= data.iloc[:,0:31]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

#apply SelectKBest class to extract top 10 best features
bestfeatures=SelectKBest(score_func=chi2, k=10)
fit=bestfeatures.fit(x,y)

dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x.columns)


#concat two dataframes for better visualization 
featurescores=pd.concat([dfcolumns,dfscores],axis=1)

featurescores.columns = ['Specs','Score']  #naming the dataframe columns
print(featurescores.nlargest(10,'Score')) 
featurescores.head(10)
"""
           Specs         Score
5       duration  1.807711e+06
2        balance  7.223507e+05
7          pdays  1.134616e+05
11   poutsuccess  4.113001e+03
8       previous  3.593220e+03
6       campaign  8.405821e+02
15   con_unknown  7.333549e+02
3        housing  3.889497e+02
13  con_cellular  2.940171e+02
24     joretired  2.696993e+02

CONSIDER THESE ONLY COLUMNS THEN
"""
#new_data=pd.DataFrame(featurescores.nlargest(10,'Score'))
new_data=data[['duration','balance','pdays','poutsuccess','previous','campaign','con_unknown','housing','con_cellular','joretired']]

new_data.isnull().sum()
"""
duration        0
balance         0
pdays           0
poutsuccess     0
previous        0
campaign        0
con_unknown     0
housing         0
con_cellular    0
joretired       0
"""

new_data.columns
"""
we need o add y output variable to the new data set
"""

data.shape
new_data['y']=y
new_data.shape
new_data.columns
from sklearn.model_selection import train_test_split 
train,test=train_test_split(new_data,test_size=0.3) 

import statsmodels.formula.api as smf
logit_model = smf.logit('y~duration+balance+pdays+poutsuccess+previous+campaign+con_unknown+housing+con_cellular+joretired',data = train).fit()
logit_model.summary()
"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                31647
Model:                          Logit   Df Residuals:                    31636
Method:                           MLE   Df Model:                           10
Date:                Sat, 07 Dec 2019   Pseudo R-squ.:                  0.2941
Time:                        06:05:38   Log-Likelihood:                -8074.2
converged:                       True   LL-Null:                       -11438.
                                        LLR p-value:                     0.000
================================================================================
                   coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept       -2.9290      0.094    -31.185      0.000      -3.113      -2.745
duration         0.0040   7.34e-05     54.051      0.000       0.004       0.004
balance       2.345e-05   5.51e-06      4.254      0.000    1.26e-05    3.43e-05
pdays            0.0011      0.000      4.851      0.000       0.001       0.001
poutsuccess      2.4671      0.079     31.236      0.000       2.312       2.622
previous         0.0205      0.009      2.195      0.028       0.002       0.039
campaign        -0.1236      0.012    -10.274      0.000      -0.147      -0.100
con_unknown     -1.0476      0.104    -10.075      0.000      -1.251      -0.844
housing         -0.8564      0.045    -18.899      0.000      -0.945      -0.768
con_cellular     0.1973      0.085      2.329      0.020       0.031       0.363
joretired        0.4523      0.079      5.758      0.000       0.298       0.606
================================================================================
"""

y_pred = logit_model.predict(train)
train["pred_prob"] = y_pred
train.head(2)

train["Att_val"]=0

train.loc[y_pred>=0.5,"Att_val"] = 1
train.head(10)
train.columns

confusion_matrix = pd.crosstab(train.Att_val,train['y'])
"""
y            0     1
Att_val             
0        27306  2533
1          649  1159
"""
acuracy=(27306+1159)/31647# 0.8994533447088192

error=1-acuracy#0.10054665529118079

from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(train.y, y_pred)
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
""" WE GOT BEST CURVE"""
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
roc_auc#0.8850430569327387


test_pred = logit_model.predict(test)
test["pred_prob"] = test_pred
test["Att_val"]=0

test.shape
test.head()

test.loc[test_pred>=0.5,"Att_val"] = 1

confusion_matrix = pd.crosstab(test.Att_val,test['y'])
"""
y            0     1
Att_val             
0        11722  1086
1          257   499
"""
acuracy=(11722+499)/13564#  0.9009879091713359

error=1-acuracy#0.09901209082866413

from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(test.y, test_pred)
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
""" WE GOT BEST CURVE again for test as well"""
roc_auc_test = metrics.auc(fpr, tpr) # area under ROC curve 
roc_auc_test#0.8813502230375292

"""
train and test
roc_auc=     0.8850430569327387
roc_auc_test=0.8813502230375292

accuracy_train=0.8994533447088192
accuracy_test=0.9009879091713359


"""