""" LOGISTIC BANK ASSIGNMENT USING ExtraTreeClassifier()"""

"""
Feature importance is an inbuilt class that comes with Tree Based Classifiers, 
we will be using Extra Tree Classifier for extracting the top 10 features for the dataset.
"""


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data=pd.read_csv(r"C:\\EXCELR\NOTES WRITTEN\\SOLVING_ASSIGNMENTS\\Logistic Regression\\bank_data.csv")
data=data.abs()

data.shape
x= data.iloc[:,0:31]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(x,y)

feat_imp=pd.Series(model.feature_importances_,index=x.columns)
feat_imp.nlargest(10).plot(kind='barh')

new_feat=pd.DataFrame(feat_imp.nlargest(10))
new_feat.head(10)#top 10 features
"""

duration     0.323247
balance      0.171755
age          0.152140
campaign     0.073892
poutsuccess  0.062856
pdays        0.040012
previous     0.026268
housing      0.019481
poutfailure  0.012370
poutunknown  0.011944
"""


""" BUT MY DOUBT IS WHEN I USE UNICARIATE METHOD THERE SCORES WERE LITTLE MORE THAN THESE
AND ALSO GOT SOME DIFFERENT FEATURES"""

new_data=data[['duration','balance','age','campaign','poutsuccess','pdays','previous','housing','poutfailure','poutunknown']]

new_data.isnull().sum()
"""
duration       0
balance        0
age            0
campaign       0
poutsuccess    0
pdays          0
previous       0
housing        0
poutfailure    0
poutunknown    0
"""

data.shape
new_data['y']=y
new_data.shape
new_data.columns
from sklearn.model_selection import train_test_split 
train,test=train_test_split(new_data,test_size=0.3) 

import statsmodels.formula.api as smf
logit_model = smf.logit('y~duration+balance+age+campaign+poutsuccess+pdays+previous+housing+poutfailure+poutunknown',data = train).fit()
logit_model.summary()
"""
LLR p-value:                     0.000
Log-Likelihood:                -8271.2
LL-Null:                       -11446.
LLR p-value:                     0.000

===============================================================================
                  coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept      -1.9907      0.148    -13.493      0.000      -2.280      -1.702
duration        0.0040   7.32e-05     54.496      0.000       0.004       0.004
balance      2.768e-05   5.52e-06      5.016      0.000    1.69e-05    3.85e-05
age            -0.0024      0.002     -1.291      0.197      -0.006       0.001
campaign       -0.1202      0.012    -10.005      0.000      -0.144      -0.097
poutsuccess     2.1004      0.111     18.856      0.000       1.882       2.319
pdays          -0.0002      0.000     -0.558      0.577      -0.001       0.000
previous        0.0314      0.011      2.881      0.004       0.010       0.053
housing        -1.0595      0.044    -23.875      0.000      -1.147      -0.973
poutfailure    -0.2886      0.102     -2.833      0.005      -0.488      -0.089
poutunknown    -0.8736      0.124     -7.049      0.000      -1.116      -0.631
===============================================================================

SO HERE AGE AND PDAYS ARE NOT SIGNIFICANT SO LETS APPLY LOG TRANSFORMATION TO THEM ONY BY ONE
"""

mod2=smf.logit('y~duration+balance+age+campaign+poutsuccess+np.log(pdays)+previous+housing+poutfailure+poutunknown',data = train).fit()
mod2.summary()

"""
=================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept        -1.7255      0.289     -5.968      0.000      -2.292      -1.159
duration          0.0040   7.32e-05     54.500      0.000       0.004       0.004
balance        2.769e-05   5.52e-06      5.020      0.000    1.69e-05    3.85e-05
age              -0.0024      0.002     -1.285      0.199      -0.006       0.001
campaign         -0.1200      0.012     -9.995      0.000      -0.144      -0.096
poutsuccess       2.1032      0.110     19.056      0.000       1.887       2.320
np.log(pdays)    -0.0612      0.052     -1.178      0.239      -0.163       0.041
previous          0.0313      0.011      2.878      0.004       0.010       0.053
housing          -1.0570      0.044    -23.901      0.000      -1.144      -0.970
poutfailure      -0.2782      0.102     -2.715      0.007      -0.479      -0.077
poutunknown      -1.1411      0.280     -4.077      0.000      -1.690      -0.592
=================================================================================

STILL PDAYS IS NOT SIGNIFICANT LETS APPLY ON AGE AS WELL
"""


mod3=smf.logit('y~duration+balance+np.log(age)+campaign+poutsuccess+np.log(pdays)+previous+housing+poutfailure+poutunknown',data = train).fit()
mod3.summary()
"""
=================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept        -0.6375      0.399     -1.598      0.110      -1.419       0.144
duration          0.0040   7.33e-05     54.499      0.000       0.004       0.004
balance         2.92e-05   5.51e-06      5.304      0.000    1.84e-05       4e-05
np.log(age)      -0.3230      0.078     -4.151      0.000      -0.475      -0.170
campaign         -0.1190      0.012     -9.923      0.000      -0.142      -0.095
poutsuccess       2.1126      0.110     19.131      0.000       1.896       2.329
np.log(pdays)    -0.0613      0.052     -1.179      0.238      -0.163       0.041
previous          0.0312      0.011      2.872      0.004       0.010       0.053
housing          -1.0748      0.044    -24.388      0.000      -1.161      -0.988
poutfailure      -0.2700      0.102     -2.635      0.008      -0.471      -0.069
poutunknown      -1.1364      0.280     -4.057      0.000      -1.685      -0.587
=================================================================================
STILL, PDAYS IS NOT SIGNIFICANT SO LETS REMOVE TRHAT VARIABLE AND CHECK
"""

mod4=smf.logit('y~duration+balance+np.log(age)+campaign+poutsuccess++previous+housing+poutfailure+poutunknown',data = train).fit()
mod4.summary()
"""
===============================================================================
                  coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept      -0.9452      0.303     -3.124      0.002      -1.538      -0.352
duration        0.0040   7.33e-05     54.494      0.000       0.004       0.004
balance      2.925e-05    5.5e-06      5.314      0.000    1.85e-05       4e-05
np.log(age)    -0.3230      0.078     -4.151      0.000      -0.475      -0.170
campaign       -0.1192      0.012     -9.939      0.000      -0.143      -0.096
poutsuccess     2.1187      0.110     19.202      0.000       1.902       2.335
previous        0.0315      0.011      2.901      0.004       0.010       0.053
housing        -1.0808      0.044    -24.678      0.000      -1.167      -0.995
poutfailure    -0.2835      0.102     -2.787      0.005      -0.483      -0.084
poutunknown    -0.8258      0.098     -8.435      0.000      -1.018      -0.634
===============================================================================

"""


y_pred = mod4.predict(train)
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
0        27287  2563
1          645  1152
"""
acuracy=(27287+1152)/31647#0.8986317818434607

error=1-acuracy# 0.10136821815653929

from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(train.y, y_pred)
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
""" WE GOT BEST CURVE"""
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
roc_auc#0.874319853695834


test_pred = mod4.predict(test)
test["pred_prob"] = test_pred
test["Att_val"]=0

test.shape
test.head()

test.loc[test_pred>=0.5,"Att_val"] = 1

confusion_matrix = pd.crosstab(test.Att_val,test['y'])
"""
y            0     1
Att_val             
0        11678  1060
1          312   514
"""
acuracy=(11678+514)/13564#  0.898849896785609

error=1-acuracy#0.101150103214391

from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(test.y, test_pred)
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
""" WE GOT BEST CURVE again for test as well"""
roc_auc_test = metrics.auc(fpr, tpr) # area under ROC curve 
roc_auc_test#0.8801552119354015

"""
train and test
roc_auc=     0.874319853695834
roc_auc_test=0.8801552119354015

accuracy_train=0.8986317818434607

accuracy_test=0.898849896785609


"""