
## All necessary modules as well as different functions that will be used in this work are explicit here.
#import all neccesary modules
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle,gzip
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn import tree
import graphviz
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#import modules created 
import my_utils
import missing_val_imput
import feature_select
import preprocessing
import adhoc_transf

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Classifier models to use
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

import joblib
#Too see all columns when describe or head is called
pd.set_option('display.max_columns', 15)

#importing file into a pandas dataframe from UCI repository
#https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records#
path_data=r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\heart_failure_clinical_records_dataset.csv'

df=pd.read_csv(path_data)

df.head()
#**********for a possible analysis for heart failure risk prediction
#df2=df.loc[df['age']>60]
#df2['age'].hist()
#df2.info()

df.describe()

df['DEATH_EVENT'].value_counts()
#the data set is quite unbalanced 203 alive and 96 dead

#Let's see the type of these features, apart from the proportion of non-null values
my_utils.info_adhoc(df) #there is no missing values which is great

#Let's see if there is any weird character in the dataset
my_utils.df_values(df)

def rand_jitter(arr):
    np.random.seed(42)
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

norm = np.linalg.norm
import random
from random import sample 

def SMOTE(data, sampling_rate, n_neigh, random_state=42):
    random.seed(random_state)
    new_samples = []
    
    if sampling_rate==0:
        return
    
    if sampling_rate>n_neigh: return      
    data = data.reset_index(drop=True)

    n_samples = data.count()[0]

    for i in range(n_samples):
        dists = []
        for j in range(n_samples):
            if i==j: continue
            dists.append((j, norm(data.loc[i]-data.loc[j])))    
        
        topk = sorted(dists, key=lambda s: s[1])[:n_neigh]
        neighs = sample(topk, sampling_rate)

        for neigh in neighs:
            alpha = random.random()
            new_samples.append(data.loc[i] + alpha * (data.loc[neigh[0]]-data.loc[i]))
            
    return new_samples


#############################
##Step -1 Pipeline creation for data preparation
#############################
#Class for correcting misspelling of features and target columns
#Due to a problem with the import with adhoc_transf
class ageRounder(BaseEstimator, TransformerMixin):
    def rounder (self,df):
#     #Some fetures content seems to have the character \t.
#     #Let's remove such character for the sake of consistency
         print('\n>>>>>>>>Calling rounder')      
         df['age']=np.around(df.loc[:,'age'])
         return df
     
    def __init__(self):
            print('\n>>>>>>>>Calling init() from ageRounder')
            
    def fit(self, X, y=None):
            print('\n>>>>>>>>Calling fit() from ageRounder')
            return self
     
    def transform(self,X,y=None):
            print('\n>>>>>>>>Calling transform() from ageRounder')        
            df=self.rounder(X)       
            return df
     
    def fit_transform(self, X, y=None,):
            return self.fit(X, y).transform(X, y)
    
#############################
##Step 0 Train-Test splitting
#############################
#Before starting to clean data, lets split train set and data set with stratrification on y=DEATH_EVENT

train_set,test_set=train_test_split(df, test_size=0.3, random_state=42, stratify=df["DEATH_EVENT"])

    
train_set['DEATH_EVENT'].value_counts()
test_set['DEATH_EVENT'].value_counts()
    
#lets back up the split just in case
train_set_copy=train_set.copy()
test_set_copy=test_set.copy()

X_train=train_set_copy.drop('DEATH_EVENT',axis=1)
y_train=train_set_copy['DEATH_EVENT'].copy()

X_test=test_set_copy.drop('DEATH_EVENT',axis=1)
y_test=test_set_copy['DEATH_EVENT'].copy()

#Lets define numeric and category features
numerical_features=['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']
category_features= ['anaemia','high_blood_pressure','diabetes','sex', 'smoking']

all_features = category_features.copy()
all_features.extend(numerical_features)  

n_to_sample = len(train_set[train_set.DEATH_EVENT==0]) - len(train_set[train_set.DEATH_EVENT==1])
new_samples = SMOTE(train_set[train_set["DEATH_EVENT"]==1][all_features],
                    sampling_rate = 1, n_neigh = 50)
n_to_sample1 = len(test_set[test_set.DEATH_EVENT==0]) - len(test_set[test_set.DEATH_EVENT==1])
new_samples1 = SMOTE(test_set[test_set["DEATH_EVENT"]==1][all_features],
                    sampling_rate = 1, n_neigh = 50)

# categorical attributes need to be fixed
for s in new_samples:
   
    train_sm = train_set.append(new_samples)
    train_sm["DEATH_EVENT"].fillna(1, inplace=True)
    
for s1 in new_samples1:
   
    test_sm = test_set.append(new_samples1)
    test_sm["DEATH_EVENT"].fillna(1, inplace=True)

    
X_train=train_sm.drop('DEATH_EVENT',axis=1)
y_train=train_sm['DEATH_EVENT'].copy()

X_test=test_sm.drop('DEATH_EVENT',axis=1)
y_test=test_sm['DEATH_EVENT'].copy()  
    


#############################
##Step 1 Experimentation with feature selection
#############################
#1.1 Filter methods: anova for num feat, chi2 for cat feat, and mutual information for both
#1.1.a ranking for numerical feaures with ANOVA
feature_select.feat_sel_Num_to_Cat(X_train[numerical_features], y_train, 'all')
#***output: time, ejection_fraction, serum_creatinine are the most relevant features
# Feature of feat_sel_Num_to_Cat age: 5.437503
# Feature of feat_sel_Num_to_Cat creatinine_phosphokinase: 3.649016
# Feature of feat_sel_Num_to_Cat ejection_fraction: 18.060764
# Feature of feat_sel_Num_to_Cat platelets: 0.184443
# Feature of feat_sel_Num_to_Cat serum_creatinine: 25.004401
# Feature of feat_sel_Num_to_Cat serum_sodium: 5.237164
# Feature of feat_sel_Num_to_Cat time: 81.909871

#1.1.b ranking for categorical features with chi2
feature_select.feat_sel_Cat_to_Cat_chi2(X_train[category_features], y_train, 'all')
#output:anaemia, high_blood_pressure, smoking are the most relevant features
# Feature of feat_sel_Cat_to_Cat chi2 anaemia: 0.721515
# Feature of feat_sel_Cat_to_Cat chi2 high_blood_pressure: 0.655560
# Feature of feat_sel_Cat_to_Cat chi2 diabetes: 0.069512
# Feature of feat_sel_Cat_to_Cat chi2 sex: 0.100904
# Feature of feat_sel_Cat_to_Cat chi2 smoking: 0.334204

#1.1.c selection for all feaures with mutual information
feature_select.feat_sel_Cat_to_Cat_mutinf(X_train, y_train, 'all')
#output:time, ejection_fraction, serum_creatinine  are the most relevant features
# Feature of feat_sel_Cat_to_Cat mutual info age: 0.041001
# Feature of feat_sel_Cat_to_Cat mutual info anaemia: 0.039651
# Feature of feat_sel_Cat_to_Cat mutual info creatinine_phosphokinase: 0.031781
# Feature of feat_sel_Cat_to_Cat mutual info diabetes: 0.000000
# Feature of feat_sel_Cat_to_Cat mutual info ejection_fraction: 0.050236
# Feature of feat_sel_Cat_to_Cat mutual info high_blood_pressure: 0.002306
# Feature of feat_sel_Cat_to_Cat mutual info platelets: 0.000000
# Feature of feat_sel_Cat_to_Cat mutual info serum_creatinine: 0.133361
# Feature of feat_sel_Cat_to_Cat mutual info serum_sodium: 0.000000
# Feature of feat_sel_Cat_to_Cat mutual info sex: 0.000000
# Feature of feat_sel_Cat_to_Cat mutual info smoking: 0.012959
# Feature of feat_sel_Cat_to_Cat mutual info time: 0.249232

#1.1.d selection for numeric feaures with mutual information
feature_select.feat_sel_Cat_to_Cat_mutinf(X_train[numerical_features], y_train, 'all')
#output: time, ejection_fraction, serum_creatinine are the most relevant

# Feature of feat_sel_Cat_to_Cat mutual info age: 0.000944
# Feature of feat_sel_Cat_to_Cat mutual info creatinine_phosphokinase: 0.027680
# Feature of feat_sel_Cat_to_Cat mutual info ejection_fraction: 0.086096
# Feature of feat_sel_Cat_to_Cat mutual info platelets: 0.000000
# Feature of feat_sel_Cat_to_Cat mutual info serum_creatinine: 0.112365
# Feature of feat_sel_Cat_to_Cat mutual info serum_sodium: 0.014158
# Feature of feat_sel_Cat_to_Cat mutual info time: 0.259273


#1.2 Filter methods: RFE and RFECV for both
#1.2.a RFE 
feature_select.feat_sel_RFE(X_train, y_train, k_out_features='all')
#output
# Optimum number of features: 9
# Score with 9 features: 0.873016
# Num Features: 9
# Selected Features: Index(['age', 'anaemia', 'diabetes', 'ejection_fraction',
#        'high_blood_pressure', 'serum_creatinine', 'serum_sodium','sex', 'smoking'],
#       dtype='object')
# Feature Ranking: [1 1 3 1 1 1 4 1 1 1 1 2]

#1.2.b RFECV
feature_select.feat_sel_RFECV(X_train, y_train)

#output
# Num Features: 10
# Selected Features: Index(['age', 'anaemia', 'diabetes', 'ejection_fraction',
#        'high_blood_pressure', 'serum_creatinine', 'serum_sodium', 'sex',
#        'smoking', 'time'],
#       dtype='object')

# Feature Ranking: [1 1 2 1 1 1 3 1 1 1 1 1]

    
#############################
##Step 2 Pipeline creation for data preparation
#############################
#we could perform two kinds of pipeline: a)a parallel pipeline that manages numerical and categorical features by separate,
# b) a pipeline that first process the numeric features and then process a feature selection and classification of the whole features

#Looking at the values of the different features we can establish
#1. There is no missing values in the entire dataset
#2. All values are numeric type which implies that it can be processed correctly
#3. In age feature there is a no int value 60,667 that should be cast down to 61


###### Pipeline option a. parallel pipeline

pipeline_numeric_feat= Pipeline([('round',ageRounder()),                                 
                                 ('scaler', MinMaxScaler())
                                 ])

pipeline_numbranch=Pipeline([('num_feat',pipeline_numeric_feat),
                             ('feat_sel_num',feature_select.Feature_Selector(strategy='wrapper_RFECV'))
                            ])

pipeline_num_featsel=Pipeline([('feat_sel_num',feature_select.Feature_Selector(strategy='wrapper_RFECV'))])
# pipeline_category_feat= Pipeline(['features_select',feature_select.Feature_Selector(strategy='wrapper_RFECV')
#                         ])

dataprep_parallelpipe=ColumnTransformer([('numeric_pipe',pipeline_numbranch,numerical_features),
                                 ('category_pipe',feature_select.Feature_Selector(strategy='wrapper_RFECV'), category_features)
                                ])


#############################
##Step 3 Classifier initialization
#############################
#Several ensemble classifier with Cross validation will be applied
#we take decision tree as base classifier

dectree_clf=DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=7, random_state=42)
rndforest_clf=RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7,max_features='log2')
extratree_clf=ExtraTreesClassifier(n_estimators=100, random_state=42)
ada_clf= AdaBoostClassifier(n_estimators=100, random_state=42)
gradboost_clf=GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1,max_depth=7, random_state=42)
xgboost_clf= xgb.XGBClassifier(colsample_bytree=0.8, learning_rate=1, max_depth=7,
                     random_state=42, reg_alpha=0.005, subsample=0.8)
svm_clf=SVC(kernel="rbf", random_state=42, probability=True, C = 10)
knn_clf= KNeighborsClassifier(n_neighbors=3)
naive_clf=GaussianNB()




#############################
##Step 4 Scoring initialization
#############################

#Lets define the scoring for the GridSearchCV
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score),
    'specificity': make_scorer(recall_score,pos_label=0),
    'precision':make_scorer(precision_score),
    'f1':make_scorer(f1_score),
    'roc_auc':make_scorer(roc_auc_score),
    'mcc':make_scorer(matthews_corrcoef)    
}

#################################################
####4.1 We will start with the parallel pipeline
full_parallel_pipeline=Pipeline ([('data_prep',dataprep_parallelpipe),
                                  ('clf',dectree_clf)])
full_parallel_pipeline.get_params().keys()

param_grid_fppipe={'clf': [dectree_clf,rndforest_clf,extratree_clf,ada_clf,gradboost_clf,xgboost_clf,svm_clf,knn_clf,naive_clf],
               'data_prep__numeric_pipe__feat_sel_num__k_out_features':[2,3],
               'data_prep__numeric_pipe__feat_sel_num__strategy':['filter_num'],
               'data_prep__category_pipe__k_out_features':['passthrough',1,2],
               'data_prep__category_pipe__strategy':['filter_cat']
               }

#load model to save time of fitting
clf_v41=joblib.load(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\clf_v41.pkl')
clf_v41=GridSearchCV(full_parallel_pipeline,param_grid_fppipe,scoring=scoring,refit='accuracy', cv=5)
clf_v41.fit(X_train,y_train)
clf_v41.best_estimator_
print('Params of best estimator of clf_v41:', clf_v41.best_params_)
#{'clf': XGBClassifier(colsample_bytree=0.8, learning_rate=1, max_depth=7,
 #             random_state=42, reg_alpha=0.005, subsample=0.8), #'data_prep__category_pipe__k_out_features': 2, #'data_prep__category_pipe__strategy': 'filter_cat', #'data_prep__numeric_pipe__feat_sel_num__k_out_features': 3, #'data_prep__numeric_pipe__feat_sel_num__strategy': 'filter_num'}
print('Score of best estimator of clf_v41:', clf_v41.best_score_)


print('Index of best estimator of clf_v41:', clf_v41.best_index_)


df_results_clf_v41=pd.DataFrame(clf_v41.cv_results_)
# create an excel with the cross val resutls
df_results_clf_v41.to_excel(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\clf_v41.xlsx',index=True)

clf_v41.refit
preds = clf_v41.predict(X_test)
np.mean(preds == y_test)
y_pred_41=clf_v41.predict(X_test)
#Saving the model
joblib.dump(clf_v41, r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\clf_v41.pkl', compress=1)



###### Pipeline option b. 
#####Lets try with the sequential pipeline
# dataprep_numericbranch=ColumnTransformer(['numeric_pipe',pipeline_numbranch,numerical_features])
# dataprep_seq_pipe=Pipeline([('num_branch',dataprep_numericbranch),
#                             ('feat_sel',feature_select.Feature_Selector(strategy='wrapper_RFECV'))
#                             ])

full_seq_pipeline=Pipeline([('feat_sel',feature_select.Feature_Selector(strategy='wrapper_RFECV')),
                             ('clf',dectree_clf)
                                ])

full_seq_pipeline.get_params().keys()

param_grid_fseqpipe={'clf': [dectree_clf,rndforest_clf,extratree_clf,ada_clf,gradboost_clf,xgboost_clf,svm_clf,knn_clf,naive_clf],
               'feat_sel__k_out_features':[2,3,4,5,6,7,12],
               'feat_sel__strategy':['filter_mutinf','wrapper_RFE','wrapper_RFECV']
               }

#load model to save time of fitting
clf_v42= joblib.load(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\clf_v42.pkl')

clf_v42=GridSearchCV(full_seq_pipeline,param_grid_fseqpipe,scoring=scoring,refit='accuracy', cv=5)
X_train_seq=X_train.copy()
X_train_seq[numerical_features]=pipeline_numeric_feat.fit_transform(X_train[numerical_features])
clf_v42.fit(X_train_seq,y_train)
clf_v42.best_estimator_
print('Params of best estimator of clf_v42:', clf_v42.best_params_)

#Params of best estimator of clf_v42: {'clf': ExtraTreesClassifier(random_state=42), 'feat_sel__k_out_features': 2, #'feat_sel__strategy': 'filter_mutinf'}
#Score of best estimator of clf_v42: 0.8985714285714286
#Index of best estimator of clf_v42: 42

print('Score of best estimator of clf_v42:', clf_v42.best_score_)


print('Index of best estimator of clf_v42:', clf_v42.best_index_)


df_results_clf_v42=pd.DataFrame(clf_v42.cv_results_)
# create an excel with the cross val resutls
df_results_clf_v42.to_excel(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\clf_v42.xlsx',index=False)

#Fit the best estimator with the test set
clf_v42.refit
X_test_seq=X_test.copy()
X_test_seq[numerical_features]=pipeline_numeric_feat.fit_transform(X_test[numerical_features])

preds = clf_v42.predict(X_test_seq)
np.mean(preds == y_test)
y_pred_42=clf_v42.predict(X_test_seq)
#Saving the model
#Ojo hemos sobreescrito el archivo pkl por error hay que volver a hacer el grid search
joblib.dump(clf_v42, r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\clf_v42.pkl', compress=1)


############################################
####4.3 pipeline without feature selection
dataprep_numericbranch=ColumnTransformer(['numeric_pipe',pipeline_numeric_feat,numerical_features])

X_train_nofeat=X_train.copy()
X_train_nofeat[numerical_features]=pipeline_numeric_feat.fit_transform(X_train[numerical_features])

full_nofeatsel_pipeline=Pipeline([('clf',dectree_clf)])

full_nofeatsel_pipeline.get_params().keys()

param_grid_nofeatselpipe={'clf': [dectree_clf,rndforest_clf,extratree_clf,ada_clf,gradboost_clf,xgboost_clf,svm_clf,knn_clf,naive_clf]
               }

#load model to save time of fitting
clf_v43= joblib.load(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\clf_v43.pkl')

clf_v43=GridSearchCV(full_nofeatsel_pipeline,param_grid_nofeatselpipe,scoring=scoring,refit='accuracy', cv=5)
clf_v43.fit(X_train_nofeat,y_train)
clf_v43.best_estimator_
print('Params of best estimator of clf_v43:', clf_v43.best_params_)


print('Score of best estimator of clf_v43:', clf_v43.best_score_)


print('Index of best estimator of clf_v43:', clf_v43.best_index_)


df_results_clf_v43=pd.DataFrame(clf_v43.cv_results_)
# create an excel with the cross val resutls
df_results_clf_v43.to_excel(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\clf_v43.xlsx',index=False)

#Fit the best estimator with the test set
clf_v43.refit
X_test_nofeat=X_test.copy()
X_test_nofeat[numerical_features]=pipeline_numeric_feat.fit_transform(X_test[numerical_features])

preds = clf_v43.predict(X_test_nofeat)
np.mean(preds == y_test)
y_pred_43=clf_v43.predict(X_test_nofeat)
#Saving the model
#Ojo hemos sobreescrito el archivo pkl por error hay que volver a hacer el grid search
joblib.dump(clf_v43, r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\clf_v43.pkl', compress=1)

#Mergin the results with the test set into one place
overall_test_results={'clf':['clf_v41','clf_v42','clf_v43'],
                  'params':[clf_v41.best_params_,clf_v42.best_params_,clf_v43.best_params_],
                  'accuracy_test':[accuracy_score(y_test, y_pred_41),accuracy_score(y_test, y_pred_42),accuracy_score(y_test, y_pred_43)],
                  'f1_test':[f1_score(y_test, y_pred_41, average='weighted'), f1_score(y_test, y_pred_42, average='weighted'),f1_score(y_test, y_pred_43, average='weighted')],
                  'precision_test':[precision_score(y_test, y_pred_41, average='weighted'),precision_score(y_test, y_pred_42, average='weighted'),precision_score(y_test, y_pred_43, average='weighted')],
                  'recall_test':[recall_score(y_test, y_pred_41, average='weighted'), recall_score(y_test, y_pred_42, average='weighted'),recall_score(y_test, y_pred_43, average='weighted')],
                  'specificity_test':[recall_score(y_test, y_pred_41, pos_label=0),recall_score(y_test, y_pred_42, pos_label=0),recall_score(y_test, y_pred_43, pos_label=0)],

    }
df_overall_test_results_paper=pd.DataFrame(data=overall_test_results)
df_overall_test_results_paper.to_excel(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\df_overall_test_results_paper.xlsx',index=False)




#{'clf': XGBClassifier(colsample_bytree=0.8, learning_rate=1, max_depth=7,
 #             random_state=42, reg_alpha=0.005, subsample=0.8), #'data_prep__category_pipe__k_out_features': 2, #'data_prep__category_pipe__strategy': 'filter_cat', #'data_prep__numeric_pipe__feat_sel_num__k_out_features': 3, #'data_prep__numeric_pipe__feat_sel_num__strategy': 'filter_num'}
win_model=clf_v41.best_estimator_

#The features selected in the win_model were:
#anaemia, time, ejection_fraction, serum_creatinine ,high_blood_pressure


X_train_feat=X_train[['anaemia','ejection_fraction','serum_creatinine','time','high_blood_pressure']].copy()
X_test_feat=X_test[['anaemia','ejection_fraction','serum_creatinine','time','high_blood_pressure']].copy()

#A Decision tree with the max_depth of XGBoost (equals to 3) and the feature selection results will be made
#to show it graphically
num_feat_sel=['ejection_fraction','serum_creatinine','time']
minmaxtrain=MinMaxScaler()
minmaxtest=MinMaxScaler()

X_train_feat[num_feat_sel]=minmaxtrain.fit_transform(X_train_feat[num_feat_sel])
X_test_feat[num_feat_sel]=minmaxtest.fit_transform(X_test_feat[num_feat_sel])

X_train_feat.head()

#####################
#Building the DT
clf_tree_feat=DecisionTreeClassifier(max_depth=3, random_state=42,criterion='gini', splitter='best')
#use grid_search_CV to see the results
clf_tree_feat.get_params().keys()

param_grid_clf_tree_feat={'max_depth': [3,None]}

clf_tree_feat_grid=GridSearchCV(clf_tree_feat,param_grid_clf_tree_feat,scoring=scoring,refit='accuracy', cv=5)
clf_tree_feat_grid.fit(X_train_feat,y_train)
clf_tree_feat_grid.best_estimator_
df_results_clf_tree_feat=pd.DataFrame(clf_tree_feat_grid.cv_results_)
# create an excel with the cross val resutls
df_results_clf_tree_feat.to_excel(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\clf_tree_feat.xlsx',index=False)
#Fit the best estimator with the test set
clf_tree_feat_grid.refit

preds = clf_tree_feat_grid.predict(X_test_feat)
np.mean(preds == y_test)#0.7444444444444445
y_pred_tree=clf_tree_feat_grid.predict(X_test_feat)

overall_test_results_clf_tree_feat={'clf':['clf_tree_feat'],
                 'params':[clf_tree_feat_grid.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_tree)],
                 'f1_test':[f1_score(y_test, y_pred_tree, average='weighted')],
                 'precision_test':[precision_score(y_test, y_pred_tree, average='weighted')],
                 'recall_test':[recall_score(y_test, y_pred_tree, average='weighted')],
                 'specificity_test':[recall_score(y_test, y_pred_tree, pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_tree)],
                 'mcc_test':[matthews_corrcoef(y_test, y_pred_tree)]
    }
#Saving results with test set to calculate interpretability measures
df_overall_test_results_clf_tree_feat=pd.DataFrame(data=overall_test_results_clf_tree_feat)
df_overall_test_results_clf_tree_feat.to_excel(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\overall_test_results_clf_tree_feat.xlsx',index=False)

##############################
#Lets print the decision tree
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(clf_tree_feat_grid.best_estimator_, 
                  feature_names=X_train_feat.columns,  
                  class_names=y_train.values.astype('str'),  
                  filled=True, rounded=True,  
                  special_characters=True,
                   out_file=None,
                           )
graph = graphviz.Source(dot_data)
graph.format = "png"
graph.render(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\dt_featsel_graph')

#the decisiontree shows cut point in the branches with the following values
#time=0.261, 0.735
#ejection_fraction=0.367, 0.192, 0.885
#serum creatinine=0.104,0.046
#To improve readability of the graph lets transform these values to their original ones
#Inverting time
X_train_feat_time=X_train[['time']].copy()
minmaxtrain_time=MinMaxScaler()
X_train_feat_time=minmaxtrain_time.fit_transform(X_train_feat_time)
time_1=minmaxtrain_time.inverse_transform(np.array(0.261).reshape(1,-1))
time_2=minmaxtrain_time.inverse_transform(np.array(0.735).reshape(1,-1))
#Inverting ejection_fraction
X_train_feat_ef=X_train[['ejection_fraction']].copy()
minmaxtrain_ef=MinMaxScaler()
X_train_feat_ef=minmaxtrain_ef.fit_transform(X_train_feat_ef)
ef_1=minmaxtrain_ef.inverse_transform(np.array(0.367).reshape(1,-1))
ef_2=minmaxtrain_ef.inverse_transform(np.array(0.192).reshape(1,-1))
ef_3=minmaxtrain_ef.inverse_transform(np.array(0.885).reshape(1,-1))
#Inverting serum_creatinine
X_train_feat_sc=X_train[['serum_creatinine']].copy()
minmaxtrain_sc=MinMaxScaler()
X_train_feat_sc=minmaxtrain_sc.fit_transform(X_train_feat_sc)
sc_1=minmaxtrain_sc.inverse_transform(np.array(0.104).reshape(1,-1))
sc_2=minmaxtrain_sc.inverse_transform(np.array(0.046).reshape(1,-1))

print ('time_1: ',time_1,'; time_2: ',time_2,'; ef_1: ',ef_1,'; ef_2: ',ef_2,'; ef_3: ',ef_3, ';sc_1: ',sc_1, ';sc_2: ' , sc_2)

############
#Update, the best estimator is not performed by removing the minmax scaler so the following part is deprecated.
#The plot and decision rules shown are affected by the minmax_scaler
#we perform the same without the scaling and check the classification metrics
X_train_feat=X_train[['anaemia','ejection_fraction','serum_creatinine','time','high_blood_pressure']].copy()
X_test_feat=X_test[['anaemia','ejection_fraction','serum_creatinine','time','high_blood_pressure']].copy()

#Building the DT
clf_tree_feat_nomin=DecisionTreeClassifier(max_depth=3, random_state=42)
#use grid_search_CV to see the results
clf_tree_feat.get_params().keys()

param_grid_clf_tree_feat_nomin={'max_depth': [3,None]}

clf_tree_feat_grid_nomin=GridSearchCV(clf_tree_feat_nomin,param_grid_clf_tree_feat_nomin,scoring=scoring,refit='accuracy', cv=5)
clf_tree_feat_grid_nomin.fit(X_train_feat,y_train)
clf_tree_feat_grid_nomin.best_estimator_
df_results_clf_tree_feat_nomin=pd.DataFrame(clf_tree_feat_grid_nomin.cv_results_)
# create an excel with the cross val resutls
df_results_clf_tree_feat_nomin.to_excel(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\clf_tree_feat_nomin.xlsx',index=False)
#Fit the best estimator with the test set
clf_tree_feat_grid_nomin.refit

preds = clf_tree_feat_grid_nomin.predict(X_test_feat)
print(np.mean(preds == y_test))#0.7444444444444445
y_pred_tree_nomin=clf_tree_feat_grid_nomin.predict(X_test_feat)

overall_test_results_clf_tree_feat_nomin={'clf':['clf_tree_feat_nomin'],
                 'params':[clf_tree_feat_grid_nomin.best_params_],
                 'accuracy_test':[accuracy_score(y_test, y_pred_tree_nomin)],
                 'f1_test':[f1_score(y_test, y_pred_tree_nomin, average='weighted')],
                 'precision_test':[precision_score(y_test, y_pred_tree_nomin, average='weighted')],
                 'recall_test':[recall_score(y_test, y_pred_tree_nomin, average='weighted')],
                 'specificity_test':[recall_score(y_test, y_pred_tree_nomin, pos_label=0)],
                 'roc_auc_test':[roc_auc_score(y_test, y_pred_tree_nomin)],
                 'mcc_test':[matthews_corrcoef(y_test, y_pred_tree)]
    }
#Saving results with test set to calculate interpretability measures
df_overall_test_results_clf_tree_feat_nomin=pd.DataFrame(data=overall_test_results_clf_tree_feat_nomin)
df_overall_test_results_clf_tree_feat_nomin.to_excel(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\overall_test_results_clf_tree_feat_nomin.xlsx',index=False)


import graphviz
dot_data = tree.export_graphviz(clf_tree_feat_grid_nomin.best_estimator_, 
                  feature_names=X_train_feat.columns,  
                  class_names=y_train.values.astype('str'),  
                  filled=True, rounded=True,  
                  special_characters=True,
                   out_file=None,
                           )
graph = graphviz.Source(dot_data)
graph.format = "png"
graph.render(r'C:\Users\mariasiouzou\Documents\ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ\SSIA\GridSearchCV_results\dt_featsel_nomin_graph')


    
    
