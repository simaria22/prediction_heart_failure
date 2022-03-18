
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import statsmodels.api as sm


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR 
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.base import BaseEstimator, TransformerMixin

################
##Filter methods
################
#Function to take a df with num attributes and cat target and return df with k_out_features

def feat_sel_Num_to_Cat(X, y, k_out_features):
    fs=SelectKBest(score_func=f_classif, k=k_out_features)
    df_sel=fs.fit_transform(X, y)
    if k_out_features=='all':
        for i in range(len(fs.scores_)):
            print('Feature of feat_sel_Num_to_Cat %s: %f' % (X.columns[i], fs.scores_[i]))
    #we have to create a dataframe
    cols=fs.get_support(indices=True)
    df_sel=X.iloc[:,cols]
    return df_sel

#Function to take a df with cat attributes and cat target and return df with k_out_features
def feat_sel_Cat_to_Cat_chi2(X, y, k_out_features):
    #chi-squared feature selection
    fs_chi2=SelectKBest(score_func=chi2, k=k_out_features)
    df_chi2=fs_chi2.fit_transform(X,y)
    if k_out_features=='all':
        for i in range(len(fs_chi2.scores_)):
            print('Feature of feat_sel_Cat_to_Cat chi2 %s: %f' % (X.columns[i], fs_chi2.scores_[i]))
    #we have to create a dataframe
    cols_chi2=fs_chi2.get_support(indices=True)
    df_chi2=X.iloc[:,cols_chi2]
        
    return df_chi2

def feat_sel_Cat_to_Cat_mutinf(X, y, k_out_features):
   
    #Mutual information feature selection
    fs_mutinf=SelectKBest(score_func=mutual_info_classif, k=k_out_features)
    df_mutinf=fs_mutinf.fit_transform(X,y)
    if k_out_features=='all':
        for i in range(len(fs_mutinf.scores_)):
            print('Feature of feat_sel_Cat_to_Cat mutual info %s: %f' % (X.columns[i], fs_mutinf.scores_[i]))
    cols_mutinf=fs_mutinf.get_support(indices=True)
    df_mutinf=X.iloc[:,cols_mutinf]
    return df_mutinf

#################
##Wrapper methods
#################

#RFE method with logistic regression or other specified estimator
def feat_sel_RFE(X,y,k_out_features=None, estimator='LogisticRegression'):
        
    #allows different kind of estimators
    if estimator=='LogisticRegression':
        model=LogisticRegression(solver='lbfgs', max_iter=2000)
    if estimator=='SVR':
        model=SVR(kernel='linear')
    
    #check the optimus number of output features for which the accuracy is highest
    #if k_out_features==None by default the number of output features is the half of total
    if k_out_features=='all':
        nof_list=np.arange(1,X.shape[1])            
        high_score=0
        #Variable to store the optimum features
        nof=0           
        score_list =[]
        from sklearn.model_selection import train_test_split
        for n in range(len(nof_list)):
            rfe = RFE(model,n_features_to_select=nof_list[n])
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
            X_train_rfe = rfe.fit_transform(X_train,y_train)
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe,y_train)
            score = model.score(X_test_rfe,y_test)
            score_list.append(score)
            if(score>high_score):
                high_score = score
                nof = nof_list[n]
        print("Optimum number of features: %d" %nof)
        print("Score with %d features: %f" % (nof, high_score))
        k_out_features=nof
                
    #obtain the pruned resultant df of features
    rfe=RFE(model,n_features_to_select=k_out_features)
    fit = rfe.fit(X, y)
    X_pruned=rfe.fit_transform(X,y)
    mask=fit.support_
    X_pruned=X.iloc[:,mask]
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    return X_pruned
        
#RFECV with the LogisticRegression estimator as default
def feat_sel_RFECV(X,y, estimator="LogisticRegression"):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    if estimator=='LogisticRegression':
        model=LogisticRegression(solver='lbfgs', max_iter=2000)
    if estimator=='SVR':
        model=SVR(kernel='linear')
    rfe=RFECV(model)    
    fit=rfe.fit(X, y)
    X_pruned=rfe.transform(X)
    mask=fit.support_
    X_pruned=X.iloc[:,mask]
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    return X_pruned


        
        
#Class to run feature selection depending on different strategies
class Feature_Selector(BaseEstimator, TransformerMixin):
    #filter_num: performing ANOVA valid for numeric input and category output
    #filter_cat: performing chi2 valid for category input and category output
    #filter_mutinf:performing mutual information valid for numeric/category input and category output
    #wrapper_RFECV: performing RFECV with two optional regressor LogisticRegression(by defaurl) or SVR valid for numeric/category input and category output

    
    #def __init__(self,y_train,strategy='wrapper_RFECV',k_out_features=5, rfe_estimator='LogisticRegression'):       
    def __init__(self,strategy='wrapper_RFECV',k_out_features=5, rfe_estimator='LogisticRegression'):
        print('\n>>>>>>>>Calling init() from Feature_Selector')
        
        #self.y_train=y_train
        self.strategy=strategy
        self.k_out_features=k_out_features
        self.rfe_estimator=rfe_estimator
        
        if self.strategy=='filter_num':
            self.feat_sel=SelectKBest(score_func=f_classif, k=self.k_out_features)
            
        if self.strategy=='filter_cat':
            self.feat_sel=SelectKBest(score_func=chi2, k=self.k_out_features)
            
        if self.strategy=='filter_mutinf':
            self.feat_sel=SelectKBest(score_func=mutual_info_classif, k=self.k_out_features)
            
        if self.strategy=='wrapper_RFECV':
            if self.rfe_estimator=='LogisticRegression':
                self.model=LogisticRegression(solver='lbfgs', max_iter=2000)
            if self.rfe_estimator=='SVR':
                self.model=SVR(kernel='linear')
            self.feat_sel=RFECV(self.model)
        
        if self.strategy=='wrapper_RFE':
            if self.rfe_estimator=='LogisticRegression':
                self.model=LogisticRegression(solver='lbfgs', max_iter=2000)
            if self.rfe_estimator=='SVR':
                self.model=SVR(kernel='linear')
            self.feat_sel=RFE(self.model, n_features_to_select=k_out_features)
        
        
        
    def fit(self,X,y=None):
        print('\n>>>>>>>>Calling fit() from Feature_Selector')
        #index=X.index
        self.y_train=y
        #print('\n********Inside fit() from Feature_Selector y_train length:', self.y_train.size)        
        #print('\n********Calling fit() from Feature_Selector X length: ', X.shape[0])
        
        self.feat_sel.fit(X,self.y_train)
        return self
    
    def transform(self,X,y=None,):
        print('\n>>>>>>>>Calling transform() from Feature_Selector')
        X_pruned=self.feat_sel.transform(X)
        return X_pruned
        
            
        
            
        
        
            