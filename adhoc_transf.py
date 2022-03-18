
#This class must include as much as function needed depeding on the 
#mispelling or wrong character detected in the dataset
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

#Class for correcting misspelling of features and target columns
class ageRounder(BaseEstimator, TransformerMixin):
    def rounder (self,df):
    #Some fetures content seems to have the character \t.
    #Let's remove such character for the sake of consistency
        #print('\n>>>>>>>>Calling rounder')      
        df['age']=np.around(df['age'])
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
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)



class ageRounder(BaseEstimator, TransformerMixin):
    def rounder (self,df):
    #Some fetures content seems to have the character \t.
    #Let's remove such character for the sake of consistency
        print('\n>>>>>>>>Calling rounder')      
        df['age']=np.around(df.loc[:,'age'])
        return df
    
    def __init__(self):
        print('\n>>>>>>>>Calling init() from ageRounder')
            
    def fit(self,X,y=None):
        print('\n>>>>>>>>Calling fit() from ageRounder')
        return self
    
    def transform(self,X,y=None):
        print('\n>>>>>>>>Calling transform() from ageRounder')        
        df=self.rounder(X)       
        return df
    
    def fit_transform(self,X,y=None):
        return self.fit(X, y).transform(X, y)