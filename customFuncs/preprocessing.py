import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

class preprocess:
    X=None
    X_stan=None
    X_norm=None
    Y=None
    
    dateVariable=None
    timeMin=None
    timeMax=None
      
    def __init__(self, df,target_variable):
        self.df=df
        self.target_variable=target_variable
    
    def remove_missing_values(self,TV):
        df=self.df
        df = df[df.columns[df.isnull().mean() < TV]] ### removes column if missing a lot
        self.df=df.fillna(df.mean())   ### replaces nans with mean of column
        
		
    def assign_time_vars(self,dateVariable,timeMin,timeMax):
        self.dateVariable=dateVariable
        self.timeMin=timeMin
        self.timeMax=timeMax
        
    def filter_by_time(self):
        df=self.df
        df[self.dateVariable] = pd.to_datetime(df[self.dateVariable])
        df = df[df[self.dateVariable].dt.year > self.timeMin]
        df = df[df[self.dateVariable].dt.year < self.timeMax]
        self.df= df.drop([self.dateVariable], axis=1)
        
    def stages_func(self,stages, value_list):
        df=self.df
        to_remove = stages[stages.Scenario.isin(value_list)]
        variable_names_remove = to_remove['Variable_Name'].values    
        var_list_len = len(variable_names_remove)    
        for i in range (0, var_list_len):
            temp = variable_names_remove[i]
            df = df.drop([temp], axis = 1)        
        self.df=df
    
    def assign_X_and_Y(self):
        target_variable=self.target_variable
        df=self.df
        self.Y = df[target_variable].values
        self.X = df.drop([target_variable], axis=1).values
        self.X_stan = StandardScaler().fit(self.X).transform(self.X)
        self.X_norm = Normalizer().fit(self.X).transform(self.X)
