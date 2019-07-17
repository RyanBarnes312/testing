# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:41:41 2019

@author: B076578
"""
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import statistics 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


def getAdjR2(x1,y1,model,target_variable):
    accuracies = cross_val_score(estimator = model, X= x1, y= y1, cv = 10)
    print(accuracies)
    R2=accuracies.mean()
    print(R2)
    n=x1.shape[0]
    p=x1.shape[1]
    adjR2 = 1-(1-R2)*(n-1)/(n-p-1)
    print(adjR2)
    return adjR2

def trainTestKCross(model,X,Y,ES=False,R=10):
    cv = KFold(n_splits=10, shuffle=True)
    trainR2List=[]
    testR2List=[]
    trainMSEList=[]
    testMSEList=[]
    for train_index, test_index in cv.split(X):
        x1, x2, y1, y2 = X[train_index], X[test_index], Y[train_index], Y[test_index]
        if ES==False:
            model.fit(x1, y1)
        else:
            eval_set = [(x2, y2)]
            model.fit(x1, y1,early_stopping_rounds=R, eval_metric="rmse", eval_set=eval_set, verbose=False)
            
        train_pred = model.predict(x1)
        trainR2=r2_score(y1, train_pred) 
        trainRMSE=(mean_squared_error(y1, train_pred))**(1/2)
        
        
        test_pred = model.predict(x2)
        testR2=r2_score(y2, test_pred) 
        testRMSE=(mean_squared_error(y2, test_pred))**(1/2)
        
        trainR2List.append(trainR2)
        testR2List.append(testR2)
        trainMSEList.append(trainRMSE)
        testMSEList.append(testRMSE)
    
    trainKCross=statistics.mean(trainR2List)
    testKCross=statistics.mean(testR2List)
    trainRMSE=statistics.mean(trainMSEList)
    testRMSE=statistics.mean(testMSEList)
    delta=abs(trainKCross-testKCross)
    delta2=abs(trainRMSE-testRMSE)
    print(trainKCross,testKCross,delta,trainRMSE,testRMSE,delta2)
    return [trainKCross,testKCross,delta,trainRMSE,testRMSE,delta2]

def KCrossFeatureImportance(model,X,Y,df,target_variable):
    cv = KFold(n_splits=10, shuffle=True)
    tempdf = pd.DataFrame() 
    tempdf["Row ID"] =list(df.drop([target_variable], axis=1).columns.values)
    temp=[0]*(X.shape[1])
    
    for train_index, test_index in cv.split(X):
        x1, x2, y1, y2 = X[train_index], X[test_index], Y[train_index], Y[test_index]
        model.fit(x1, y1)
        FI=list(model.feature_importances_)
        for count, feature in enumerate(FI):
            temp[count]=(temp[count]+feature)/2        
    tempdf["Variable Importance"] = temp
    tempdf=tempdf.sort_values(by='Variable Importance',ascending=False)
    tempdf=tempdf.reset_index(drop=True)            

    return tempdf

def reduceFI(df,target_variable,model,method,n=20,sigValue=0.05):
    y = df[target_variable].values
    x = df.drop([target_variable], axis=1).values
    X, X_none, Y, y_none = train_test_split(x, y, test_size = 0,shuffle=True)
    feats=KCrossFeatureImportance(model,X,Y,df,target_variable)
    if method=="top_n":
        feats=feats.head(n)
        features=list(feats['Row ID'].values)
        rdf=df[features+[target_variable]]
    elif method=="sigValue":
        feats=feats[feats['Variable Importance']  >= sigValue]
        features=list(feats['Row ID'].values)
        rdf=df[features+[target_variable]]
    return rdf



def saveModelResults(model,X,Y,filename):
    print("Doing "+filename)
    trainTestKCross(model,X,Y)
    pickle.dump(model, open(filename, 'wb'))