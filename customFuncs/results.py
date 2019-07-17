# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:05:01 2019

@author: B076578
"""
from customFuncs.scoring import trainTestKCross
import pickle
import pandas as pd
import os 

def printScoresTable(folderPath,X,Y):
    fileList=os.listdir(folderPath)
    headers=["Model",'Train R2', "Test R2", "Delta R2", "Train RMSE", "Test RMSE", 'Delta RMSE']
    rowList=[]
    for file in fileList:
        print(file.split(".sav")[0])
        model=pickle.load( open( folderPath+ "\\" + file , "rb" ) )
        row=trainTestKCross(model,X,Y)
        rowList.append([file.split(".sav")[0]]+row)
    df=pd.DataFrame(rowList)
    df.columns=headers
    df= df.T
    df = df[df.columns.dropna()]
    df.to_csv("allModelResults.csv",index =True,header=False)
    return df
