# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 09:18:54 2019

@author: B076578
"""
import pandas as pd
import openpyxl

def change_feature_sheet(feature_list,sheet_name,file_name,number,new_sheet_name):
    df=pd.read_excel(file_name, sheet_name)
    df=df.set_index('Variable_Name')
    for i in feature_list:
        df.loc[i,'Scenario']=number   
    writer = pd.ExcelWriter(file_name, engine='openpyxl')
    book = openpyxl.load_workbook(file_name)
    writer.book = book
    df.to_excel(writer, sheet_name=new_sheet_name)
    writer.save()
    writer.close()
    return df
    
file_name='//ijmrdtappl00/PT_Blast_Furnace/Wave1_BlastFurnaces/BlastFurnace_5/VariableSelection/Active/VariableReductionBF5.xlsx'
sheet_name='Base'
feature_list=['slag_MgO',
 'slag_Al2O3',
 'hearth_pad_temp',
 'bf5_hot_blast_temp',
 'rotofeed_c_calc_calib',
 'sinter_perc_tio2',
 'flxd_perc_mgo',
 'acid_perc_cao',
 'coal_volatiles',
 'coke_moisture',
 'ferrous_k2o_load'] 

temp=change_feature_sheet(feature_list,sheet_name,file_name,0,"noisyFeaturesRemoved")
print(temp)