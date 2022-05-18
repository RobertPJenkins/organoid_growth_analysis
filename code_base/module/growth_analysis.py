"""# growth_analysis

## csv_reader(...).

## rate_fitter(...).

## stats_function(...).

## stats_output(...).

## csv_writer(...).

"""

from typing import Dict, List, Tuple

import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

model = LinearRegression()

def csv_reader(
    csv_input_directory: str,
    input_file_type: str,
):

    """read tabulated data that contains quantitative features
    

    Parameters
    ----------
    csv_input_directory : str
        path to tabulated data
    input_file_type : str
        file type of tabulated data
    Returns
    -------
    pd.DataFrame
        a data frame with column names ["TRACK_ID", "FRAME_TIME", "AREA_SCALED",
        "EXPERIMENT_DATE","EXPERIMENT_CONDITION","GLOBAL_TRACK_ID"]. These columns 
        refer to...

    """
    csv_input_directory = csv_input_directory.replace('/',os.sep)
    csv_input_directory = csv_input_directory.replace('\\',os.sep)
    
    file_target = os.path.join(
        csv_input_directory,
        '*'+input_file_type
    )
    
    spreadsheet_files = glob.glob(file_target)
    df_all_exps=pd.DataFrame(columns=[
        'TRACK_ID',
        'FRAME_TIME',
        'AREA_SCALED',
        'EXPERIMENT_DATE',
        'EXPERIMENT_CONDITION',
        'GLOBAL_TRACK_ID'
    ]) 
    for spread_sheet_file in spreadsheet_files:
        base_name = os.path.basename(spread_sheet_file)
        file_name, file_extension = os.path.splitext(base_name)
        df = pd.read_csv(spread_sheet_file)
        df = df[['TRACK_ID','FRAME_TIME','AREA_SCALED']]
    
        max_unique = np.max(df_all_exps.GLOBAL_TRACK_ID)
        if np.isnan(max_unique):
            max_unique=0;
    
        Values, inverseList = np.unique(df.TRACK_ID, return_inverse=True)
        normalised_unique = np.arange(0,np.shape(Values)[0]) + max_unique
        df = df.assign(GLOBAL_TRACK_ID = normalised_unique[inverseList])
        df=df.assign(EXPERIMENT_DATE=file_name[0:8])
        df=df.assign(EXPERIMENT_CONDITION=file_name[9:])
        df_all_exps = pd.concat([df_all_exps,df])       
    return df_all_exps


def rate_fitter(df_in):
    df_out=pd.DataFrame(columns=[
        'GLOBAL_TRACK_ID',
        'TRACK_ID',
        'EXPERIMENT_DATE',
        'EXPERIMENT_CONDITION',
        'INITIAL_VOLUME',
        'NORMALISED_VOLUME',
        'R_SQUARED',
        'CONSTANT',
        'RATE'
    ]) 
    values = np.unique(df_in.GLOBAL_TRACK_ID)
    for value in values:
        df=pd.DataFrame(columns=df_out.columns)
        
        index = np.where(df_in.GLOBAL_TRACK_ID == value)
        time = np.array(df_in.iloc[index].FRAME_TIME).reshape(-1,1)
        volume = np.array(df_in.iloc[index].AREA_SCALED).reshape(-1,1)

        model.fit(time, np.log(volume))
        df = pd.DataFrame({
            'GLOBAL_TRACK_ID': [df_in.iloc[index[0][0]].GLOBAL_TRACK_ID],
            'TRACK_ID' : [df_in.iloc[index[0][0]].TRACK_ID],
            'EXPERIMENT_DATE' : [df_in.iloc[index[0][0]].EXPERIMENT_DATE],
            'EXPERIMENT_CONDITION' : [df_in.iloc[index[0][0]].EXPERIMENT_CONDITION],
            'INITIAL_VOLUME' : volume[0],
            'NORMALISED_VOLUME': volume[-1,0]/np.mean(volume[0:5,0]),
            'R_SQUARED' : [model.score(time, np.log(volume))],
            'CONSTANT' : [np.exp(model.intercept_)[0]],
            'RATE' : [model.coef_[0][0]]
        })
        df_out = pd.concat([df_out,df])
        df_out.reset_index(drop=True, inplace=True)
    return df_out

def stats_function(date,experiment,rate,constant,r_squared):
    df = pd.DataFrame({
        'EXPERIMENT_DATE' : [date],
        'EXPERIMENT_CONDITION' : [experiment],
        'MIN_RATE' : [rate.min()[0]],
        'MAX_RATE' : [rate.max()[0]],
        'MEAN_RATE' : [rate.mean()[0]],
        'STD_RATE' : [rate.std()[0]],
        'LQ_RATE' : [rate.quantile(0.25)[0]],
        'MEDIAN_RATE' : [rate.median()[0]],
        'UQ_RATE' : [rate.quantile(0.75)[0]],
        'MIN_CONSTANT' : [constant.min()[0]],
        'MAX_CONSTANT' : [constant.max()[0]],
        'MEAN_CONSTANT' : [constant.mean()[0]],
        'STD_CONSTANT' : [constant.std()[0]],
        'LQ_CONSTANT' : [constant.quantile(0.25)[0]],
        'MEDIAN_CONSTANT' : [constant.median()[0]],
        'UQ_CONSTANT' : [constant.quantile(0.75)[0]],
        'MIN_R_SQUARED' : [r_squared.min()[0]],
        'MAX_R_SQUARED' : [r_squared.max()[0]],
        'MEAN_R_SQUARED' : [r_squared.mean()[0]],
        'STD_R_SQUARED' : [r_squared.std()[0]],
        'LQ_R_SQUARED' : [r_squared.quantile(0.25)[0]],
        'MEDIAN_R_SQUARED' : [r_squared.median()[0]],
        'UQ_R_SQUARED' : [r_squared.quantile(0.75)[0]]
    })
    return df


def stats_output(df_in):
    df_out=pd.DataFrame(columns=[
        'EXPERIMENT_DATE',
        'EXPERIMENT_CONDITION',
        'MIN_RATE',
        'MAX_RATE',
        'MEAN_RATE',
        'STD_RATE',
        'LQ_RATE',
        'MEDIAN_RATE',
        'UQ_RATE',
        'MIN_CONSTANT',
        'MAX_CONSTANT',
        'MEAN_CONSTANT',
        'STD_CONSTANT',
        'LQ_CONSTANT',
        'MEDIAN_CONSTANT',
        'UQ_CONSTANT',
        'MIN_R_SQUARED',
        'MAX_R_SQUARED',
        'MEAN_R_SQUARED',
        'STD_R_SQUARED',
        'LQ_R_SQUARED',
        'MEDIAN_R_SQUARED',
        'UQ_R_SQUARED'
    ]) 
    kdes_rates = []
    kdes_constants = []
    kdes_rsquareds = []
    condition = []
    date = []

    unique_conditions = pd.unique(df_in.EXPERIMENT_CONDITION)
    for unique_condition in unique_conditions:
        df_condition = df_in.loc[df_in.EXPERIMENT_CONDITION == unique_condition]
        rate = df_condition[['RATE']]
        constant = df_condition[['CONSTANT']]
        r_squared = df_condition[['R_SQUARED']]
        
        
        kdes_rates.append(stats.gaussian_kde(np.array(rate).T))
        kdes_constants.append(stats.gaussian_kde(np.array(constant).T))
        kdes_rsquareds.append(stats.gaussian_kde(np.array(r_squared).T))
        
        df = stats_function('POOLED',unique_condition,rate,constant,r_squared)
        df_out = pd.concat([df_out,df])
        condition.append(unique_condition)
        date.append('GLOBAL')
        unique_dates = pd.unique(df_condition['EXPERIMENT_DATE'])
        for unique_date in unique_dates:
            
            rate = df_condition[['RATE']].loc[df_condition.EXPERIMENT_DATE == unique_date]
            constant = df_condition[['CONSTANT']].loc[df_condition.EXPERIMENT_DATE == unique_date]
            r_squared = df_condition[['R_SQUARED']].loc[df_condition.EXPERIMENT_DATE == unique_date]
            df = stats_function(unique_date,unique_condition,rate,constant,r_squared)
            df_out = pd.concat([df_out,df])
            kdes_rates.append(stats.gaussian_kde(np.array(rate).T))
            kdes_constants.append(stats.gaussian_kde(np.array(constant).T))
            kdes_rsquareds.append(stats.gaussian_kde(np.array(r_squared).T))
            condition.append(unique_condition)
            date.append(unique_date)
    df_out.reset_index(drop=True, inplace=True)
    return df_out, kdes_rates, kdes_constants, kdes_rsquareds, condition, date

def csv_writer(
    csv_output_directory: str,
    rate_file_name: str,
    statistics_file_name: str,
    df_rate,
    df_statistics,
):
    csv_output_directory = csv_output_directory.replace('/',os.sep)
    csv_output_directory = csv_output_directory.replace('\\',os.sep)
    
    sub_directory = os.path.join(
        csv_output_directory,
        'output_data'
    )

    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)
    
    rate_file_out = os.path.join(
        sub_directory,
        rate_file_name + '.csv'
    )
    
    statistics_file_out = os.path.join(
        sub_directory,
        statistics_file_name + '.csv'
    )
    rate_files_present = glob.glob(rate_file_out)
    statistics_files_present = glob.glob(statistics_file_out)
    
    df_rate.to_csv(rate_file_out,index=False)
    df_statistics.to_csv(statistics_file_out,index=False) 
                    