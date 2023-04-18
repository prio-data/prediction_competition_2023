#!/usr/bin/env python
# coding: utf-8

# # Benchmark model generation

# Imports
## Basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import os
from functools import partial

## Views 3
import views_runs
from viewser.operations import fetch
from views_forecasts.extensions import *
from viewser import Queryset, Column


# Auxilliary functions

## Extract sc predictions for a given calendar year

def extract_sc_predictions(year,ss_predictions):
    ''' Extract sc predictions if necessary, split into calendar years '''
    first_month = (year - 1980)*12 + 1
    months=list(range(first_month,first_month+12))
    df = ss_predictions.loc[months].copy()
    if 'prediction' in ss_predictions:
        df = pd.DataFrame(df['prediction'])
    else:
        df['prediction'] = 0
        for month in range(1,12+1):
            this_month = first_month + month - 1
            df_temp = df.loc[this_month]
            this_col = 'step_pred_' + str(month+2)

            df_temp['prediction'] = np.expm1(df_temp[this_col].values) 
            df['prediction'].loc[this_month] = df_temp['prediction'].values
    #        print(month, this_col, this_month)
    #        print(df_temp[this_col].values)
    return pd.DataFrame(df['prediction'])


def extract_year(year, df):
    """
    Extract data from input dataframe in year-chunks
    """
    first_month = (year - 1980)*12 + 1
    last_month = first_month + 11
    return pd.DataFrame(df.loc[first_month:last_month])


def describe_expanded(df, df_expanded, month, country):
    # Verify that the distribution is right
    this_month = 457
    this_country = 57
    print("Mean and std of original predictions, all rows:")
    print(df.describe())
    print("Mean and std of expanded predictions, all rows:")
    print(df_expanded.describe())
    print("Mean and std of original predictions, one cm:")
    print(df.loc[this_month,this_country].describe())
    print("Mean and std of expanded predictions, one cm:")
    print(df_expanded.loc[this_month,this_country].describe())
    print("Variance:",df_expanded.loc[this_month,this_country].var())


def sample_poisson_row(row: pd.DataFrame, ndraws:int = 100) -> np.ndarray:
    """Given a dataframe row, produce ndraws poisson draws from the prediction column in the row.
    Attention, this is a row vectorized approach, should be used with apply.
    :return an np array. Should be exploded for long df.
    """
    row.prediction = 0 if row.prediction <= 0 else row.prediction
    return np.random.poisson(row.prediction, size=ndraws)


def sample_uniform_row(row: pd.DataFrame, ndraws:int = 100) -> np.ndarray:
    """Given a dataframe row, produce ndraws poisson draws from the prediction column in the row.
    Attention, this is a row vectorized approach, should be used with apply.
    :return an np array. Should be exploded for long df.
    """
    row.prediction = 0 if row.prediction <= 0 else row.prediction
    return np.random.uniform(low=row.prediction, high=row.prediction, size=ndraws)

def sample_bootstrap_row(row: pd.DataFrame, draw_from: np.array, ndraws: int = 100) -> np.ndarray:
    """Given a dataframe row, produce ndraws draws from the prediction column in the df.
    Attention, this is a row vectorized approach, should be used with apply.
    :return an np array. Should be exploded for long df.
    """
    return np.random.choice(draw_from, size=ndraws, replace=True)


def expanded_df_distribution(df, ndraws=1000, level='cm', distribution = 'poisson'):
    if distribution == 'poisson':
        function_with_draws = partial(sample_poisson_row, ndraws=ndraws)
    if distribution == 'uniform':
        function_with_draws = partial(sample_uniform_row, ndraws=ndraws)
    df['draws'] = df.apply(function_with_draws, axis=1)
    exploded_df = df.explode('draws').astype('int32')
    if level == 'cm':
        exploded_df['draw'] = exploded_df.groupby(['month_id', 'country_id']).cumcount()
    if level == 'pgm':
        exploded_df['draw'] = exploded_df.groupby(['month_id', 'priogrid_id']).cumcount()
    exploded_df.drop(columns=['prediction'], inplace=True)
    exploded_df.rename(columns={'draws':'outcome'}, inplace=True)
    exploded_df.set_index('draw', append=True, inplace=True)
    return exploded_df


def expanded_df_bootstrap(df, ndraws=1000, draw_from=None, level='cm'):
    function_with_draws = partial(sample_bootstrap_row, draw_from=draw_from, ndraws=ndraws)
    df['draws'] = df.apply(function_with_draws, axis=1)
    exploded_df = df.explode('draws').astype('int32')
    if level == 'cm':
        exploded_df['draw'] = exploded_df.groupby(['month_id', 'country_id']).cumcount()
    if level == 'pgm':
        exploded_df['draw'] = exploded_df.groupby(['month_id', 'priogrid_id']).cumcount()
    exploded_df.drop(columns=['ln_ged_sb_dep'], inplace=True)
    exploded_df.rename(columns={'draws':'outcome'}, inplace=True)
    exploded_df.set_index('draw', append=True, inplace=True)
    return exploded_df

# cm level
## Based on ensemble; expanded using a Poisson draw with mean=variance=\hat{y}_{it}


# Assembling benchmark based on VIEWS ensemble predictions

def distribution_expand_single_point_predictions(predictions_df,level,year_list,draws=1000,distribution='poisson'):
    ''' Expands an input prediction df with one prediction per unit to n draws from the point predictions 
    assuming a poisson distribution with mean and variance equal to the point prediction, and returns a list of 
    dictionaries with prediction and metadata for all years in year_list '''

    sc_predictions_ensemble = []
    
    for year in year_list:
        sc_dict = {
            'year': year,
            'prediction_df': extract_sc_predictions(year=year,ss_predictions=predictions_df)
        }
        sc_predictions_ensemble.append(sc_dict)

    # Expanding by drawing n draws from Poisson distribution   

    for year_record in sc_predictions_ensemble:
        print(year_record['year'])
        df = year_record.get('prediction_df')
        year_record['expanded_df'] = expanded_df_distribution(df,ndraws=draws,level=level,distribution=distribution)
    return sc_predictions_ensemble


def bootstrap_expand_single_point_predictions(ensemble_df, draw_from_column, level, year_list, draws=1000):
    """
    Expands an input prediction df with one prediction per unit to n draws from the 'draw_from' array,
    bootstrap-fashion, and returns a list of dictionaries with prediction and metadata for all years in year_list
    """

    actuals = np.expm1(ensemble_df[draw_from_column].fillna(0))
    actuals_by_year = []
    for year in year_list:
        actuals_dict = {
            'year': year,
            'actuals_df': extract_year(year=year, df=actuals)
        }
        actuals_by_year.append(actuals_dict)

    # Expanding by drawing n draws from specified draw_from array

    for year_record in actuals_by_year:
        print(year_record['year'])
        df = year_record.get('actuals_df')
        year_record['expanded_df'] = expanded_df_bootstrap(df, ndraws=draws, draw_from=df[draw_from_column],
                                                           level=level)
    return actuals_by_year

  
def distribution_expand_multiple_point_predictions(ModelList,level,year_list,draws=1000,distribution='poisson'):
    ''' Expands an input prediction df with multiple prediction per unit to n draws from the point predictions 
    assuming a poisson distribution with mean and variance equal to each point prediction. 
    The function then merges all these draws, and returns a list of 
    dictionaries with prediction and metadata for all years in year_list '''

    draws_per_model = np.floor_divide(draws,len(ModelList))
    
    # Drawing from the specified distribution for each of the models in model list
    for model in ModelList:
        print(model['modelname'])

        model['sc_predictions_constituent'] = []
        for year in year_list:
            sc_dict = {
                'year': year,
                'prediction_df': extract_sc_predictions(year=year,ss_predictions=model['predictions_test_df'])
            }
            model['sc_predictions_constituent'].append(sc_dict)

            
        # Expanding by drawing n draws from Poisson distribution   
        for year_record in model['sc_predictions_constituent']:
            print(year_record['year'])
            df = year_record.get('prediction_df')
            year_record['expanded_df'] = expanded_df_distribution(df,draws_per_model,level='cm',distribution=distribution)

    # Assembling benchmark based on the list of expanded model predictions

    sc_predictions_constituent = []

    for year in year_list:
        print(year)
        print(ModelList[0]['modelname'])
        merged_expanded_df = ModelList[0]['sc_predictions_constituent'][year-2018]['expanded_df']
    #    print(expanded_df.describe())
        i = 0
        for model in ModelList[1:]:
            print(model['modelname'])
            merged_expanded_df = pd.concat([merged_expanded_df,model['sc_predictions_constituent'][year-2018]['expanded_df']])
    #        print(expanded_df.describe())

        sc_dict = {
            'year': year,
            'expanded_df': merged_expanded_df
        }
        sc_predictions_constituent.append(sc_dict)
        i = i + 1

    return(sc_predictions_constituent)


def save_models(level,model_names,model_list, filepath):
    ''' Saves the models to dropbox '''
    
    i = 0
    for bm_model in model_list:
        for record in bm_model:
            year_record = record # First part of record list is list of yearly predictions, second is string name for benchmark model
            print(year_record['year'])
            filename = filepath + 'bm_' + level + '_' + model_names[i] + '_expanded_' + str(year_record['year']) + '.parquet'
            print(filename)
            year_record['expanded_df'].to_parquet(filename)
        i = i + 1

def save_actuals(level, df, filepath, year_list):
    ''' Saves the actuals from the given prediction file '''
    # Dataframe with actuals
    df_actuals = pd.DataFrame(df['ln_ged_sb_dep'])
    actuals = df_actuals
    actuals['ged_sb'] = np.expm1(actuals['ln_ged_sb_dep'])
    actuals.drop(columns=['ln_ged_sb_dep'], inplace=True)
    print(actuals.head())
    print(actuals.tail())
    print(actuals.describe())

    # Annual dataframes with actuals, saved to disk
    for year in year_list:
        first_month = (year - 1980)*12 + 1
        last_month = (year - 1980 + 1)*12
        df_annual = actuals.loc[first_month:last_month]
        filename = filepath + level + '_actuals_' + str(year) + '.parquet'
        print(year, first_month, last_month, filename)
        print(df_annual.head())
        df_annual.to_parquet(filename)
    # For all four years
    filename = filepath + level + '_actuals_allyears.parquet'
    actuals.to_parquet(filename)

    
if False:



    # ## Saving the pgm models

    # In[ ]:


    model_names = ['ensemble','constituent']
    i = 0
    for bm_model in [sc_predictions_ensemble_pgm]:
        for record in bm_model:
            year_record = record # First part of record list is list of yearly predictions, second is string name for benchmark model
            print(year_record['year'])
            filename = filepath + 'bm_pgm_' + model_names[i] + '_expanded_' + str(year_record['year']) + '.parquet'
            print(filename)
            year_record['expanded_df'].to_parquet(filename)
        i = i + 1

    # Dataframe with actuals
    df_actuals = pd.DataFrame(ensemble_pgm_df)
    pgm_actuals = df_actuals
    pgm_actuals['ged_sb'] = np.expm1(pgm_actuals['ln_ged_sb_dep'])
    pgm_actuals.drop(columns=['ln_ged_sb_dep'], inplace=True)
    print(pgm_actuals.head())
    print(pgm_actuals.tail())
    print(pgm_actuals.describe())


    # Annual dataframes with actuals, saved to disk
    for year in year_list:
        first_month = (year - 1980)*12 + 1
        last_month = (year - 1980 + 1)*12
        df_annual = pgm_actuals.loc[first_month:last_month]
        filename = filepath + 'cm_actuals_' + str(year) + '.parquet'
        print(year, first_month, last_month, filename)
        print(df_annual.head())
        df_annual.to_parquet(filename)
    # For all four years
    filename = filepath + 'pgm_actuals_allyears.parquet'
    pgm_actuals.to_parquet(filename)


    # In[ ]:




