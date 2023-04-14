#!/usr/bin/env python
# coding: utf-8

# # Benchmark model generation

# In[ ]:


# Basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import os
from functools import partial

# Views 3
import views_runs
from viewser.operations import fetch
from views_forecasts.extensions import *
from viewser import Queryset, Column


# In[ ]:


# Common parameters:

dev_id = 'Fatalities002'
run_id = 'Fatalities002'
EndOfHistory = 508
get_future = False

username = os.getlogin()

steps = [*range(1, 36+1, 1)] # Which steps to train and predict for

fi_steps = [1,3,6,12,36]
# Specifying partitions

calib_partitioner_dict = {"train":(121,396),"predict":(397,456)}
test_partitioner_dict = {"train":(121,444),"predict":(457,504)}
future_partitioner_dict = {"train":(121,492),"predict":(505,512)}
calib_partitioner =  views_runs.DataPartitioner({"calib":calib_partitioner_dict})
test_partitioner =  views_runs.DataPartitioner({"test":test_partitioner_dict})
future_partitioner =  views_runs.DataPartitioner({"future":future_partitioner_dict})

Mydropbox = f'/Users/{username}/Dropbox (ViEWS)/ViEWS/'
overleafpath = f'/Users/{username}/Dropbox (ViEWS)/Apps/Overleaf/Prediction competition 2023/'


print('Dropbox path set to',Mydropbox)
print('Overleaf path set to',overleafpath)


# In[ ]:


# Benchmark model parameters
filepath = Mydropbox + 'Prediction_competition_2023/'

year_list = [2018, 2019, 2020, 2021]
draws_cm = 1000
draws_pgm = 100

steps = [3,4,5,6,7,8,9,10,11,12,13,14]
stepcols = ['ln_ged_sb_dep']
for step in steps:
    stepcols.append('step_pred_' + str(step))
print(stepcols)


# # Auxilliary functions

# In[ ]:


# Extract sc predictions for a given calendar year

def extract_sc_predictions(year,ss_predictions):
    ''' Extract sc predictions'''
    first_month = (year - 1980)*12 + 1
    months=list(range(first_month,first_month+12))
    df = ss_predictions.loc[months].copy()
    df['prediction'] = 0
    for month in range(1,12+1):
        this_col = 'step_pred_' + str(month+2)
        this_month = first_month + month - 1
#        print(month, this_col, this_month)
        df_temp = df.loc[this_month]
#        print(df_temp[this_col].values)
        df_temp['prediction'] = np.expm1(df_temp[this_col].values) 
        df['prediction'].loc[this_month] = df_temp['prediction'].values
    return pd.DataFrame(df['prediction'])


# In[ ]:


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


# In[ ]:


def sample_poisson_row(row: pd.DataFrame, ndraws:int = 100) -> pd.DataFrame:
    """Given a dataframe row, produce ndraws poisson draws from the prediction column in the row.
    Attention, this is a row vectorized approach, should be used with apply.
    :return an np array. Should be exploded for long df.
    """
    row.prediction = 0 if row.prediction<=0 else row.prediction
    return np.random.poisson(row.prediction, size=ndraws)

def expanded_df(df, ndraws=1000,level='cm'):
    function_with_draws = partial(sample_poisson_row, ndraws=ndraws)
    df['draws'] = df.apply(function_with_draws, axis=1)
    exploded_df = df.explode('draws')
    if level=='cm':
        exploded_df['draw'] = exploded_df.groupby(['month_id','country_id']).cumcount()
    if level=='pgm':
        exploded_df['draw'] = exploded_df.groupby(['month_id','priogrid_id']).cumcount()
    exploded_df.drop(columns=['prediction'],inplace=True)
    exploded_df.rename(columns={'draws':'outcome'},inplace=True)
    exploded_df.set_index('draw', append=True,inplace=True)
    return(exploded_df)


# # cm level
# ## Based on ensemble; expanded using a Poisson draw with mean=variance=\hat{y}_{it}

# In[ ]:


# Assembling benchmark based on VIEWS ensemble predictions
sc_predictions_ensemble = []
cm_ensemble_name = 'cm_ensemble_genetic_test'
    
ensemble_df = pd.DataFrame.forecasts.read_store(cm_ensemble_name, run=dev_id)[stepcols]
ensemble_df.head()

for year in year_list:
    sc_dict = {
        'year': year,
        'prediction_df': extract_sc_predictions(year=year,ss_predictions=ensemble_df)
    }
    sc_predictions_ensemble.append(sc_dict)
    


# In[ ]:


# Expanding by drawing n draws from Poisson distribution   

for year_record in sc_predictions_ensemble:
    print(year_record['year'])
    df = year_record.get('prediction_df')
    year_record['expanded_df'] = expanded_df(df,ndraws=1000,level='cm')
    
describe_expanded(df=sc_predictions_ensemble[0]['prediction_df'], df_expanded=sc_predictions_ensemble[0]['expanded_df'], month=457, country=57)   

sc_predictions_ensemble[0]['expanded_df'].head()<


# # Based on constituent models
# 
# Short version, 20 models: 1 "draw" from each of 20 constituent models
# 
# Plus version with 45 draws from Poisson distribution for each model.
# 
# Possibly obsolete: Long version, 440 models: 20 "draws" from each of 22 constituent models, using predictions for adjacent steps (from s-4 to s+6). Some duplications to weight the most proximate steps more.

# In[ ]:


# Fatalities002 stuff - contains the list of the current fatalities002 ensemble models

from ModelDefinitions import DefineEnsembleModels

level = 'cm'
ModelList_cm = DefineEnsembleModels(level)
ModelList_cm = ModelList_cm[0:20] # Drop Markov models

i = 0
for model in ModelList_cm:
    print(i, model['modelname'], model['data_train'])
    i = i + 1

# Retrieving the predictions for calibration and test partitions
# The ModelList contains the predictions organized by model
from Ensembling import CalibratePredictions, RetrieveStoredPredictions, mean_sd_calibrated, gam_calibrated

ModelList_cm = RetrieveStoredPredictions(ModelList_cm, steps, EndOfHistory, dev_id, level, get_future)

ModelList_cm = CalibratePredictions(ModelList_cm, EndOfHistory, steps)


# In[ ]:


# Assembling benchmark based on VIEWS constituent model predictions
draws_per_model = np.floor_divide(draws_cm,len(ModelList_cm))

for model in ModelList_cm:
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
        year_record['expanded_df'] = expanded_df(df,ndraws=50,level='cm')


# In[ ]:


sc_predictions_constituent = []

for year in year_list:
    print(year)
    print(ModelList_cm[0]['modelname'])
    merged_expanded_df = ModelList_cm[0]['sc_predictions_constituent'][year-2018]['expanded_df']
#    print(expanded_df.describe())
    i = 0
    for model in ModelList_cm[1:19]:
        print(model['modelname'])
        merged_expanded_df = pd.concat([merged_expanded_df,model['sc_predictions_constituent'][year-2018]['expanded_df']])
#        print(expanded_df.describe())
        
    sc_dict = {
        'year': year,
        'expanded_df': merged_expanded_df
    }
    sc_predictions_constituent.append(sc_dict)
    i = i + 1
       
#sc_predictions


# # Saving the cm benchmark models

# In[ ]:


model_names = ['ensemble','constituent']
i = 0
for bm_model in [sc_predictions_ensemble,sc_predictions_constituent]:
    for record in bm_model:
        year_record = record # First part of record list is list of yearly predictions, second is string name for benchmark model
        print(year_record['year'])
        filename = filepath + 'bm_cm_' + model_names[i] + '_expanded_' + str(year_record['year']) + '.parquet'
        print(filename)
        year_record['expanded_df'].to_parquet(filename)
    i = i + 1

# Dataframe with actuals
df_actuals = pd.DataFrame(ModelList_cm[0]['predictions_test_df']['ln_ged_sb_dep'])
cm_actuals = df_actuals
cm_actuals['ged_sb'] = np.expm1(cm_actuals['ln_ged_sb_dep'])
cm_actuals.drop(columns=['ln_ged_sb_dep'], inplace=True)
print(cm_actuals.head())
print(cm_actuals.tail())
print(cm_actuals.describe())


# Annual dataframes with actuals, saved to disk
for year in year_list:
    first_month = (year - 1980)*12 + 1
    last_month = (year - 1980 + 1)*12
    df_annual = cm_actuals.loc[first_month:last_month]
    filename = filepath + 'cm_actuals_' + str(year) + '.parquet'
    print(year, first_month, last_month, filename)
    print(df_annual.head())
    df_annual.to_parquet(filename)
# For all four years
filename = filepath + 'cm_actuals_allyears.parquet'
cm_actuals.to_parquet(filename)


# # pgm level

# In[ ]:


# Assembling benchmark based on VIEWS ensemble predictions
sc_predictions_ensemble_pgm = []
# any old pgm data
pgm_ensemble_name = 'pgm_ensemble_cm_calib_test'
    
ensemble_pgm_df = pd.DataFrame.forecasts.read_store(pgm_ensemble_name, run=dev_id)[stepcols]
ensemble_pgm_df.head()

for year in year_list[0:3]:
    sc_dict = {
        'year': year,
        'prediction_df': extract_sc_predictions(year=year,ss_predictions=ensemble_pgm_df)
    }
    sc_predictions_ensemble_pgm.append(sc_dict)
    


# In[ ]:


# Expanding by drawing n draws from Poisson distribution   
function_with_draws = partial(sample_poisson_row, ndraws=500)
for year_record in sc_predictions_ensemble_pgm:
    print(year_record['year'])
    df = year_record.get('prediction_df')
    year_record['expanded_df'] = expanded_df(df,ndraws=500,level='pgm')

#describe_expanded(df=sc_predictions_ensemble_pgm[0]['prediction_df'], df_expanded=sc_predictions_ensemble_pgm[0]['expanded_df'], month=457, country=57)   


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




