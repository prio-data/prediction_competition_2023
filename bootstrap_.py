import pandas as pd
import argparse
from functools import partial
import numpy as np
import pyarrow.parquet as pq 
#import pdb; pdb.set_trace()
from pathlib import Path

def bootstrap_expand_single_point_predictions(ensemble_df, draw_from_column, level, year_list, draws=1000):
    """
    Expands an input prediction df with one prediction per unit to n draws from the 'draw_from' array,
    bootstrap-fashion, and returns a list of dictionaries with prediction and metadata for all years in year_list
    """

    actuals = ensemble_df[draw_from_column].fillna(0)
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

def expanded_df_bootstrap(df, ndraws=1000, draw_from=None, level='cm'):
    function_with_draws = partial(sample_bootstrap_row, draw_from=draw_from, ndraws=ndraws)
    df['draws'] = df.apply(function_with_draws, axis=1)
    exploded_df = df.explode('draws').astype('int32')
    if level == 'cm':
        exploded_df['draw'] = exploded_df.groupby(['month_id', 'country_id']).cumcount()
    if level == 'pgm':
        exploded_df['draw'] = exploded_df.groupby(['month_id', 'priogrid_gid']).cumcount()
    exploded_df.drop(columns=['outcome'], inplace=True)
    exploded_df.rename(columns={'draws':'outcome'}, inplace=True)
    exploded_df.set_index('draw', append=True, inplace=True)
    return exploded_df

def extract_year(year, df):
    """
    Extract data from input dataframe in year-chunks
    """
    first_month = (year - 1980)*12 + 1
    last_month = first_month + 11
    return pd.DataFrame(df.loc[first_month:last_month])

def sample_bootstrap_row(row: pd.DataFrame, draw_from: np.array, ndraws: int = 100) -> np.ndarray:
    """Given a dataframe row, produce ndraws draws from the prediction column in the df.
    Attention, this is a row vectorized approach, should be used with apply.
    :return an np array. Should be exploded for long df.
    """
    return np.random.choice(draw_from, size=ndraws, replace=True)

def save_models(level,model_names,model_list, filepath):
    ''' Saves the models to dropbox '''
    
    i = 0
    for bm_model in model_list:
        for record in bm_model:
            year_record = record # First part of record list is list of yearly predictions, second is string name for benchmark model
            print(year_record['year'])
            filename = filepath + 'bm_' + level + '_' + model_names[i] + '_expanded_' + str(year_record['year']+1) + '.parquet'
            print(filename)
            year_record['expanded_df'].to_parquet(filename)
        i = i + 1
        
def benchmark_bootstrap(input_file,year,cmorpgm):
    try:
        # Read the input Parquet file
        df = pd.read_parquet(input_file)

        # Add transformation code here
        year_list = [year]
        actual_pgm = bootstrap_expand_single_point_predictions(df,draw_from_column='outcome',level=cmorpgm,year_list=year_list,draws=1000) 
        model_names = ['bootstrap']
        model_list = [actual_pgm]
        filepath = str(Path(input_file).parent)+'/'
        save_models(cmorpgm,model_names,model_list, filepath)
        
        # Save the DataFrame to the output Parquet file
        #df.to_parquet(output_file, index=False)
        #print(f"Parquet file '{input_file}' copied to '{output_file}' successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy a Parquet file.")
    parser.add_argument("input_file", help="Path to the input Parquet file.")
    #parser.add_argument("output_file", help="Path to the output Parquet file.")
    parser.add_argument("year_of_actuals",type=int,help="Year of actuals")
    parser.add_argument("cm_or_pgm",type=str,help='Enter cm or pgm based on file name')
    args = parser.parse_args()
    benchmark_bootstrap(args.input_file,args.year_of_actuals,args.cm_or_pgm)
