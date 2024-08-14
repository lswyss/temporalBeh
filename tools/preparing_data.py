#this is in folder tools -> dataframe.py
#Livia 06.21.2024
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from tools.filtering import filter_experiments
from tools.bootstrapTest import bootstrap_traces

# Define markers and color palette globally within this script
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', 'P', '*', 'X', 'd']
color_palette = sns.color_palette("tab20", 20)  # More distinct colors

# Function to prepare data based on experiments and organize into a DataFrame with trials as columns
def prepare_data_df(test_result, filtered_experiments, trials_range, exclude_worms=None):
    experiment_to_color = {exp: color_palette[i % len(color_palette)] for i, exp in enumerate(filtered_experiments)}
    print("Experiment to Color Mapping:")
    for exp, color in experiment_to_color.items():
        print(f"{exp}: {color}")
    
    data_records = []
    for exp in filtered_experiments:
        experiment_data = test_result[exp]['data']
        for worm_idx, worm_data in enumerate(experiment_data):
            if exclude_worms and exp in exclude_worms and worm_idx in exclude_worms[exp]:
                continue
            record = {
                'experiment': exp, 
                'worm_id': worm_idx,
                'color': experiment_to_color[exp],
                'marker': markers[worm_idx % len(markers)]
            }
            for trial_idx in trials_range:
                if trial_idx < worm_data.shape[0]:
                    trial_data = worm_data[trial_idx]
                    record[f'trial_{trial_idx}'] = trial_data
                else:
                    record[f'trial_{trial_idx}'] = np.nan
            data_records.append(record)
    df = pd.DataFrame(data_records)
    return df

# Function to exclude specific worms
def exclude_worms_from_df(df, exclude_worms):
    for exp, worms in exclude_worms.items():
        df = df[~((df['experiment'] == exp) & (df['worm_id'].isin(worms)))]
    return df

def print_df_summary(df):
    print("DataFrame Summary:")
    print(df.info())
    print("\nExperiment Counts:")
    print(df['experiment'].value_counts())
    print("\nWorm Counts per Experiment:")
    print(df.groupby('experiment')['worm_id'].nunique())
    print("\nTrial Counts per Worm:")
    print(df.groupby(['experiment', 'worm_id']).count())



def load_and_filter_data(filepath, genotype, duration, period_suffix, exclude_dates=None):
    """
    Load data and filter experiments based on the given criteria.
    """
    with open(filepath, 'rb') as f:
        test_result = pickle.load(f)
    
    # Filter experiments
    filtered_experiments = filter_experiments(test_result, genotype, duration, period_suffix, exclude_dates)
    return test_result, filtered_experiments

# def bootstrap_traces(data, n_boot=1000, conf_interval=99):
#     """
#     Bootstrap data for generating confidence intervals.
#     """
#     sample_size = data.shape[0]
#     bootstrap = []
#     for _ in range(int(n_boot)):
#         sampled_indices = np.random.choice(range(sample_size), size=sample_size, replace=True)
#         sampled_data = data[sampled_indices, :]
#         bootstrap.append(np.mean(sampled_data, axis=0))
#     bootstrap = np.array(bootstrap)
#     mean = np.mean(bootstrap, axis=0)
#     lower_bound = np.percentile(bootstrap, (100 - conf_interval) / 2, axis=0)
#     upper_bound = np.percentile(bootstrap, 100 - (100 - conf_interval) / 2, axis=0)
#     return mean, lower_bound, upper_bound


def prepare_aggregated_data(test_result, filtered_experiments, tau, max_trials_limit=None):
    """
    Prepare aggregated data for trials with an optional limit on the number of trials.
    """
    time_indices = np.where((tau >= -10) & (tau <= 40))[0]
    sliced_tau = tau[time_indices]
    
    aggregated_data = []
    
    first_exp_key = next(iter(filtered_experiments))
    if 'stim' in test_result[first_exp_key]:
        stim_data = test_result[first_exp_key]['stim'][time_indices]
        adjusted_stim_data = (stim_data - np.min(stim_data)) / (np.max(stim_data) - np.min(stim_data))
    else:
        adjusted_stim_data = None
    
    max_trials = 0
    for exp_key in filtered_experiments:
        experiment_data = test_result[exp_key]['data']
        for worm_data in experiment_data:
            max_trials = max(max_trials, worm_data.shape[0])
    
    if max_trials_limit is not None:
        max_trials = min(max_trials, max_trials_limit)
    
    for trial_index in range(max_trials):
        trial_data = []
        for exp_key in filtered_experiments:
            experiment_data = test_result[exp_key]['data']
            for worm_data in experiment_data:
                if worm_data.shape[0] > trial_index:
                    trial_data.append(worm_data[trial_index][time_indices])
        if trial_data:
            trial_data = np.vstack(trial_data)
            aggregated_data.append(trial_data)
    
    return aggregated_data, sliced_tau, adjusted_stim_data, max_trials
