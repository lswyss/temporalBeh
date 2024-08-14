#04.24.2024 Livia
#Functions for timeseries data on individual trials and worms


import numpy as np
import matplotlib.pyplot as plt
import pickle
from tools.preparing_data import *
from tools.filtering import filter_experiments
from tools.bootstrapTest import bootstrap_traces
from analysis_functions.PCA_behavior import *
#from analysis_functions.results_sin_indiv import bootstrap_traces


def load_and_filter_data(filepath, genotype, duration, period_suffix, exclude_dates=None):
    """
    Load data and filter experiments based on the given criteria.
    """
    with open(filepath, 'rb') as f:
        test_result = pickle.load(f)
    
    # Filter experiments
    filtered_experiments = filter_experiments(test_result, genotype, duration, period_suffix, exclude_dates)
    return test_result, filtered_experiments

#######################################################

# def plot_worm_trials(experiment_name):
#     """Load data and set up plot parameters, then call plotting function."""
#     # Load your dataset
#     with open('data/LDS_response_sinFunc_indiv.pickle', 'rb') as f:
#         test_result = pickle.load(f)
    
#     # Access the experiment data
#     experiment_data = test_result[experiment_name]['data']
#     stim_data = test_result[experiment_name]['stim']
#     num_worms = len(experiment_data)
#     max_trials = max([worm.shape[0] for worm in experiment_data])  # Determine the maximum number of trials any worm underwent
    
#     # Find the index where the stimulus first becomes non-zero
#     stim_start_index = np.argmax(stim_data > 0)

#     # Time configuration
#     frames_per_second = 2
#     seconds_per_point = 1 / frames_per_second
#     time_axis_seconds = (np.arange(stim_data.size) - stim_start_index) * seconds_per_point
#     time_axis_minutes = time_axis_seconds / 60  # Convert seconds to minutes

#     # Create a grid of plots with better spacing
#     fig, axs = plt.subplots(nrows=num_worms, ncols=max_trials, figsize=(max_trials * 3, num_worms * 3), gridspec_kw={'hspace': 0.5, 'wspace': 0.3})
    
#     # Call the plotting function
#     gridplot_indiv_trials(axs, experiment_data, stim_data, time_axis_minutes, num_worms, max_trials)

#     plt.tight_layout()
#     plt.show()

def prepare_aggregated_data_from_df(df, tau, test_result, max_trials_limit=None):
    # Extracting the relevant time indices based on tau
    time_indices = np.where((tau >= -10) & (tau <= 40))[0]
    sliced_tau = tau[time_indices]
    
    # Initialize the list to store aggregated data for each trial
    aggregated_data = []
    
    # Get stimulus data and normalize it
    first_exp_key = next(iter(df['experiment'].unique()))
    stim_data = test_result[first_exp_key]['stim'][time_indices]
    adjusted_stim_data = (stim_data - np.min(stim_data)) / (np.max(stim_data) - np.min(stim_data))
    
    # Determine the maximum number of trials
    max_trials = 0
    for trial_col in df.columns[4:]:
        if df[trial_col].notna().any():
            max_trials += 1
    
    if max_trials_limit is not None:
        max_trials = min(max_trials, max_trials_limit)
    
    # Aggregate data for each trial
    for trial_index in range(max_trials):
        trial_data = []
        trial_col = f'trial_{trial_index}'
        for exp_key in df['experiment'].unique():
            experiment_data = df[df['experiment'] == exp_key]
            for worm_id in experiment_data['worm_id'].unique():
                worm_data = experiment_data[experiment_data['worm_id'] == worm_id]
                if not worm_data[trial_col].isna().all():
                    valid_data = worm_data[trial_col].dropna().values
                    if valid_data.size > 0:
                        trial_data.append(np.vstack(valid_data)[:, time_indices])
        if trial_data:
            trial_data = np.vstack(trial_data)
            aggregated_data.append(trial_data)
    
    return aggregated_data, sliced_tau, adjusted_stim_data, max_trials

def plot_aggregated_trials(aggregated_data, sliced_tau, adjusted_stim_data, max_trials, n_boot=1000, conf_interval=95):
    fig, axs = plt.subplots(max_trials, 1, figsize=(8, 2 * max_trials), sharex=True)
    plt.subplots_adjust(hspace=0.05)
    
    for trial_index in range(max_trials):
        ax = axs[trial_index]
        if trial_index < len(aggregated_data):
            trial_data = aggregated_data[trial_index]
            mean, lower_bound, upper_bound = bootstrap_traces(trial_data, n_boot=n_boot, conf_interval=conf_interval)
            ax.fill_between(sliced_tau, lower_bound, upper_bound, color='cornflowerblue', alpha=0.2)
            ax.plot(sliced_tau, mean, color='cornflowerblue')
            if adjusted_stim_data is not None:
                ax.plot(sliced_tau, adjusted_stim_data, color='darkorange', linestyle='--', label='Stimulus' if trial_index == 0 else "")
            ax.text(1.01, 0.9, f'Trial {trial_index + 1}', transform=ax.transAxes, verticalalignment='center', horizontalalignment='left', fontsize=9)
            ax.text(1.01, 0.1, f'N={trial_data.shape[0]}', transform=ax.transAxes, verticalalignment='center', horizontalalignment='left', fontsize=9)
        ax.set_ylim(0, 3.5)
        ax.set_yticks([0, 1, 2, 3])
    
    axs[-1].set_xlabel('Time (min)', fontsize=12)
    fig.text(0.04, 0.5, 'Activity', ha='center', va='center', rotation='vertical', fontsize=12)
    plt.show()