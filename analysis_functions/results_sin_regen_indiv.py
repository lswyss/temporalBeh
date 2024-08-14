#this is in folder analysis_functions -> results_sin_regen_indiv.py
#Livia 06.28.2024
# regen_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tools.plotting_functions import *
from tools.preparing_data import *
from tools.bootstrapTest import bootstrap_traces


def load_and_filter_regen_data(filepath, genotype, duration, period_suffix, exclude_dates=None):
    """
    Load regeneration data and filter experiments based on the given criteria.
    """
    with open(filepath, 'rb') as f:
        test_result = pickle.load(f)
    
    # Filter experiments based on the provided criteria
    filtered_experiments = {}
    for exp_key in test_result.keys():
        if exp_key == 'tau':
            continue
        exp_parts = exp_key.split('_')
        
        # Ensure exp_parts has the expected length
        if len(exp_parts) < 5:
            print(f"Skipping invalid exp_key: {exp_key}")
            continue
        
        # Check if genotype exists within the first part
        exp_genotype = exp_parts[0]
        exp_duration = exp_parts[3]
        exp_period_suffix = exp_parts[4]
        
        if (genotype in exp_genotype and exp_duration == duration and
                exp_period_suffix == period_suffix and
                (exclude_dates is None or exp_key not in exclude_dates)):
            filtered_experiments[exp_key] = test_result[exp_key]
    
    return test_result, filtered_experiments

def determine_trial_shift(exp_key):
    """Determine the trial shift based on the dpa in the experiment key."""
    if '_0dpa_' in exp_key:
        return 0
    elif '_1dpa_' in exp_key:
        return 12
    elif '_2dpa_' in exp_key:
        return 24
    elif '_3dpa_' in exp_key:
        return 36
    elif '_4dpa_' in exp_key:
        return 48
    elif '_5dpa_' in exp_key:
        return 60
    else:
        return 0  # Default case if no dpa is specified


def prepare_regen_df(test_result, filtered_experiments, trials_range=range(60)):
    """
    Prepare a DataFrame for regeneration experiments with shifted trials.
    """
    rows = []
    
    for exp_key in filtered_experiments:
        shift = determine_trial_shift(exp_key)
        experiment_data = test_result[exp_key]['data']
        color = (np.random.random(), np.random.random(), np.random.random())
        
        for worm_id, worm_data in enumerate(experiment_data):
            for trial_index in trials_range:
                shifted_index = trial_index + shift
                if shifted_index < len(trials_range):
                    if trial_index < worm_data.shape[0]:
                        trial_data = worm_data[trial_index]
                    else:
                        trial_data = [np.nan] * worm_data.shape[1]
                    row = {
                        'experiment': exp_key,
                        'worm_id': worm_id,
                        'trial_index': shifted_index,
                        'trial_data': trial_data,
                        'color': color
                    }
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.pivot(index=['experiment', 'worm_id'], columns='trial_index', values='trial_data')


def aggregate_and_bootstrap(df, trial_groups, tau, n_boot=1000, conf_interval=95):
    """
    Aggregate and bootstrap data for specified trial groups.
    """
    results = {}
    time_indices = np.where((tau >= -2) & (tau <= 35))[0]  # Adjusted to start from -2 minutes
    sliced_tau = tau[time_indices]
    
    for group_name, trial_indices in trial_groups.items():
        group_data = []
        for trial_index in trial_indices:
            if trial_index in df.columns:
                trial_col = df[trial_index].dropna()
                if not trial_col.empty:
                    trial_data = np.vstack(trial_col.values)[:, time_indices]
                    trial_data = trial_data[~np.isnan(trial_data).any(axis=1)]
                    group_data.append(trial_data)
        if group_data:
            group_data = np.vstack(group_data)
            print(f"Group data shape after removing NaNs for {group_name}: {group_data.shape}")
            mean, lower_bound, upper_bound = bootstrap_traces(group_data, n_boot=n_boot, conf_interval=conf_interval)
            results[group_name] = (mean, lower_bound, upper_bound)
    
    return results, sliced_tau

def plot_bootstrapped_trials(results, sliced_tau, stimuli):
    """
    Plot bootstrapped trials for different days with stimuli overlay.
    """
    fig, axs = plt.subplots(len(results), 1, figsize=(15, 5 * len(results)), sharex=True)
    plt.subplots_adjust(hspace=0.1)  # Reduce vertical space between plots

    y_min, y_max = 0, 1.4  # Set y-axis limits
    
    for i, (group_name, (mean, lower_bound, upper_bound)) in enumerate(results.items()):
        ax = axs[i] if len(results) > 1 else axs  # Handle single subplot case
        
        ax.plot(sliced_tau, mean, lw=2, color='cornflowerblue', label=f'Trial Group: {group_name}', zorder=-2)
        ax.fill_between(sliced_tau, lower_bound, upper_bound, alpha=0.5, color='cornflowerblue', lw=0, edgecolor='None', zorder=-2)
        
        ax.plot(sliced_tau, stimuli[:len(sliced_tau)], c='darkorange', label='Stimuli', zorder=-10)  # Overlay stimuli
        
        ax.set_ylabel('Activity', fontsize=12)
        ax.set_ylim(y_min, y_max)  # Set y-axis limits
        ax.legend(fontsize=7, frameon=False, loc='upper right')
        ax.grid(False)

    axs[-1].set_xlabel('Time (min)', fontsize=12) if len(results) > 1 else axs.set_xlabel('Time (min)', fontsize=12)
    axs[-1].set_xlim(-2, 35)  # Ensure x-axis starts at -2 minutes
    plt.tight_layout()
    plt.show()
