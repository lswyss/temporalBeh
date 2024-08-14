#this is in folder analysis_functions -> results_random.py
#Livia 07.16.2024
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tools.bootstrapTest import bootstrap_traces_sam


def load_and_filter_randsin_data(filepath, genotype, duration, period_suffix, exclude_dates=None):
    """
    Load random sine data and filter experiments based on the given criteria.
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
        if len(exp_parts) < 3:
            print(f"Skipping invalid exp_key: {exp_key}")
            continue
        
        exp_genotype = exp_parts[1] if exp_parts[0].isdigit() else exp_parts[0]
        exp_duration = exp_parts[2] if exp_parts[0].isdigit() else exp_parts[1]
        exp_period_suffix = exp_parts[-1]
        
        if (genotype in exp_genotype and exp_duration == duration and
                exp_period_suffix == period_suffix and
                (exclude_dates is None or exp_key not in exclude_dates)):
            filtered_experiments[exp_key] = test_result[exp_key]
    
    return test_result, filtered_experiments


def generate_stimulus(tau, stim_list):
    """
    Generate a stimulus array based on the given stim_list.
    """
    new_stim = np.zeros_like(tau)
    start_index = np.where(tau == 0)[0][0]
    end_stimulus = np.where(tau == 30)[0][0]

    for s in stim_list:
        period = s    
        period_steps = int(period * 120)
        end_index = start_index + period_steps

        if end_index > end_stimulus:
            end_index = end_stimulus

        t_values = tau[start_index:end_index]
        freq = 1 / period
        new_stim[start_index:end_index] = -0.5 * np.cos(2 * np.pi * freq * (t_values - tau[start_index]))+0.5

        start_index = end_index

        if start_index >= end_stimulus:
            break

    return new_stim

def response_random_sin(filtered_experiments, original_data_filepath, n_boot=1000, statistic=np.mean, conf_interval=95, t_samp=(-2, 35), stim_list=None):
    """
    Process and plot random sine data from filtered experiments.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

    # Load the original data to get the tau time points
    with open(original_data_filepath, 'rb') as f:
        original_data = pickle.load(f)
    
    tau = original_data['tau']
    ind_t = np.where((tau >= t_samp[0]) & (tau <= t_samp[1]))[0]

    # Concatenate data from filtered experiments
    concatenated_data = np.concatenate([exp['data'] for exp in filtered_experiments.values()], axis=0)

    # Perform bootstrapping on the concatenated data
    y, rng = bootstrap_traces_sam(concatenated_data[:, ind_t], n_boot=n_boot, statistic=statistic, conf_interval=conf_interval)

    # Generate or extract stimulus data
    if stim_list is not None:
        stim_data = generate_stimulus(tau, stim_list)[ind_t]
    else:
        stim_data = original_data[next(iter(filtered_experiments))]['stim'][ind_t]

    # Plotting function
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'svg.fonttype': 'none'})
    
    ax.plot(tau[ind_t], y, lw=2, color='cornflowerblue', label='Random Sine Data', zorder=-2)
    
    # Filling between ranges with confidence interval
    ax.fill_between(tau[ind_t], *rng, alpha=0.5, color='cornflowerblue', lw=0, edgecolor='None', zorder=-2)
    
    if stim_data is not None:
        ax.plot(tau[ind_t], stim_data, c='darkorange', zorder=-10)
    
    # Set labels and title with font properties directly applied
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Activity', fontsize=12)
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    
    # Set y-axis limits
    ax.set_ylim(0, 1.8)
    ax.set_xlim(-2, 35)

    ax.grid(False)

    plt.show()
    return fig, ax

