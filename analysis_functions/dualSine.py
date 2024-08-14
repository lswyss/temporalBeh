import numpy as np
import matplotlib.pyplot as plt
import pickle
from tools.bootstrapTest import bootstrap_traces_sam

def load_and_filter_dualsine(filepath, period_UV, period_Vis, amplitude, phase_shift, exclude_dates=None):
    """
    Load dual sine data and filter experiments based on the given criteria, specifying UV and visible periods, amplitude, and phase shift.
    """
    with open(filepath, 'rb') as f:
        test_result = pickle.load(f)
    
    # Filter experiments based on the provided criteria
    filtered_experiments = {}
    
    for exp_key in test_result.keys():
        if exp_key == 'tau':
            continue
        
        # Split the experiment key into parts
        exp_parts = exp_key.split('_')
        
        
        # Ensure exp_parts has the expected length
        if len(exp_parts) < 7:
            print(f"Skipping invalid exp_key: {exp_key}")
            continue
        
        exp_period_UV = exp_parts[3]
        exp_amplitude_UV = exp_parts[4]
        exp_period_Vis = exp_parts[7]
        exp_phase_shift = exp_parts[-1] if 'Phase' in exp_parts[-1] else None
        
      
        # Match the 'step' condition first
        if period_UV == 'step' and exp_period_UV == 'step':
            print(f"Match found for 'step': {exp_key}")
            if (exp_amplitude_UV == amplitude and
                exp_period_Vis == period_Vis and
                (phase_shift is None or exp_phase_shift == phase_shift) and
                (exclude_dates is None or exp_key not in exclude_dates)):
                filtered_experiments[exp_key] = test_result[exp_key]
                print(f"Added to filtered experiments: {exp_key}")
        
        # Match other conditions
        elif (exp_period_UV == period_UV and
              exp_amplitude_UV == amplitude and
              exp_period_Vis == period_Vis and
              (phase_shift is None or exp_phase_shift == phase_shift) and
              (exclude_dates is None or exp_key not in exclude_dates)):
            filtered_experiments[exp_key] = test_result[exp_key]
            print(f"Added to filtered experiments: {exp_key}")
    
    return test_result, filtered_experiments

def response_dualsine(filtered_experiments, original_data_filepath, period_UV, period_Vis, phase_shift, n_boot=1000, statistic=np.mean, conf_interval=95, t_samp=(-2, 35)):
    """
    Process and plot dual sine data from filtered experiments with UV and visible stimuli.
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

    # Convert phase shift input from string to radians
    if phase_shift == '-.5piPhase':
        phase_shift_radians = 1.5 * np.pi
    elif phase_shift == '1piPhase':
        phase_shift_radians = np.pi
    elif phase_shift == '0piPhase':
        phase_shift_radians = 0
    else:
        phase_shift_radians = 0  # Default to no phase shift if not specified

    # Determine if period_UV is a step or a regular period
    if period_UV == 'step':
        stim_uv = calculate_fixed_period_stimulus(tau, 'step')
    else:
        stim_uv = calculate_fixed_period_stimulus(tau, float(period_UV.split('m')[0]), phase_shift=0)  # no phase shift for UV

    # Calculate the Visible light stimulus with phase shift
    stim_vis = calculate_fixed_period_stimulus(tau, float(period_Vis.split('m')[0]), phase_shift=phase_shift_radians)

    # Plotting function
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'svg.fonttype': 'none'})
    
    ax.plot(tau[ind_t], y, lw=2, color='cornflowerblue', label='Dual Sine Data', zorder=-2)
    
    # Filling between ranges with confidence interval
    ax.fill_between(tau[ind_t], *rng, alpha=0.5, color='cornflowerblue', lw=0, edgecolor='None', zorder=-2)
    
    # Plot UV and visible stimuli
    ax.plot(tau[ind_t], stim_uv[ind_t], c='darkorange', label='UV Stimulus (Step)' if period_UV == 'step' else 'UV Stimulus', zorder=-10)
    ax.plot(tau[ind_t], stim_vis[ind_t], c='limegreen', linestyle='--', label='Visible Stimulus', zorder=-10)
    
    # Set labels and title with font properties directly applied
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Activity', fontsize=12)
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    
    # Set y-axis limits
    ax.set_ylim(0, 1.4)
    ax.set_xlim(-2, 35)

    ax.grid(False)

    plt.show()
    return fig, ax


def calculate_fixed_period_stimulus(tau, period, phase_shift=0):
    """
    Calculate a stimulus with a fixed period for the given tau values, including an optional phase shift.
    The stimulus starts at 0 min and ends at 30 min, oscillating with an amplitude from 0 to 1.
    
    If `period` is 'step', the stimulus is a step function that turns on at 0 min and stays on for 30 min with an activity level of 0.5.
    
    Parameters:
    - tau: numpy array of time points
    - period: float or string ('step'), the period of the sine wave in minutes, or 'step'
    - phase_shift: float, the phase shift in radians (e.g., 0.5π, 1π)
    """
    # Initialize the stimulus array with zeros
    new_stim = np.zeros_like(tau)

    # Find the indices corresponding to 0 min and 30 min
    start_index = np.where(tau >= 0)[0][0]
    end_stimulus = np.where(tau <= 30)[0][-1]

    if period == 'step':
        # Create a step function: turn on at 0 min, stay on for 30 min with activity level 0.5
        new_stim[start_index:end_stimulus] = 0.5
    else:
        # Calculate the period steps in tau (assuming tau is in minutes)
        period_steps = int(float(period) * 120)  # Assuming tau is in minutes and sampled at 2 Hz (120 samples per minute)
        end_index = start_index

        while end_index < end_stimulus:
            end_index = start_index + period_steps
            if end_index > end_stimulus:
                end_index = end_stimulus

            t_values = tau[start_index:end_index]
            freq = 1 / float(period)
            new_stim[start_index:end_index] = -0.5 * np.cos(2 * np.pi * freq * (t_values) + phase_shift) + 0.5

            start_index = end_index

    return new_stim