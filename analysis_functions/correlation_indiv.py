#04.24.2024 Livia
#Functions for calculating correlation on individual trials and worms
#file analysis_functions import correlation_indiv.py"""

"""
This script processes experimental data to compute the correlation between activity data and stimulus segments
over time. The key steps include:

1. **Load and Filter Data**:
   - Load data from a pickle file.
   - Filter experiments based on genotype and duration.
2. **Extract Active Stimulus Segment**:
   - Identify the start and end of the active stimulus segment where the stimulus is non-zero.
3. **Sliding Window Correlation**:
   - Slide a window over the active stimulus segment.
   - Calculate the correlation with the corresponding activity segment for each trial.
4. **Output Correlations**:
   - Print the computed correlations for further analysis.


Key Functions:
- `compute_correlation(activity_data, stimulus_segment)`: Computes the correlation if segments have non-zero standard deviation.
- `extract_active_stimulus_segment(stimulus_data)`: Extracts the first non-zero to last non-zero segment of the stimulus.
- `slide_and_correlate(aggregated_data, stimulus_data, period_points)`: Slides a window over the stimulus segment and computes correlations.
- `quantify_response(test_result, filtered_experiments, tau, period_suffix)`: Main function to load data, process it, and compute correlations.

Ensure the necessary dependencies are installed before running the script.
"""

# File: correlation_indiv.py

import numpy as np
import pickle
from tools.filtering import filter_experiments
from analysis_functions.results_sin_indiv import *


def compute_correlation(activity_data, stimulus_segment):
    if np.std(activity_data) == 0 or np.std(stimulus_segment) == 0:
        print("Zero standard deviation detected")
        return np.nan  # Skip correlation if any segment has zero standard deviation
    return np.corrcoef(activity_data, stimulus_segment)[0, 1]

def extract_sine_wave_segment(stimulus_data, period_points):
    sine_wave_segment = stimulus_data[:period_points]
    return sine_wave_segment

def extract_active_stimulus_segment(stimulus_data):
    start_index = np.argmax(stimulus_data > 0)
    end_index = len(stimulus_data) - np.argmax(stimulus_data[::-1] > 0)
    active_stimulus_segment = stimulus_data[start_index:end_index]
    return active_stimulus_segment, start_index, end_index

def segment_activity_data(activity_data, start_index, end_index, period_points):
    """
    Segment the activity data into periods, aligned with the active segment of the stimulus.
    """
    active_activity_data = activity_data[start_index:end_index]
    segments = []
    for start in range(0, len(active_activity_data), period_points):
        segment = active_activity_data[start:start + period_points]
        if len(segment) == period_points and np.std(segment) != 0:  # Ensure complete and non-constant segments
            segments.append(segment)
    return segments

def calculate_segment_correlations(segments, sine_wave_segment):
    """
    Calculate the correlation for each segment.
    """
    correlations = []
    for segment in segments:
        correlation = compute_correlation(segment, sine_wave_segment)
        correlations.append(correlation)
    return correlations

def slide_and_correlate(aggregated_data, stimulus_data, period_points):
    active_stimulus_segment, start_index, _ = extract_active_stimulus_segment(stimulus_data)
    correlations = []
    half_period_points = period_points // 2
    
    for trial_data in aggregated_data:
        trial_correlations = []
        for start in range(0, len(active_stimulus_segment) - period_points + 1, half_period_points):
            segment = active_stimulus_segment[start:start + period_points]
            if segment.size == period_points:
                activity_segment = trial_data[:, start_index + start:start_index + start + period_points].mean(axis=0)
                correlation = compute_correlation(activity_segment, segment)
                trial_correlations.append(correlation)
        correlations.append(trial_correlations)
    return correlations

def quantify_response(test_result, filtered_experiments, tau, period_suffix, max_trials_limit=None):
    period_points = int(float(period_suffix.replace('m', '')) * 60 * 2)  # Use period_suffix directly after removing 'm'
    aggregated_data, sliced_tau, adjusted_stim_data, _ = prepare_aggregated_data(test_result, filtered_experiments, tau, max_trials_limit)
    correlations = slide_and_correlate(aggregated_data, adjusted_stim_data, period_points)
    return correlations

def plot_correlations(correlations):
    """
    Plot the correlations over segments with a separate subplot for each trial.
    """
    num_trials = len(correlations)
    fig, axs = plt.subplots(num_trials, 1, figsize=(10, 2 * num_trials), sharex=True)
    plt.subplots_adjust(hspace=0.5)
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_trials))
    
    for i, (trial_corr, color) in enumerate(zip(correlations, colors)):
        ax = axs[i]
        ax.plot(range(len(trial_corr)), trial_corr, color=color, alpha=0.6, label=f'Trial {i + 1}')
        
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_ylim(-1, 1)  # Assuming correlation range
        ax.set_ylabel('Correlation')
        ax.set_title(f'Trial {i + 1}')
        ax.legend(loc='upper right')
    
    axs[-1].set_xlabel('Segment Index')
    fig.suptitle('Correlation between Activity and Stimulus Over Time', fontsize=16)
    plt.show()

def load_and_filter_data(filepath, genotype, duration, period_suffix, exclude_dates=None):
    with open(filepath, 'rb') as f:
        test_result = pickle.load(f)
    test_result['duration'] = duration
    filtered_experiments = [key for key in test_result if key not in exclude_dates and key.endswith(f"{genotype}_{duration}_{period_suffix}Period")]
    return test_result, filtered_experiments

def prepare_aggregated_data(test_result, filtered_experiments, tau, max_trials_limit=None):
    aggregated_data = []
    adjusted_stim_data = []
    
    for exp_key in filtered_experiments:
        experiment_data = test_result[exp_key]['data']
        if max_trials_limit and len(experiment_data) > max_trials_limit:
            experiment_data = experiment_data[:max_trials_limit]
        
        time_indices = np.arange(2400, 8400)  # Adjust this range as necessary for your specific data
        adjusted_stim_data = test_result[exp_key]['stimulus']
        
        for worm_data in experiment_data:
            aggregated_data.append(worm_data[time_indices])
    
    aggregated_data = np.array(aggregated_data)
    
    return aggregated_data, adjusted_stim_data, test_result, filtered_experiments


# def load_and_filter_data(filepath, genotype, duration, period_suffix, exclude_dates=None):
#     with open(filepath, 'rb') as f:
#         test_result = pickle.load(f)
#     test_result['duration'] = duration
#     filtered_experiments = [key for key in test_result if key not in exclude_dates and key.endswith(f"{genotype}_{duration}_{period_suffix}Period")]
#     return test_result, filtered_experiments

# def extract_active_stimulus_segment(stimulus_data, period_points):
#     start_index = np.argmax(stimulus_data > 0)
#     end_index = len(stimulus_data) - np.argmax(stimulus_data[::-1] > 0)
#     active_stimulus_segment = stimulus_data[start_index:end_index]
#     return active_stimulus_segment[:period_points]

# def compute_correlation(activity_data, stimulus_segment):
#     if np.std(activity_data) == 0 or np.std(stimulus_segment) == 0:
#         return np.nan
#     return np.corrcoef(activity_data, stimulus_segment)[0, 1]

# def slide_and_correlate(trial_data, stimulus_segment, period_points, slide_step):
#     correlations = []
#     for start in range(0, len(trial_data) - period_points + 1, slide_step):
#         activity_segment = trial_data[start:start + period_points]
#         if len(activity_segment) == period_points:
#             correlation = compute_correlation(activity_segment, stimulus_segment)
#             correlations.append(correlation)
#     return correlations

# def plot_correlations(correlations):
#     num_trials = len(correlations)
#     fig, axs = plt.subplots(num_trials, 1, figsize=(10, 2 * num_trials), sharex=True)
#     plt.subplots_adjust(hspace=0.5)
    
#     colors = plt.cm.viridis(np.linspace(0, 1, num_trials))
    
#     for i, (trial_corr, color) in enumerate(zip(correlations, colors)):
#         ax = axs[i]
#         ax.plot(range(len(trial_corr)), trial_corr, color=color, alpha=0.6, label=f'Trial {i + 1}')
        
#         ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
#         ax.set_ylim(-1, 1)
#         ax.set_ylabel('Correlation')
#         ax.set_title(f'Trial {i + 1}')
#         ax.legend(loc='upper right')
    
#     axs[-1].set_xlabel('Slide Index')
#     fig.suptitle('Correlation between Activity and Stimulus Over Time', fontsize=16)
#     plt.show()

# def prepare_aggregated_data(test_result, filtered_experiments, tau, max_trials_limit=None):
#     aggregated_data = []
#     valid_experiments = []

#     for exp_key in filtered_experiments[:max_trials_limit]:
#         experiment_data = test_result[exp_key]['data']
#         for worm_data in experiment_data:
#             data_length = len(worm_data)
#             if data_length >= 7200:  # Ensuring at least 7200 data points
#                 time_indices = np.arange(data_length)
#                 print(f"Experiment {exp_key} data length: {data_length}")
#                 aggregated_data.append(worm_data[time_indices])
#                 valid_experiments.append(exp_key)
#             else:
#                 print(f"Experiment {exp_key} has insufficient data points, skipping.")

#     aggregated_data = np.array(aggregated_data)
#     if not aggregated_data.size:
#         return None, None, None, None

#     first_exp_key = valid_experiments[0]
#     if 'stim' in test_result[first_exp_key]:
#         stim_data = test_result[first_exp_key]['stim']
#         adjusted_stim_data = (stim_data - np.min(stim_data)) / (np.max(stim_data) - np.min(stim_data)) * 1
    
#     return aggregated_data, tau[:len(time_indices)], adjusted_stim_data, stim_data

# def quantify_response(test_result, filtered_experiments, tau, period_suffix, max_trials_limit=None, slide_step=1):
#     period_points = int(float(period_suffix.replace('m', '')) * 60 * 2)
#     aggregated_data, _, adjusted_stim_data, _ = prepare_aggregated_data(test_result, filtered_experiments, tau, max_trials_limit)
    
#     if aggregated_data is None:
#         print("No valid aggregated data found.")
#         return []

#     stimulus_segment = extract_active_stimulus_segment(adjusted_stim_data, period_points)
    
#     all_correlations = []
#     for trial_data in aggregated_data:
#         trial_correlations = slide_and_correlate(trial_data.mean(axis=0), stimulus_segment, period_points, slide_step)
#         if trial_correlations:
#             all_correlations.append(trial_correlations)
#         else:
#             print(f"No valid correlations found for trial data with shape {trial_data.shape}")
    
#     return all_correlations
