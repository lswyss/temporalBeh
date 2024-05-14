import pickle
import numpy as np
import matplotlib.pyplot as plt
from tools.filtering import filter_experiments
from tools.plotting_functions import plot_combined_heatmap

def load_data(filepath):
    """Load experiment data from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def aggregate_heatmap(test_result, genotype, duration, period_suffix, exclude_dates=None):
    """
    Aggregate trial data and plot a heatmap based on specified filters, and print the number of worms in each trial.

    Args:
        test_result (dict): Data containing experiments.
        genotype (str): Genotype to filter by.
        duration (str): Duration to filter by.
        period_suffix (str): Period suffix to filter by.
        exclude_dates (list, optional): List of date prefixes to exclude. Defaults to None.
    """
    # Filter experiments based on provided criteria
    filtered_experiments = filter_experiments(test_result, genotype, duration, period_suffix, exclude_dates)

    if not filtered_experiments:
        print("No experiments matched the criteria.")
        return None, None

    all_trial_averages = []
    worm_counts = []
    max_trials = 0

    # Find the maximum number of trials across the experiments
    for name in filtered_experiments:
        experiment_data = test_result[name]['data']
        max_trials = max(max_trials, max(worm.shape[0] for worm in experiment_data))

    # Aggregate trial data and count the number of worms per trial
    for trial_idx in range(max_trials):
        trial_data = [worm[trial_idx, :] for name in filtered_experiments
                      for worm in test_result[name]['data'] if worm.shape[0] > trial_idx]
        
        if trial_data:
            all_trial_averages.append(np.mean(trial_data, axis=0))
            worm_counts.append(len(trial_data))

    if not all_trial_averages:
        print("No data available to plot.")
        return None, None

    # Print the number of worms in each trial
    for idx, count in enumerate(worm_counts, start=1):
        print(f"Trial {idx}: {count} worms")

    # Time axis calculation
    stim_data = test_result[filtered_experiments[0]]['stim']
    stim_start_index = np.argmax(stim_data > 0)
    frames_per_second = 2
    seconds_per_point = 1 / frames_per_second
    time_axis_seconds = (np.arange(len(all_trial_averages[0])) - stim_start_index) * seconds_per_point
    time_axis_minutes = time_axis_seconds / 60

    # Debugging: Print time axis and trial averages shape
    print("Time axis (minutes):", time_axis_minutes)
    print("Shape of trial averages:", np.array(all_trial_averages).shape)

    plot_combined_heatmap(all_trial_averages, time_axis_minutes, f"Aggregated Trials Across Experiments: {genotype}, {duration}, {period_suffix}")

    return all_trial_averages, time_axis_minutes
