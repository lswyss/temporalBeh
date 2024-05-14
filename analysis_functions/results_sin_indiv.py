#04.24.2024 Livia
#Functions for timeseries data on individual trials and worms

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tools.bootstrapTest import *
from tools.plotting_functions import *


def plot_worm_trials(experiment_name):
    """Load data and set up plot parameters, then call plotting function."""
    # Load your dataset
    with open('data/LDS_response_sinFunc_indiv.pickle', 'rb') as f:
        test_result = pickle.load(f)
    
    # Access the experiment data
    experiment_data = test_result[experiment_name]['data']
    stim_data = test_result[experiment_name]['stim']
    num_worms = len(experiment_data)
    max_trials = max([worm.shape[0] for worm in experiment_data])  # Determine the maximum number of trials any worm underwent
    
    # Find the index where the stimulus first becomes non-zero
    stim_start_index = np.argmax(stim_data > 0)

    # Time configuration
    frames_per_second = 2
    seconds_per_point = 1 / frames_per_second
    time_axis_seconds = (np.arange(stim_data.size) - stim_start_index) * seconds_per_point
    time_axis_minutes = time_axis_seconds / 60  # Convert seconds to minutes

    # Create a grid of plots with better spacing
    fig, axs = plt.subplots(nrows=num_worms, ncols=max_trials, figsize=(max_trials * 3, num_worms * 3), gridspec_kw={'hspace': 0.5, 'wspace': 0.3})
    
    # Call the plotting function
    gridplot_indiv_trials(axs, experiment_data, stim_data, time_axis_minutes, num_worms, max_trials)

    plt.tight_layout()
    plt.show()


