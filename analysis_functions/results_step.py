#this is in folder analysis_functions -> results_step.py
#Livia 07.20.2024
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tools.bootstrapTest import bootstrap_traces_sam

def response_step(interest_list, exclude, amplitude, duration, n_boot, statistic, conf_interval, t_samp):
    fig, ax_all = plt.subplots(nrows=len(interest_list), ncols=1, sharex=True, sharey=True, figsize=(15, 5 * len(interest_list)))
    ax_all = np.atleast_2d(ax_all).T

    with open('data/LDS_response_step.pickle', 'rb') as f:
        result = pickle.load(f)

    for i, interest in enumerate(interest_list):
        xp = result['tau']
        ind_t = np.where((xp >= t_samp[0] - 1e-5) & (xp <= t_samp[1]))[0]

        to_plot = data_of_interest(result.keys(), [f'{interest}'], exclude=exclude, duration=duration, amplitude=amplitude)

        if not to_plot:
            print(f"No data to plot for {interest} with amplitude '{amplitude}'.")
            continue

        print(f"Plotting experiments for interest '{interest}' with amplitude '{amplitude}':")
        for exp in to_plot:
            print(f"  - {exp}")

        yp = np.concatenate([result[dat]['data'] for dat in to_plot])
        y, rng = bootstrap_traces_sam(yp[:, ind_t], n_boot=n_boot, statistic=statistic, conf_interval=conf_interval)
        stim_data = result[to_plot[0]]['stim'][ind_t]

        plot_step(ax_all[i, 0], xp[ind_t], y, rng, f'{interest}, {amplitude}, ({yp.shape[0]})', color='cornflowerblue', plot_stim=True, stim_data=stim_data)

    plt.show()
    return fig, ax_all

def data_of_interest(keys, interests, exclude, duration, amplitude):
    """
    Filter keys based on the interests, exclude list, duration, and amplitude.
    """
    filtered = []
    for key in keys:
        if any(ex in key for ex in exclude):
            continue
        parts = key.split('_')
        if len(parts) == 3:
            if any(interest in parts[0] for interest in interests) and parts[1] == amplitude and parts[2] == duration:
                filtered.append(key)
    return filtered

def plot_step(ax, x, y, rng, label, color='cornflowerblue', plot_stim=False, stim_data=None):
    """
    Plot the step function response.
    """
    ax.plot(x, y, label=label, color=color)
    ax.fill_between(x, rng[0], rng[1], color=color, alpha=0.5)
    if plot_stim and stim_data is not None:
        ax.plot(x, stim_data, color='darkorange', label='Stimulus')
    ax.legend()
     # Set y-axis limits
    ax.set_ylim(0, 1.5)