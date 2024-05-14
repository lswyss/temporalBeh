#04.24.2024 Livia
#Functions for timeseries data

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tools.bootstrapTest import *
from tools.plotting_functions import plot_sin

# def data_of_interest(names, interest=[], exclude=[]):
#     to_plot = set()
#     exclude = set(exclude)
#     full_amp = all('127amp' in i or 'amp' not in i for i in interest)
#     if full_amp:
#         exclude.add('amp')
#     for dat in names:
#         if any(i in dat for i in interest) and not any(ex in dat for ex in exclude) and all(dat.count('+') <= i.count('+') for i in interest):
#             to_plot.add(dat)
#     return list(to_plot)

# def calc_delta(x, w=2):
#     if w % 2 == 0:
#         w += 1
#     filt = np.concatenate((-np.ones(w // 2), [0], np.ones(w // 2)))
#     return np.convolve(x, filt, mode='same') / w

# def response_sin(interest_list, exclude, periods=[ 3, 4, 5,], duration=30,
#                  n_boot=1e3, statistic=np.median,
#                 baseline=128, conf_interval=95, t_samp=(-3, 40), ):
#     fig, ax_all = plt.subplots(nrows=len(periods), ncols=1, sharex=True, sharey=True, figsize=(10, 2 * len(periods)))
#     if len(periods) == 1:
#         ax_all = np.array([ax_all])
#     if True:
#         ax_all = ax_all[:, None]

#     for ii, interest in enumerate(interest_list):
#         name = '../data/LDS_response_sinFunc.pickle'
#         with open(name, 'rb') as f:
#             result = pickle.load(f)

#         for i, p in enumerate(periods):
#             xp = result['tau']
#             ind_t = np.where((xp >= t_samp[0] - 1e-5) & (xp <= t_samp[1]))[0]
#             p_name = f'{p}m'
#             to_plot = data_of_interest(result.keys(), [f'{interest}_{duration}m_{p_name}'], exclude=exclude)

#             if len(to_plot) == 0 and (p < 1 or p % 1 > 0):
#                 p_name = f'{int(p * 60)}s'
#                 to_plot = data_of_interest(result.keys(), [f'{interest}_{duration}m_{p_name}'], exclude=exclude)

#             if len(to_plot) == 0:
#                 print(f"No data to plot for {interest} at {p}m period.")
#                 continue

#             yp = np.concatenate([result[dat]['data'] for dat in to_plot])
#             y, rng = bootstrap_traces(yp[:, ind_t], n_boot=n_boot, statistic=statistic, conf_interval=conf_interval)
#             stim_data = result[to_plot[0]]['stim'][ind_t]  # Apply slicing to align the stimulus data.

#             plot_sin(ax_all[i, 0], xp[ind_t], y, rng, f'{interest} {p}m, ({yp.shape[0]})', color='cornflowerblue', plot_stim=True, stim_data=stim_data)

#             ax_all[i, 0].legend()

#     plt.show()
#     return fig, ax_all


import numpy as np
import matplotlib.pyplot as plt
import pickle
from tools.bootstrapTest import bootstrap_traces
from tools.plotting_functions import plot_sin

# Load the data once outside the function
def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Optimize the data filtering
def data_of_interest(names, interest=[], exclude=[]):
    to_plot = []
    exclude = set(exclude)
    full_amp = all('127amp' in i or 'amp' not in i for i in interest)
    if full_amp:
        exclude.add('amp')
    for dat in names:
        if any(i in dat for i in interest) and not any(ex in dat for ex in exclude) and all(dat.count('+') <= i.count('+') for i in interest):
            to_plot.append(dat)
    return to_plot

def response_sin(dataset, interest_list, exclude, periods, n_boot, statistic, baseline, conf_interval, t_samp):

    # Initialize subplots
    fig, ax_all = plt.subplots(nrows=len(periods), ncols=1, sharex=True, sharey=True, figsize=(10, 2 * len(periods)))
    ax_all = np.atleast_2d(ax_all).reshape(-1, 1)  # Ensure ax_all is always 2D

    xp = dataset['tau']
    ind_t = np.where((xp >= t_samp[0] - 1e-5) & (xp <= t_samp[1]))[0]

    for ii, interest in enumerate(interest_list):
        for i, p in enumerate(periods):
            p_name = f'{p}m'
            to_plot = data_of_interest(dataset.keys(), [f'{interest}_{duration}m_{p_name}'], exclude=exclude)

            if len(to_plot) == 0:
                print(f"No data to plot for {interest} at {p}m period.")
                continue

            yp = np.concatenate([result[dat]['data'] for dat in to_plot])
            y, rng = bootstrap_traces(yp[:, ind_t], n_boot=n_boot, statistic=statistic, conf_interval=conf_interval)
            stim_data = result[to_plot[0]]['stim'][ind_t]  # Apply slicing to align the stimulus data.

            plot_sin(ax_all[i, 0], xp[ind_t], y, rng, f'{interest} {p}m, ({yp.shape[0]})', color='cornflowerblue', plot_stim=True, stim_data=stim_data)
            ax_all[i, 0].legend()

    plt.show()
    return fig, ax_all

# Usage
if __name__ == "__main__":
    filepath = '../data/LDS_response_sinFunc.pickle'
    result = load_data(filepath)