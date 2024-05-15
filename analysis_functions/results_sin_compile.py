#04.24.2024 Livia
# in fodler analysis_functions script results_sin_compile.py
#Functions for timeseries data

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tools.bootstrapTest import bootstrap_traces
from tools.plotting_functions import plot_sin

def data_of_interest(names, interest=[], exclude=[], duration='30m'):
    to_plot = set()
    exclude = set(exclude)
    full_amp = all('127amp' in i or 'amp' not in i for i in interest)
    if full_amp:
        exclude.add('amp')
    for dat in names:
        if any(i in dat for i in interest) and not any(ex in dat for ex in exclude):
            parts = dat.split('_')
            if len(parts) > 1 and parts[1] == duration:
                to_plot.add(dat)
    return list(to_plot)

def response_sin(interest_list, exclude, periods, duration, n_boot, statistic, conf_interval, t_samp):
    fig, ax_all = plt.subplots(nrows=len(periods), ncols=1, sharex=True, sharey=True, figsize=(10, 2 * len(periods)))
    ax_all = np.atleast_2d(ax_all).T

    with open('data/LDS_response_sinFunc.pickle', 'rb') as f:
        result = pickle.load(f)

    for i, p in enumerate(periods):
        xp = result['tau']
        ind_t = np.where((xp >= t_samp[0] - 1e-5) & (xp <= t_samp[1]))[0]
        p_name = f'{p}mPeriod'

        for interest in interest_list:
            to_plot = data_of_interest(result.keys(), [f'{interest}_{duration}_{p_name}'], exclude=exclude)

            if not to_plot:
                print(f"No data to plot for {interest} at {p}m period.")
                continue

            yp = np.concatenate([result[dat]['data'] for dat in to_plot])
            y, rng = bootstrap_traces(yp[:, ind_t], n_boot=n_boot, statistic=statistic, conf_interval=conf_interval)
            stim_data = result[to_plot[0]]['stim'][ind_t]

            plot_sin(ax_all[i, 0], xp[ind_t], y, rng, f'{interest} {p}m, ({yp.shape[0]})', color='cornflowerblue', plot_stim=True, stim_data=stim_data)

    plt.show()
    return fig, ax_all