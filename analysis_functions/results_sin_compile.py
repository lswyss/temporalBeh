#04.24.2024 Livia
# in fodler analysis_functions script results_sin_compile.py
#Functions for timeseries data

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tools.bootstrapTest import bootstrap_traces_sam
from tools.plotting_functions import plot_sin


# def data_of_interest(names, interest=[], exclude=[], duration='30m', period=None):
#     to_plot = set()
#     exclude_set = set(exclude)
    
#     for dat in names:
#         if dat in exclude_set:
#             continue
        
#         parts = dat.split('_')
#         experiment_interest = parts[0]
#         experiment_duration = None
#         experiment_period = parts[-1]
        
#         for part in parts:
#             if part == duration:
#                 experiment_duration = part
#                 break
        
#         if not any(i in experiment_interest for i in interest):
#             continue
        
#         if experiment_duration != duration:
#             continue
        
#         if period and experiment_period != period:
#             continue
        
#         to_plot.add(dat)
    
#     return list(to_plot)


# def response_sin(interest_list, exclude, periods, duration, n_boot, statistic, conf_interval, t_samp):
#     fig, ax_all = plt.subplots(nrows=len(periods), ncols=1, sharex=True, sharey=True, figsize=(10, 2 * len(periods)))
#     ax_all = np.atleast_2d(ax_all).T

#     with open('data/LDS_response_sinFunc.pickle', 'rb') as f:
#         result = pickle.load(f)

#     for i, p in enumerate(periods):
#         xp = result['tau']
#         ind_t = np.where((xp >= t_samp[0] - 1e-5) & (xp <= t_samp[1]))[0]
#         p_name = f'{p}mPeriod'

#         for interest in interest_list:
#             to_plot = data_of_interest(result.keys(), [f'{interest}'], exclude=exclude, duration=duration, period=p_name)

#             if not to_plot:
#                 print(f"No data to plot for {interest} at {p}m period.")
#                 continue

#             # Print the experiments being plotted
#             print(f"Plotting experiments for interest '{interest}' at {p}m period:")
#             for exp in to_plot:
#                 print(f"  - {exp}")

#             yp = np.concatenate([result[dat]['data'] for dat in to_plot])
#             y, rng = bootstrap_traces_sam(yp[:, ind_t], n_boot=n_boot, statistic=statistic, conf_interval=conf_interval)
#             stim_data = result[to_plot[0]]['stim'][ind_t]

#             plot_sin(ax_all[i, 0], xp[ind_t], y, rng, f'{interest} {p}m, ({yp.shape[0]})', color='cornflowerblue', plot_stim=True, stim_data=stim_data)

#     plt.show()
#     return fig, ax_all

# def response_sin_with_extracted_periods(interest_list, exclude, periods, duration, n_boot, statistic, conf_interval, t_samp):
#     fig_main, ax_main = plt.subplots(nrows=len(periods), ncols=1, sharex=True, figsize=(15, 5 * len(periods)))
    
#     with open('data/LDS_response_sinFunc.pickle', 'rb') as f:
#         result = pickle.load(f)

#     if len(periods) == 1:
#         ax_main = [ax_main]

#     for i, p in enumerate(periods):
#         xp = result['tau']
#         ind_t = np.where((xp >= t_samp[0] - 1e-5) & (xp <= t_samp[1]))[0]
#         p_name = f'{p}mPeriod'
        
#         for interest in interest_list:
#             to_plot = data_of_interest(result.keys(), [f'{interest}'], exclude=exclude, duration=duration, period=p_name)

#             if not to_plot:
#                 print(f"No data to plot for {interest} at {p}m period.")
#                 continue

#             print(f"Plotting experiments for interest '{interest}' at {p}m period:")
#             for exp in to_plot:
#                 print(f"  - {exp}")

#             yp = np.concatenate([result[dat]['data'] for dat in to_plot])
#             y, rng = bootstrap_traces_sam(yp[:, ind_t], n_boot=n_boot, statistic=statistic, conf_interval=conf_interval)
#             stim_data = result[to_plot[0]]['stim'][ind_t]

#             period_length = p  # in minutes
#             period_indices = extract_period_indices(xp[ind_t], period_length)
#             period_colors = ['lightcoral', 'lightseagreen', 'lightsteelblue']
            
#             plot_sin(ax_main[i], xp[ind_t], y, rng, f'{interest} {p}m, ({yp.shape[0]})', color='cornflowerblue', plot_stim=True, stim_data=stim_data)

#             # Add boxes to indicate the extracted periods
#             for idx, (period_name, (start_idx, end_idx)) in enumerate(period_indices.items()):
#                 ax_main[i].axvspan(xp[ind_t][start_idx], xp[ind_t][end_idx], color=period_colors[idx], alpha=0.3)

#     fig_periods, ax_periods = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
#     plot_extracted_periods(ax_periods, xp[ind_t], yp[:, ind_t], stim_data, period_indices, period_colors)
    
#     plt.show()
#     return fig_main, ax_main, fig_periods, ax_periods

# def extract_period_indices(tau, period_length):
#     period_indices = {}
#     period_length_samples = int(period_length * (len(tau) / (tau[-1] - tau[0])))
    
#     period_indices['first'] = (0, period_length_samples)
#     period_indices['fourth'] = (3 * period_length_samples, 4 * period_length_samples)
#     period_indices['ninth'] = (8 * period_length_samples, 9 * period_length_samples)
    
#     return period_indices

# def plot_extracted_periods(ax, tau, data, stim_data, period_indices, period_colors):
#     period_names = ['first', 'fourth', 'ninth']
    
#     for idx, period_name in enumerate(period_names):
#         start_idx, end_idx = period_indices[period_name]
        
#         sliced_tau = tau[start_idx:end_idx]
#         sliced_data = data[:, start_idx:end_idx]
#         sliced_stim_data = stim_data[start_idx:end_idx]
        
#         mean_data = np.mean(sliced_data, axis=0)
        
#         ax[idx].plot(sliced_stim_data, mean_data, color=period_colors[idx], label=f'{period_name.capitalize()} period')
#         ax[idx].set_xlabel('Stimulation Intensity')
#         ax[idx].set_ylabel('Activity')
#         ax[idx].legend()
    
#     plt.tight_layout()

def data_of_interest(names, interest=[], exclude=[], duration='30m', period=None, amplitude='127amp'):
    to_plot = set()
    exclude_set = set(exclude)

    for dat in names:
        if dat in exclude_set:
            continue

        parts = dat.split('_')
        experiment_interest = parts[0]
        experiment_duration = None
        experiment_period = parts[-1]
        experiment_amplitude = None

        for part in parts:
            if part == duration:
                experiment_duration = part
            if 'amp' in part:
                experiment_amplitude = part

        if not experiment_amplitude:
            experiment_amplitude = '127amp'

        if not any(i in experiment_interest for i in interest):
            continue

        if experiment_duration != duration:
            continue

        if period and experiment_period != period:
            continue

        if amplitude and experiment_amplitude != amplitude:
            continue

        to_plot.add(dat)

    return list(to_plot)

def response_sin(interest_list, exclude, periods, duration, n_boot, statistic, conf_interval, t_samp, amplitudes=None):
    fig, ax_all = plt.subplots(nrows=len(periods), ncols=1, sharex=True, sharey=True, figsize=(15, 5 * len(periods)))
    ax_all = np.atleast_2d(ax_all).T

    with open('data/LDS_response_sinFunc.pickle', 'rb') as f:
        result = pickle.load(f)

    if amplitudes is None:
        amplitudes = ['127amp']  # Default to '127amp' if no amplitudes are specified
    elif isinstance(amplitudes, str):
        amplitudes = [amplitudes]  # Convert to list if a single amplitude is provided

    for i, p in enumerate(periods):
        xp = result['tau']
        ind_t = np.where((xp >= t_samp[0] - 1e-5) & (xp <= t_samp[1]))[0]
        p_name = f'{p}mPeriod'

        for interest in interest_list:
            for amplitude in amplitudes:
                to_plot = data_of_interest(result.keys(), [f'{interest}'], exclude=exclude, duration=duration, period=p_name, amplitude=amplitude)

                if not to_plot:
                    print(f"No data to plot for {interest} at {p}m period with amplitude '{amplitude}'.")
                    continue

                print(f"Plotting experiments for interest '{interest}' at {p}m period with amplitude '{amplitude}':")
                for exp in to_plot:
                    print(f"  - {exp}")

                yp = np.concatenate([result[dat]['data'] for dat in to_plot])
                y, rng = bootstrap_traces_sam(yp[:, ind_t], n_boot=n_boot, statistic=statistic, conf_interval=conf_interval)
                stim_data = result[to_plot[0]]['stim'][ind_t]

                plot_sin(ax_all[i, 0], xp[ind_t], y, rng, f'{interest} {p}m, {amplitude}, ({yp.shape[0]})', color='cornflowerblue', plot_stim=True, stim_data=stim_data)

    plt.show()
    return fig, ax_all

def extract_period_indices(tau, period_length):
    period_indices = {}
    period_length_samples = int(period_length * (len(tau) / (tau[-1] - tau[0])))

    period_indices['first'] = (0, period_length_samples)
    period_indices['third'] = (2 * period_length_samples, 3 * period_length_samples)
    period_indices['eight'] = (7 * period_length_samples, 8 * period_length_samples)

    return period_indices

def plot_extracted_periods(ax, tau, data, stim_data, period_indices, period_colors):
    period_names = ['first', 'third', 'eight']

    for idx, period_name in enumerate(period_names):
        start_idx, end_idx = period_indices[period_name]

        sliced_tau = tau[start_idx:end_idx]
        sliced_data = data[:, start_idx:end_idx]
        sliced_stim_data = stim_data[start_idx:end_idx]

        mean_data = np.mean(sliced_data, axis=0)

        ax[idx].plot(sliced_stim_data, mean_data, color=period_colors[idx], label=f'{period_name.capitalize()} period')
        ax[idx].set_xlabel('Stimulation Intensity')
        ax[idx].set_ylabel('Activity')
        ax[idx].legend()

    plt.tight_layout()

def response_sin_with_extracted_periods(interest_list, exclude, periods, duration, n_boot, statistic, conf_interval, t_samp, amplitudes=None):
    fig_main, ax_main = plt.subplots(nrows=len(periods), ncols=1, sharex=True, figsize=(15, 5 * len(periods)))

    with open('data/LDS_response_sinFunc.pickle', 'rb') as f:
        result = pickle.load(f)

    if len(periods) == 1:
        ax_main = [ax_main]

    if amplitudes is None:
        amplitudes = ['127amp']  # Default to '127amp' if no amplitudes are specified
    elif isinstance(amplitudes, str):
        amplitudes = [amplitudes]  # Convert to list if a single amplitude is provided

    # Ensure the default amplitude '127amp' is always plotted if no specific amplitudes are provided
    if '127amp' not in amplitudes:
        amplitudes.append('127amp')

    for amplitude in amplitudes:
        for i, p in enumerate(periods):
            xp = result['tau']
            ind_t = np.where((xp >= t_samp[0] - 1e-5) & (xp <= t_samp[1]))[0]
            p_name = f'{p}mPeriod'

            for interest in interest_list:
                to_plot = data_of_interest(result.keys(), [f'{interest}'], exclude=exclude, duration=duration, period=p_name, amplitude=amplitude)

                if not to_plot:
                    print(f"No data to plot for {interest} at {p}m period with amplitude '{amplitude}'.")
                    continue

                print(f"Plotting experiments for interest '{interest}' at {p}m period with amplitude '{amplitude}':")
                for exp in to_plot:
                    print(f"  - {exp}")

                yp = np.concatenate([result[dat]['data'] for dat in to_plot])
                y, rng = bootstrap_traces_sam(yp[:, ind_t], n_boot=n_boot, statistic=statistic, conf_interval=conf_interval)
                stim_data = result[to_plot[0]]['stim'][ind_t]

                period_length = p  # in minutes
                period_indices = extract_period_indices(xp[ind_t], period_length)
                period_colors = ['lightcoral', 'lightseagreen', 'lightsteelblue']

                # Plot trace with confidence intervals
                plot_sin(ax_main[i], xp[ind_t], y, rng, f'{interest} {p}m, {amplitude}, ({yp.shape[0]})', color='cornflowerblue', plot_stim=True, stim_data=stim_data)

                # Add boxes to indicate the extracted periods
                for idx, (period_name, (start_idx, end_idx)) in enumerate(period_indices.items()):
                    ax_main[i].axvspan(xp[ind_t][start_idx], xp[ind_t][end_idx], color=period_colors[idx], alpha=0.3)

        # Plot the extracted periods without confidence intervals
        fig_periods, ax_periods = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        plot_extracted_periods(ax_periods, xp[ind_t], yp[:, ind_t], stim_data, period_indices, period_colors)

    plt.show()
    return fig_main, ax_main, fig_periods, ax_periods
