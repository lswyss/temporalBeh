import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from analysis_functions.results_sin_compile import response_sin

# Updated function with normalization based on the periods
def calculate_phase_shifts(interest_list, exclude, periods, duration, n_boot, statistic, conf_interval, t_samp, amplitudes=None, focus_range=None, smooth_sigma=15, save_svg=False):
    """
    Calculate the phase shift between stimulus and activity data, both for peaks and troughs,
    and normalize them by the provided periods.
    
    :param periods: List of periods used for normalizing phase shifts.
    """
    result_data = response_sin(interest_list, exclude, periods, duration, n_boot, statistic, conf_interval, t_samp, amplitudes, return_data=True)
    
    phase_shifts_peaks = {}
    phase_shifts_troughs = {}
    all_peak_shifts = []
    all_trough_shifts = []
    period_labels = []

    for i, (key, data) in enumerate(result_data.items()):
        xp, stim_data, activity_data = data['xp'], data['stim'], data['activity']
        period = periods[i]  # Get the corresponding period for this experiment

        if focus_range:
            start_idx, end_idx = np.searchsorted(xp, focus_range)
            xp, stim_data, activity_data = xp[start_idx:end_idx], stim_data[start_idx:end_idx], activity_data[start_idx:end_idx]

        # Peaks
        stim_peaks = find_peaks(stim_data)[0][-3:]  # Last 4 stimulus peaks
        smoothed_activity = gaussian_filter1d(activity_data, sigma=smooth_sigma * 2)
        smoothed_activity = gaussian_filter1d(smoothed_activity, sigma=smooth_sigma)
        activity_peaks = find_peaks(smoothed_activity)[0]

        peak_diffs = []
        peak_pairs = []
        for stim_peak in stim_peaks:
            window = 200
            possible_peaks = [p for p in activity_peaks if abs(p - stim_peak) <= window]
            if possible_peaks:
                closest_activity_peak = min(possible_peaks, key=lambda x: abs(x - stim_peak))
                peak_diff = xp[closest_activity_peak] - xp[stim_peak]
                normalized_peak_diff = peak_diff / period  # Normalize the peak shift by the period
                peak_diffs.append(normalized_peak_diff)
                peak_pairs.append((stim_peak, closest_activity_peak))
                #all_peak_shifts.append(normalized_peak_diff)# Convert normalized values to radians before appending
                all_peak_shifts.append(normalized_peak_diff * 2 * np.pi)

                period_labels.append(key)

        phase_shifts_peaks[key] = peak_diffs

        # Troughs (find deepest troughs by finding peaks in inverted data)
        stim_troughs = find_peaks(-stim_data)[0][-4:]  # Last 5 stimulus troughs
        activity_troughs, _ = find_peaks(-smoothed_activity)

        # Find the deepest troughs based on smoothed_activity values
        sorted_activity_troughs = sorted(activity_troughs, key=lambda x: smoothed_activity[x])  # Sort by depth (lowest value)
        selected_troughs = sorted_activity_troughs[:3]  # Select the 5 deepest troughs

        trough_diffs = []
        trough_pairs = []
        for stim_trough in stim_troughs:
            window = 200
            possible_troughs = [p for p in selected_troughs if abs(p - stim_trough) <= window]
            if possible_troughs:
                closest_activity_trough = min(possible_troughs, key=lambda x: abs(x - stim_trough))
                trough_diff = xp[closest_activity_trough] - xp[stim_trough]
                normalized_trough_diff = trough_diff / period  # Normalize the trough shift by the period
                trough_diffs.append(normalized_trough_diff)
                trough_pairs.append((stim_trough, closest_activity_trough))
                #all_trough_shifts.append(normalized_trough_diff)
                all_trough_shifts.append(normalized_trough_diff * 2 * np.pi)


        print(f"Trough phase differences for {key}: {trough_diffs}")

        phase_shifts_troughs[key] = trough_diffs

        # Call plotting function for this experiment
        #plot_experiment(xp, stim_data, smoothed_activity, stim_peaks, activity_peaks, stim_troughs, selected_troughs, peak_pairs, trough_pairs, key)
        plot_experiment(xp, stim_data, smoothed_activity, stim_peaks, activity_peaks, stim_troughs, activity_troughs, peak_pairs, trough_pairs, "UVandVisExperiment", save_svg=True)


    plot_phase_shifts(period_labels, all_peak_shifts, all_trough_shifts, save_svg)
    return phase_shifts_peaks, phase_shifts_troughs


# Plot for individual experiments, showing peaks and troughs, along with dashed lines between paired peaks and troughs
def plot_experiment(xp, stim_data, smoothed_activity, stim_peaks, activity_peaks, stim_troughs, activity_troughs, peak_pairs, trough_pairs, title, save_svg=False):
    plt.figure(figsize=(10, 5))

    # Plot stimulus and smoothed activity
    plt.plot(xp, stim_data, label='Stimulus', color='darkorange')
    plt.plot(xp, smoothed_activity, label='Smoothed Activity', color='blue')

    # Plot peaks
    plt.scatter(xp[stim_peaks], stim_data[stim_peaks], color='red', marker='o', label='Stimulus Peaks')
    plt.scatter(xp[activity_peaks], smoothed_activity[activity_peaks], color='green', marker='x', label='Activity Peaks')

    # Plot troughs
    plt.scatter(xp[stim_troughs], stim_data[stim_troughs], color='purple', marker='o', label='Stimulus Troughs')
    plt.scatter(xp[activity_troughs], smoothed_activity[activity_troughs], color='yellow', marker='x', label='Activity Troughs')

    # Draw dashed lines for paired peaks
    for stim_peak, activity_peak in peak_pairs:
        plt.plot([xp[stim_peak], xp[activity_peak]], [stim_data[stim_peak], smoothed_activity[activity_peak]], 'k--', label='Peak Pair')

    # Draw dashed lines for paired troughs
    for stim_trough, activity_trough in trough_pairs:
        plt.plot([xp[stim_trough], xp[activity_trough]], [stim_data[stim_trough], smoothed_activity[activity_trough]], 'r--', label='Trough Pair')

    plt.xlabel('Time (min)')
    plt.ylabel('Response')
    plt.title(f'Phase Shift Analysis for {title}')
    plt.legend(frameon=False, loc='best')

    # Save as SVG if the option is set to True
    if save_svg:
        plt.savefig(f'{title}_phase_shift_analysis.svg', format='svg')
        print(f"Plot saved as {title}_phase_shift_analysis.svg")

    plt.show()

# Plot phase shifts (box plots) for both peaks and troughs with a specified order
def plot_phase_shifts(period_labels, all_peak_shifts, all_trough_shifts, save_svg=False):
    """
    Plot phase shifts as box plots for both peaks and troughs in the specified order.

    :param period_labels: Labels for each period (experiment key).
    :param all_peak_shifts: Calculated phase shifts for peaks.
    :param all_trough_shifts: Calculated phase shifts for troughs.
    :param save_svg: Boolean to save plots as SVG.
    """
    # Custom order for the box plots
    custom_order = ['2.3m', '2.7m', '3m', '3.2m', '3.4m', '3.8m', '4m']

    # Initialize dictionaries to store shifts in the custom order
    shifts_per_label_peaks = {label: [] for label in custom_order}
    shifts_per_label_troughs = {label: [] for label in custom_order}

    # Normalize period_labels to match custom_order
    normalized_period_labels = [label.split('_')[-1].replace('Period', '') for label in period_labels]

    print("Normalized Period Labels:", normalized_period_labels)
    print("All Peak Shifts:", all_peak_shifts)
    print("All Trough Shifts:", all_trough_shifts)


    # Populate shifts for peaks and troughs according to the custom order
    for label, shift in zip(normalized_period_labels, all_peak_shifts):
        if label in shifts_per_label_peaks:
            shifts_per_label_peaks[label].append(shift)
    for label, shift in zip(normalized_period_labels, all_trough_shifts):
        if label in shifts_per_label_troughs:
            shifts_per_label_troughs[label].append(shift)

    # Prepare data for box plot in the specified order
    box_data_peaks = [shifts_per_label_peaks[label] for label in custom_order]
    box_data_troughs = [shifts_per_label_troughs[label] for label in custom_order]
    box_positions = np.arange(len(custom_order))

    print("Shifts per Label (Peaks):", shifts_per_label_peaks)
    print("Shifts per Label (Troughs):", shifts_per_label_troughs)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot for peaks
    ax.boxplot(box_data_peaks, positions=box_positions - 0.15, widths=0.3, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'), medianprops=dict(color='black'), zorder=2, capprops={'color': 'black'})

    # Box plot for troughs
    ax.boxplot(box_data_troughs, positions=box_positions + 0.15, widths=0.3, patch_artist=True,
               boxprops=dict(facecolor='lightcoral', color='black'), medianprops=dict(color='black'), zorder=2, capprops={'color': 'black'})

    # Add jittered points for peaks
    jitter_strength = 0.05
    for i, label in enumerate(custom_order):
        individual_peaks = shifts_per_label_peaks[label]
        jittered_positions = box_positions[i] - 0.15 + np.random.uniform(-jitter_strength, jitter_strength, size=len(individual_peaks))
        ax.scatter(jittered_positions, individual_peaks, color='#1E90FF', alpha=0.8, edgecolor='k', s=100, zorder=3)

    # Add jittered points for troughs
    for i, label in enumerate(custom_order):
        individual_troughs = shifts_per_label_troughs[label]
        jittered_positions = box_positions[i] + 0.15 + np.random.uniform(-jitter_strength, jitter_strength, size=len(individual_troughs))
        ax.scatter(jittered_positions, individual_troughs, color='#FF4500', alpha=0.8, edgecolor='k', s=100, zorder=3)

    # Set custom order for x-axis labels
    ax.set_xticks(box_positions)
    ax.set_xticklabels(custom_order, rotation=45, fontsize=12)
    
    # Define custom y-ticks in radians and their corresponding π/x labels
    y_ticks = [np.pi / 7, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
    y_tick_labels = [r'$\pi/7$', r'$\pi/6$', r'$\pi/4$', r'$\pi/3$', r'$\pi/2$']

    # Set the custom y-axis ticks
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=12)

    ax.set_xlabel('Experiment/Frequency', fontsize=14)
    ax.set_ylabel('Normalized Phase Shift', fontsize=14)
    ax.set_title('Normalized Phase Shift (Peaks and Troughs) Across Different Frequencies', fontsize=16)

    plt.tight_layout()
    if save_svg:
        plt.savefig('combined_phase_shift_peaks_troughs_normalized.svg', format='svg')
    plt.show()

#______________________________________________________________________________________________________________________________________________________________
''' This is UV & Vis plots'''
#
def calculate_phase_shifts_UVandVis(interest_list, exclude, periods, duration, n_boot, statistic, conf_interval, t_samp, amplitudes=None, focus_range=None, smooth_sigma=15, save_svg=False):
    """
    Calculate the phase shift between stimulus and activity data, both for peaks and troughs,
    normalized by the provided periods, specifically for a single UV or Vis experiment.
    """
    result_data = response_sin(interest_list, exclude, periods, duration, n_boot, statistic, conf_interval, t_samp, amplitudes, return_data=True)
    
    phase_shifts_peaks = {}
    phase_shifts_troughs = {}
    all_peak_shifts = []
    all_trough_shifts = []
    period_labels = []

    for i, (key, data) in enumerate(result_data.items()):
        xp, stim_data, activity_data = data['xp'], data['stim'], data['activity']
        period = periods[i]

        # Apply focus range if specified
        if focus_range:
            start_idx, end_idx = np.searchsorted(xp, focus_range)
            xp, stim_data, activity_data = xp[start_idx:end_idx], stim_data[start_idx:end_idx], activity_data[start_idx:end_idx]

        # Find the last 3 stimulus peaks
        stim_peaks = find_peaks(stim_data)[0][-3:]
        
        # Smooth activity data and find peaks and troughs
        smoothed_activity = gaussian_filter1d(activity_data, sigma=smooth_sigma * 2)
        smoothed_activity = gaussian_filter1d(smoothed_activity, sigma=smooth_sigma)
        activity_peaks = find_peaks(smoothed_activity)[0]
        
        # Calculate phase shifts for peaks
        peak_diffs = []
        peak_pairs = []
        for stim_peak in stim_peaks:
            window = 200
            possible_peaks = [p for p in activity_peaks if abs(p - stim_peak) <= window]
            if possible_peaks:
                closest_activity_peak = min(possible_peaks, key=lambda x: abs(x - stim_peak))
                peak_diff = xp[closest_activity_peak] - xp[stim_peak]
                normalized_peak_diff = peak_diff / period * 2 * np.pi  # Convert to radians
                peak_diffs.append(normalized_peak_diff)
                peak_pairs.append((stim_peak, closest_activity_peak))
                all_peak_shifts.append(normalized_peak_diff)
                period_labels.append(key)

        phase_shifts_peaks[key] = peak_diffs

        # Find the last 3 stimulus troughs
        stim_troughs = find_peaks(-stim_data)[0][-3:]
        activity_troughs, _ = find_peaks(-smoothed_activity)
        
        # Get the deepest activity troughs
        sorted_activity_troughs = sorted(activity_troughs, key=lambda x: smoothed_activity[x])
        selected_troughs = sorted_activity_troughs[:4]

        # Calculate phase shifts for troughs
        trough_diffs = []
        trough_pairs = []
        for stim_trough in stim_troughs:
            possible_troughs = [p for p in selected_troughs if abs(p - stim_trough) <= window]
            if possible_troughs:
                closest_activity_trough = min(possible_troughs, key=lambda x: abs(x - stim_trough))
                trough_diff = xp[closest_activity_trough] - xp[stim_trough]
                normalized_trough_diff = trough_diff / period * 2 * np.pi  # Convert to radians
                trough_diffs.append(normalized_trough_diff)
                trough_pairs.append((stim_trough, closest_activity_trough))
                all_trough_shifts.append(normalized_trough_diff)

        phase_shifts_troughs[key] = trough_diffs

        # Plot the experiment
        plot_experiment(xp, stim_data, smoothed_activity, stim_peaks, activity_peaks, stim_troughs, activity_troughs, peak_pairs, trough_pairs, "UVandVisExperiment", save_svg=True)

    # Plot box plot for the single experiment
    plot_phase_UVandVis(period_labels, all_peak_shifts, all_trough_shifts, save_svg)
    return phase_shifts_peaks, phase_shifts_troughs


def plot_phase_UVandVis(period_labels, all_peak_shifts, all_trough_shifts, save_svg=False):
    print("Period labels:", period_labels)
    print("All peak shifts:", all_peak_shifts)
    print("All trough shifts:", all_trough_shifts)

    if not all_peak_shifts and not all_trough_shifts:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    box_data_peaks = [all_peak_shifts] if all_peak_shifts else [[]]
    box_data_troughs = [all_trough_shifts] if all_trough_shifts else [[]]

    try:
        ax.boxplot(
            box_data_peaks, 
            positions=[1], 
            widths=0.3, 
            patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='black'), 
            medianprops=dict(color='black'),
            zorder=1
        )

        ax.boxplot(
            box_data_troughs, 
            positions=[2], 
            widths=0.3, 
            patch_artist=True, 
            boxprops=dict(facecolor='lightcoral', color='black'), 
            medianprops=dict(color='black'),
            zorder=1
        )

        jitter_strength = 0.1
        if all_peak_shifts:
            jittered_positions_peaks = 1 + np.random.uniform(-jitter_strength, jitter_strength, size=len(all_peak_shifts))
            ax.scatter(jittered_positions_peaks, all_peak_shifts, color='#1E90FF', alpha=0.8, edgecolor='k', s=100, zorder=3)

        if all_trough_shifts:
            jittered_positions_troughs = 2 + np.random.uniform(-jitter_strength, jitter_strength, size=len(all_trough_shifts))
            ax.scatter(jittered_positions_troughs, all_trough_shifts, color='#FF4500', alpha=0.8, edgecolor='k', s=100, zorder=3)

        # Set x-axis labels
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Peaks', 'Troughs'])

        # Set fixed y-axis limits and π-based ticks
        ax.set_ylim([-np.pi, np.pi/2])
        y_ticks = [-np.pi, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
        y_tick_labels = [r'$-\pi$', r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$']
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

        ax.set_xlabel('Phase Shift Type')
        ax.set_ylabel('Phase Shift (radians)')
        ax.set_title(f'Phase Shift for {period_labels[0]}')

        plt.tight_layout()
        if save_svg:
            plt.savefig(f'phase_shift_UVandVis_{period_labels[0]}.svg', format='svg')
        plt.show()

    except Exception as e:
        print("Error while plotting:", e)

