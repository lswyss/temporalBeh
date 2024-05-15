#04.24.23_Livia
# in tools folder plotting_functions.py
#stremling plotting so it is easier to make stylistic changes quickly

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_sin(ax, xp, y, rng, label, color='blue', zorder=-2, plot_stim=False, stim_data=None):
    """Enhanced and concise plotting function with improved aesthetics."""
    # Set font properties for the plot
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'svg.fonttype': 'none'})
    
    ax.plot(xp, y, lw=2, color=color, label=label, zorder=zorder)
    
    # Filling between ranges with confidence interval
    ax.fill_between(xp, *rng, alpha=0.5, color=color, lw=0, edgecolor='None', zorder=zorder)
    
    if plot_stim:
        ax.plot(xp, stim_data, c='lightgray', zorder=-10)
    
    # Set labels and title with font properties directly applied
    ax.set_xlabel('Time (min)', fontsize=10)
    ax.set_ylabel('Response', fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc='upper right')

    ax.grid(False)


def gridplot_indiv_trials(axs, worm_data, stim_data, time_axis_minutes, num_worms, max_trials):
    """Plot individual worm trials in a grid format with adjusted stimulus visibility and labeling."""
    for i in range(num_worms):
        for j in range(max_trials):
            ax = axs[i][j] if num_worms > 1 else axs[j]
            if j < len(worm_data[i]):  # Check if the current worm has this trial
                # Plot stimulus first with lower visibility
                ax.plot(time_axis_minutes, stim_data * max(worm_data[i][j, :]), color='red', alpha=0.2, label='Stimulus' if i == 0 and j == 0 else "")  # Reduced alpha for faint appearance
                # Plot response data on top
                ax.plot(time_axis_minutes, worm_data[i][j, :], color='cornflowerblue', label='Response')
                
                if i == 0:
                    ax.set_title(f"Trial {j+1}")  # Set trial numbers only on the top row
                if i == num_worms - 1:
                    ax.set_xlabel('Time (min)')  # Set time label only on the bottom row
            else:
                ax.axis('off')  # Turn off axis if no data for this cell
            if j == 0:  # Label the rows with worm identifiers only on the first column
                ax.set_ylabel(f"Worm {i+1}")

    # Add legend to the first plot if there is at least one plot and if it's appropriate
    if num_worms > 0 and max_trials > 0:
        axs[0][0].legend(loc='upper right')

