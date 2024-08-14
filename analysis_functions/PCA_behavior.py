import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.markers import MarkerStyle
from sklearn.cluster import KMeans
from analysis_functions.results_sin_indiv import load_and_filter_data, prepare_aggregated_data
from tools.preparing_data import *




# # Utility function to generate distinct colors
# def get_distinct_colors(n):
#     colors = plt.cm.tab20(np.linspace(0, 1, 20))
#     if n > 20:
#         additional_colors = plt.cm.tab20b(np.linspace(0, 1, (n - 20)))
#         colors = np.vstack((colors, additional_colors))
#     return colors

# # Combined function to plot PCA results for individual worms and clusters
# def plot_pca_combined(aggregated_data, tau, intervals, num_clusters=3, n_components=3, trial_number=0):
#     fig, axs = plt.subplots(2, len(intervals), figsize=(20, 12))
#     num_worms = aggregated_data[trial_number].shape[0]
#     worm_colors = get_distinct_colors(num_worms)  # Colors for individual worms
#     cluster_colors = get_distinct_colors(num_clusters)  # Colors for clusters

#     for i, (start, end) in enumerate(intervals):
#         # Select the time points within the given interval
#         time_indices = np.where((tau >= start) & (tau < end))[0]
#         if len(time_indices) == 0:
#             print(f"No data points found in interval {start}-{end} min")
#             continue

#         # Use the specified trial data
#         interval_data = aggregated_data[trial_number][:, time_indices]

#         # Flatten the data across selected time points for PCA
#         num_timepoints = interval_data.shape[1]
#         flattened_data = interval_data.reshape(num_worms, num_timepoints)

#         if flattened_data.shape[1] == 0:
#             print(f"No features to process in interval {start}-{end} min")
#             continue

#         # Apply PCA
#         pca = PCA(n_components=n_components)
#         pca_result = pca.fit_transform(flattened_data)
        
#         # Print explained variance
#         explained_variance = pca.explained_variance_ratio_
#         print(f"Explained variance for interval {start}-{end}: {explained_variance}")

#         # Plot the PCA results for individual worms (first two components)
#         ax_indiv = axs[0, i]
#         for worm_idx in range(num_worms):
#             ax_indiv.scatter(pca_result[worm_idx, 0], pca_result[worm_idx, 1], color=worm_colors[worm_idx])
        
#         ax_indiv.set_title(f'PCA (Individual Worms): {start}-{end} min')
#         ax_indiv.set_xlabel('Principal Component 1')
#         ax_indiv.set_ylabel('Principal Component 2')
#         ax_indiv.grid(True)

#         # Apply K-means clustering
#         kmeans = KMeans(n_clusters=num_clusters)
#         clusters = kmeans.fit_predict(pca_result)

#         # Plot the PCA results with clusters (first two components)
#         ax_cluster = axs[1, i]
#         for cluster_idx in range(num_clusters):
#             cluster_points = pca_result[clusters == cluster_idx]
#             ax_cluster.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_colors[cluster_idx])
        
#         ax_cluster.set_title(f'PCA (Clusters): {start}-{end} min')
#         ax_cluster.set_xlabel('Principal Component 1')
#         ax_cluster.set_ylabel('Principal Component 2')
#         ax_cluster.grid(True)

#     plt.tight_layout()
#     plt.show()


#     # Function to plot PCA results for all trials of individual worms
# def plot_pca_all_trials_with_clusters(aggregated_data, tau, intervals, num_worms_to_plot=10, num_clusters=3):
#     fig, axs = plt.subplots(2, len(intervals), figsize=(20, 12))
#     worm_colors = get_distinct_colors(num_worms_to_plot)  # Colors for individual worms
#     cluster_colors = get_distinct_colors(num_clusters)  # Colors for clusters

#     for i, (start, end) in enumerate(intervals):
#         # Select the time points within the given interval
#         time_indices = np.where((tau >= start) & (tau < end))[0]
#         #print(f"Time indices for interval {start}-{end}:", time_indices)
#         if len(time_indices) == 0:
#             print(f"No data points found in interval {start}-{end} min")
#             continue

#         # Collect data for all trials of the specified worms
#         combined_data = []
#         worm_labels = []

#         for worm_idx in range(num_worms_to_plot):
#             worm_trials = []
#             for trial_idx in range(len(aggregated_data)):
#                 worm_trials.append(aggregated_data[trial_idx][worm_idx, time_indices])
#             worm_trials = np.vstack(worm_trials)
#             combined_data.append(worm_trials)
#             worm_labels.extend([worm_idx] * worm_trials.shape[0])

#         combined_data = np.vstack(combined_data)
#         worm_labels = np.array(worm_labels)

#         # Apply PCA
#         pca = PCA(n_components=2)
#         pca_result = pca.fit_transform(combined_data)
        
#         # Print explained variance
#         explained_variance = pca.explained_variance_ratio_
#         print(f"Explained variance for interval {start}-{end}: {explained_variance}")

#         # Plot the PCA results for all trials of individual worms
#         ax_indiv = axs[0, i]
#         for worm_idx in range(num_worms_to_plot):
#             worm_points = pca_result[worm_labels == worm_idx]
#             ax_indiv.scatter(worm_points[:, 0], worm_points[:, 1], color=worm_colors[worm_idx], label=f'Worm {worm_idx+1}' if i == 0 else "")
        
#         ax_indiv.set_title(f'PCA (All Trials): {start}-{end} min')
#         ax_indiv.set_xlabel('Principal Component 1')
#         ax_indiv.set_ylabel('Principal Component 2')
#         ax_indiv.grid(True)
#         if i == 0:
#             ax_indiv.legend()

#         # Apply K-means clustering
#         kmeans = KMeans(n_clusters=num_clusters)
#         clusters = kmeans.fit_predict(pca_result)

#         # Plot the PCA results with clusters
#         ax_cluster = axs[1, i]
#         for cluster_idx in range(num_clusters):
#             cluster_points = pca_result[clusters == cluster_idx]
#             ax_cluster.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_colors[cluster_idx])
        
#         ax_cluster.set_title(f'PCA (Clusters): {start}-{end} min')
#         ax_cluster.set_xlabel('Principal Component 1')
#         ax_cluster.set_ylabel('Principal Component 2')
#         ax_cluster.grid(True)

#     plt.tight_layout()
#     plt.show()

#Livia in folder analysis_functions called PCA_behavior.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib.lines import Line2D
from tools.preparing_data import markers, color_palette

def flatten_trial_data(df, trial_number):
    data_records = []
    for _, row in df.iterrows():
        trial_data = row[f'trial_{trial_number}']
        if isinstance(trial_data, float) and np.isnan(trial_data):
            continue
        if isinstance(trial_data, np.ndarray) and not pd.isna(trial_data).all():
            flattened_data = trial_data.flatten()
            data_records.append({
                'experiment': row['experiment'],
                'worm_id': row['worm_id'],
                'data': flattened_data,
                'color': row['color'],
                'marker': row['marker']
            })
    return pd.DataFrame(data_records)

def plot_pca_trials(df, trial_number, title_suffix=''):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    axs = axs.flatten()

    intervals = [(0, 4), (4, 24), (24, 30)]
    interval_titles = ['0-4 min', '4-24 min', '24-30 min']

    for i, (start, end) in enumerate(intervals):
        interval_data = []
        metadata = []

        for _, row in df.iterrows():
            trial_data = row['data']
            interval_trial_data = trial_data[start:end] if isinstance(trial_data, np.ndarray) else []
            if len(interval_trial_data) > 0:
                interval_data.append(interval_trial_data)
                metadata.append((row['experiment'], row['worm_id'], row['color'], row['marker']))

        interval_data = np.array(interval_data)
        if interval_data.size == 0:
            continue

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(interval_data)

        for idx, (experiment, worm_id, color, marker) in enumerate(metadata):
            axs[i].scatter(pca_result[idx, 0], pca_result[idx, 1],
                           color=color, marker=marker,
                           label=f'{experiment}, Worm {worm_id}' if i == 0 else "")

        axs[i].set_title(f'Trial {trial_number} PCA ({interval_titles[i]})')
        if i == 0:
            axs[i].set_ylabel('Principal Component 2')
        axs[i].set_xlim(-5, 15)
        axs[i].set_ylim(-2.5, 5)

        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(pca_result)
        axs[i + 3].scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
        axs[i + 3].set_title(f'Trial {trial_number} KMeans ({interval_titles[i]})')
        if i == 0:
            axs[i + 3].set_ylabel('Principal Component 2')
        axs[i + 3].set_xlim(-5, 15)
        axs[i + 3].set_ylim(-2.5, 5)

    # Set the x-labels for the bottom row of plots
    for i in range(3, 6):
        axs[i].set_xlabel('Principal Component 1')
    axs[0].set_ylabel('Principal Component 2')
    axs[3].set_ylabel('Principal Component 2')
    axs[3].set_xlabel('Principal Component 1')
    
    # Create custom legends
    experiment_labels = df['experiment'].unique()
    color_mapping = {exp: df[df['experiment'] == exp]['color'].iloc[0] for exp in experiment_labels}
    custom_lines_experiments = [Line2D([0], [0], color=color_mapping[exp], lw=4) for exp in experiment_labels]
    custom_lines_worms = [Line2D([0], [0], marker=markers[i % len(markers)], color='w', markerfacecolor='k', markersize=10) for i in range(len(markers))]
    
    legend_experiments = fig.legend(custom_lines_experiments, experiment_labels, loc='center right', title='Experiments')
    legend_worms = fig.legend(custom_lines_worms, [f'Worm {i}' for i in range(len(markers))], loc='lower right', title='Worms')
    
    fig.add_artist(legend_experiments)  # Add the experiments legend to the plot

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the plot to make space for the legends
    plt.show()