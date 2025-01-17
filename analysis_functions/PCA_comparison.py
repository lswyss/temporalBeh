# PCA_comparison in analysis_functions
#Livia 11.01.2024

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from analysis_functions.PCA_behavior import get_distinct_colors
from analysis_functions.results_sin_indiv import load_and_filter_data, prepare_aggregated_data

# Function to plot PCA for two genotypes

def plot_pca_two_genotypes(aggregated_data_cntrl, aggregated_data_expt, tau, intervals, num_worms_to_plot=10, save_path=None):
    fig, axs = plt.subplots(2, len(intervals), figsize=(20, 10))
    worm_colors = get_distinct_colors(num_worms_to_plot)  # Colors for individual worms

    for i, (start, end) in enumerate(intervals):
        # Select the time points within the given interval
        time_indices = np.where((tau >= start) & (tau < end))[0]
        if len(time_indices) == 0:
            print(f"No data points found in interval {start}-{end} min")
            continue

        combined_data_cntrl = []
        combined_data_expt = []

        for worm_idx in range(num_worms_to_plot):
            cntrl_trials = []
            expt_trials = []
            for trial_idx in range(len(aggregated_data_cntrl)):
                if time_indices[-1] < aggregated_data_cntrl[trial_idx].shape[1]:
                    cntrl_trials.append(aggregated_data_cntrl[trial_idx][worm_idx, time_indices])
                else:
                    cntrl_trials.append(aggregated_data_cntrl[trial_idx][worm_idx, :time_indices[-1]+1])
                if time_indices[-1] < aggregated_data_expt[trial_idx].shape[1]:
                    expt_trials.append(aggregated_data_expt[trial_idx][worm_idx, time_indices])
                else:
                    expt_trials.append(aggregated_data_expt[trial_idx][worm_idx, :time_indices[-1]+1])
            cntrl_trials = np.vstack(cntrl_trials)
            expt_trials = np.vstack(expt_trials)
            combined_data_cntrl.append(cntrl_trials)
            combined_data_expt.append(expt_trials)

        combined_data_cntrl = np.vstack(combined_data_cntrl)
        combined_data_expt = np.vstack(combined_data_expt)

        # Apply PCA
        combined_data = np.vstack((combined_data_cntrl, combined_data_expt))
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_data)

        # Print explained variance
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance for interval {start}-{end}: {explained_variance}")

        # Plot the PCA results for all trials of individual worms
        ax_indiv = axs[0, i]
        num_cntrl_points = combined_data_cntrl.shape[0]
        for worm_idx in range(num_worms_to_plot):
            # cntrl points (using circles)
            cntrl_points = pca_result[worm_idx * len(aggregated_data_cntrl):(worm_idx + 1) * len(aggregated_data_cntrl)]
            ax_indiv.scatter(cntrl_points[:, 0], cntrl_points[:, 1], color=worm_colors[worm_idx], marker='o', label=f'Control Worm {worm_idx + 1}' if i == 0 else "")
            # expt points (using triangles)
            expt_points = pca_result[num_cntrl_points + worm_idx * len(aggregated_data_expt):num_cntrl_points + (worm_idx + 1) * len(aggregated_data_expt)]
            ax_indiv.scatter(expt_points[:, 0], expt_points[:, 1], color=worm_colors[worm_idx], marker='^', label=f'Experimental Worm {worm_idx + 1}' if i == 0 else "")

        ax_indiv.set_title(f'PCA (All Trials): {start}-{end} min')
        ax_indiv.set_xlabel('Principal Component 1')
        ax_indiv.set_ylabel('Principal Component 2')
        ax_indiv.set_ylim(-30, 45)

        if i == 0:
            # Move legend to the far right next to the plots
            handles, labels = ax_indiv.get_legend_handles_labels()
            fig.legend(handles, labels, loc='center right', title='Worms')

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(pca_result)

        # Plot the PCA results with clusters
        ax_cluster = axs[1, i]
        cluster_colors = get_distinct_colors(3)  # Colors for clusters
        for cluster_idx in range(3):
            cluster_points = pca_result[clusters == cluster_idx]
            ax_cluster.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_colors[cluster_idx])

        ax_cluster.set_title(f'PCA (Clusters): {start}-{end} min')
        ax_cluster.set_xlabel('Principal Component 1')
        ax_cluster.set_ylabel('Principal Component 2')
        ax_cluster.set_ylim(-30, 45)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, format='svg')

    plt.show()


def plot_pca_two_genotypes_simple(aggregated_data_cntrl, aggregated_data_expt, tau, intervals, num_worms_to_plot=10, save_path=False):
    fig, axs = plt.subplots(2, len(intervals), figsize=(20, 12))
    cntrl_color = 'teal'
    expt_color = 'orange'

    for i, (start, end) in enumerate(intervals):
        # Select the time points within the given interval
        time_indices = np.where((tau >= start) & (tau < end))[0]
        if len(time_indices) == 0:
            print(f"No data points found in interval {start}-{end} min")
            continue

        combined_data_cntrl = []
        combined_data_expt = []

        for worm_idx in range(num_worms_to_plot):
            cntrl_trials = []
            expt_trials = []
            for trial_idx in range(len(aggregated_data_cntrl)):
                if time_indices[-1] < aggregated_data_cntrl[trial_idx].shape[1]:
                    cntrl_trials.append(aggregated_data_cntrl[trial_idx][worm_idx, time_indices])
                else:
                    cntrl_trials.append(aggregated_data_cntrl[trial_idx][worm_idx, :time_indices[-1]+1])
                if time_indices[-1] < aggregated_data_expt[trial_idx].shape[1]:
                    expt_trials.append(aggregated_data_expt[trial_idx][worm_idx, time_indices])
                else:
                    expt_trials.append(aggregated_data_expt[trial_idx][worm_idx, :time_indices[-1]+1])
            cntrl_trials = np.vstack(cntrl_trials)
            expt_trials = np.vstack(expt_trials)
            combined_data_cntrl.append(cntrl_trials)
            combined_data_expt.append(expt_trials)

        combined_data_cntrl = np.vstack(combined_data_cntrl)
        combined_data_expt = np.vstack(combined_data_expt)

        # Apply PCA
        combined_data = np.vstack((combined_data_cntrl, combined_data_expt))
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_data)

        # Print explained variance
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance for interval {start}-{end}: {explained_variance}")

        # Plot the PCA results for all trials of individual worms
        ax_indiv = axs[0, i]
        num_cntrl_points = combined_data_cntrl.shape[0]
        # Control points
        cntrl_points = pca_result[:num_cntrl_points]
        ax_indiv.scatter(cntrl_points[:, 0], cntrl_points[:, 1], color=cntrl_color, label='Control' if i == 0 else "")
        # Experimental points
        expt_points = pca_result[num_cntrl_points:]
        ax_indiv.scatter(expt_points[:, 0], expt_points[:, 1], color=expt_color, label='Experimental' if i == 0 else "")

        ax_indiv.set_title(f'PCA (All Trials): {start}-{end} min')
        ax_indiv.set_xlabel('Principal Component 1')
        ax_indiv.set_ylabel('Principal Component 2')
        ax_indiv.set_ylim(-30, 45)

        if i == 0:
            # Move legend to the far right next to the plots
            handles, labels = ax_indiv.get_legend_handles_labels()
            fig.legend(handles, labels, loc='center right', title='Genotypes')

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(pca_result)

        # Plot the PCA results with clusters
        ax_cluster = axs[1, i]
        cluster_colors = get_distinct_colors(3)  # Colors for clusters
        for cluster_idx in range(3):
            cluster_points = pca_result[clusters == cluster_idx]
            ax_cluster.scatter(cluster_points[:, 0], cluster_points[:, 1], color=cluster_colors[cluster_idx])

        ax_cluster.set_title(f'PCA (Clusters): {start}-{end} min')
        ax_cluster.set_xlabel('Principal Component 1')
        ax_cluster.set_ylabel('Principal Component 2')
        ax_cluster.set_ylim(-30, 45)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    # Save the plot if save_path is True
    if save_path:
        plt.savefig(save_path, format='svg')

    plt.show()