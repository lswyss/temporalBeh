#this is in folder tools -> dataframe.py
#Livia 06.21.2024
import numpy as np
import pandas as pd
import seaborn as sns

# Define markers and color palette globally within this script
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', 'P', '*', 'X', 'd']
color_palette = sns.color_palette("tab20", 20)  # More distinct colors

# Function to prepare data based on experiments and organize into a DataFrame with trials as columns
def prepare_data_df(test_result, filtered_experiments, trials_range, exclude_worms=None):
    experiment_to_color = {exp: color_palette[i % len(color_palette)] for i, exp in enumerate(filtered_experiments)}
    print("Experiment to Color Mapping:")
    for exp, color in experiment_to_color.items():
        print(f"{exp}: {color}")
    
    data_records = []
    for exp in filtered_experiments:
        experiment_data = test_result[exp]['data']
        for worm_idx, worm_data in enumerate(experiment_data):
            if exclude_worms and exp in exclude_worms and worm_idx in exclude_worms[exp]:
                continue
            record = {
                'experiment': exp, 
                'worm_id': worm_idx,
                'color': experiment_to_color[exp],
                'marker': markers[worm_idx % len(markers)]
            }
            for trial_idx in trials_range:
                if trial_idx < worm_data.shape[0]:
                    trial_data = worm_data[trial_idx]
                    record[f'trial_{trial_idx}'] = trial_data
                else:
                    record[f'trial_{trial_idx}'] = np.nan
            data_records.append(record)
    df = pd.DataFrame(data_records)
    return df

# Function to exclude specific worms
def exclude_worms_from_df(df, exclude_worms):
    for exp, worms in exclude_worms.items():
        df = df[~((df['experiment'] == exp) & (df['worm_id'].isin(worms)))]
    return df

def print_df_summary(df):
    print("DataFrame Summary:")
    print(df.info())
    print("\nExperiment Counts:")
    print(df['experiment'].value_counts())
    print("\nWorm Counts per Experiment:")
    print(df.groupby('experiment')['worm_id'].nunique())
    print("\nTrial Counts per Worm:")
    print(df.groupby(['experiment', 'worm_id']).count())

    