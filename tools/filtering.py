# tools/filtering.py

import logging

def filter_experiments(test_result, genotype, duration, period_suffix, exclude_experiments=None):
    """
    Filter experiments based on genotype, duration, period suffix, and exclusion experiment names,
    and return and print the matching names.

    Args:
        test_result (dict): Dictionary containing the experiments.
        genotype (str): Genotype to filter by.
        duration (str): Duration to filter by.
        period_suffix (str): Period suffix to filter by.
        exclude_experiments (list, optional): List of full experiment names to exclude. Defaults to None.

    Returns:
        list: A list of experiment names matching the provided criteria.
    """
    # Construct the criteria strings to look for in experiment names
    target_genotype = genotype
    target_duration = duration
    target_period_suffix = f"_{period_suffix}Period"
    
    # Initialize an empty list for storing filtered experiment names
    filtered_names = []

    # Loop through all experiment names in test_result and filter according to criteria
    for name in test_result.keys():
        # Skip the experiment if it is in the exclude_experiments list
        if exclude_experiments is not None and name in exclude_experiments:
            continue
        
        # Check if all criteria are present in the experiment name
        if (target_genotype in name and
            target_duration in name and
            name.endswith(target_period_suffix)):
            filtered_names.append(name)

    # Log and print results
    if filtered_names:
        logging.info(f"Filtered Experiments: {filtered_names}")
        print("Filtered Experiments:")
        for name in filtered_names:
            print(name)
    else:
        logging.info("No experiments found matching the criteria.")
        print("No experiments found matching the criteria.")

    # Return the filtered experiment names for further processing
    return filtered_names