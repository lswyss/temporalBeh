# tools/filtering.py

import logging

def filter_experiments(test_result, genotype, duration, period_suffix, exclude_dates=None):
    """
    Filter experiments based on genotype, duration, period suffix, and exclusion dates,
    and return and print the matching names.

    Args:
        test_result (dict): Dictionary containing the experiments.
        genotype (str): Genotype to filter by.
        duration (str): Duration to filter by.
        period_suffix (str): Period suffix to filter by.
        exclude_dates (list, optional): List of date prefixes to exclude. Defaults to None.

    Returns:
        list: A list of experiment names matching the provided criteria.
    """
    # Construct the suffix pattern to look for in experiment names
    target_suffix = f"{genotype}_{duration}_{period_suffix}Period"
    
    # Initialize an empty list for storing filtered experiment names
    filtered_names = []

    # Loop through all experiment names in test_result and filter according to criteria
    for name in test_result.keys():
        if name.endswith(target_suffix):
            if exclude_dates is None or not any(name.startswith(date) for date in exclude_dates):
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
