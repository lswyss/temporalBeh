import pickle

def print_pickle_structure(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    print("Keys in the pickle file:", data.keys())

    # Print details about 'tau' since it's important for understanding timestamps
    print("\nDetails about 'tau' (time points):")
    if 'tau' in data:
        print("Type of 'tau':", type(data['tau']))
        print("Length of 'tau':", len(data['tau']))

    # Print details about one example experiment
    experiment_keys = [key for key in data.keys() if key != 'tau' and isinstance(data[key], dict)]
    if experiment_keys:
        example_key = experiment_keys[0]
        print("\nExample experiment key:", example_key)
        print("Data type of example experiment entry:", type(data[example_key]))
        print("Sub-keys in the example experiment entry:", data[example_key].keys())

        # Details about 'data' under the experiment
        if 'data' in data[example_key]:
            print("\nDetails about 'data' (containing data for worms):")
            print("Type of 'data':", type(data[example_key]['data']))
            print("Number of worms in 'data':", len(data[example_key]['data']))
            print("Type of the first element in 'data' (representing a worm):", type(data[example_key]['data'][0]))
            print("Shape of the first numpy array in 'data' (representing trials and time points for the first worm):", data[example_key]['data'][0].shape)

        # Details about 'stim' under the experiment
        if 'stim' in data[example_key]:
            print("\nDetails about 'stim' (stimulus data):")
            print("Type of 'stim':", type(data[example_key]['stim']))
            print("Shape of 'stim' data:", data[example_key]['stim'].shape)
