from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, StratifiedGroupKFold, learning_curve
import itertools
import pandas as pd
import numpy as np
from scipy.stats import t
import hashlib
import os


def cross_validation(dataset: pd.DataFrame, groupkfold: bool):
    """
    Perform cross-validation on a given model using a dataset.

    Parameters:
    - dataset: The dataset containing the features and target variable.
    - groupkfold: A boolean indicating whether to use GroupKFold or StratifiedKFold for cross-validation.

    Returns:
    - results: A dictionary containing the evaluation scores for different combinations of gases.

    """
    model = XGBClassifier(n_estimators=100, n_jobs=-1)
    
    # force the dataset to be balanced
    if len(dataset['ship_y'] == 0) < len(dataset['ship_y'] == 1):
        dataset = pd.concat([dataset[dataset['ship_y'] == 0].sample(n=len(dataset[dataset['ship_y'] == 1]), replace=True), dataset[dataset['ship_y'] == 1]])
    else:
        dataset = pd.concat([dataset[dataset['ship_y'] == 1].sample(n=len(dataset[dataset['ship_y'] == 0]), replace=True), dataset[dataset['ship_y'] == 0]])

    if groupkfold:
        group = dataset['date']
    else:
        group = None

    # Define the list of gases
    gases = ['NO2', 'HCHO', 'SO2']

    # Initialize a dictionary to store the results
    results = {}

    # Iterate through each gas and its combinations
    for i in range(1, len(gases) + 1):
        combinations = itertools.combinations(gases, i)
        for combination in combinations:
            gasses = list(combination)
            features = [[f"{gas}_scd_min", f"{gas}_scd_max", f"{gas}_scd_mean", f"{gas}_scd_median", f"{gas}_scd_std"] for gas in gasses]        
            features = [item for sublist in features for item in sublist]
            # Select the columns corresponding to the current combination
            X = dataset[features + ['wind_mer_mean', 'wind_zon_mean', 'sensor_zenith_angle', 'solar_azimuth_angle', 'solar_zenith_angle']]
            Y = dataset['ship_y']

            scores_list = []
            for _ in range(5):
                if groupkfold:
                    cv = StratifiedGroupKFold(n_splits=5, shuffle=True)
                else:
                    cv = StratifiedKFold(n_splits=5, shuffle=True)
                scores = cross_validate(model, X, Y, groups=group, cv=cv, scoring=['accuracy', 'average_precision', 'recall', 'f1', 'roc_auc'], 
                                        n_jobs=-1)
                scores_list.append(scores)
            
            scores = {}
            for key in scores_list[0].keys():
                scores[key] = [score[key] for score in scores_list]
            results[", ".join(gasses)] = scores
    return results

def proxy_threshold(
    dataset: pd.DataFrame,
    space: str = "linspace",
    n_steps: int = 20,
    threshold_method: str = "floor",
    window_size: float = None,
    seed = None
) -> dict:
    """
    Generates thresholded DataFrames based on proxy values using either fixed floor thresholds or a sliding window approach.
    The data is balanced by class, ensuring that for each threshold, the number of samples of each class is the same.

    Parameters:
    - dataset (pd.DataFrame): The input dataframe.
    - space (str, optional): The space in which to generate the thresholds. Options: "linspace", "logspace", "quantile". Default is "linspace".
    - n_steps (int, optional): The number of steps or windows to generate. Default is 20.
    - threshold_method (str, optional): The method to use for thresholding. Options: "floor", "bins". Default is "floor".
    - window_size (float, optional): The width of the sliding window for the "bins" method. Required if `threshold_method` is "bins".

    Returns:
    - dict: 
        - If `space` is not "quantile":
            - Keys are either scalar thresholds or tuples representing window ranges.
            - Values are the corresponding thresholded DataFrames.
        - If `space` is "quantile":
            - Returns a tuple containing:
                - The thresholded DataFrames dictionary.
                - The list of quantiles used.
    """
    if space == "linspace":
        if threshold_method == "bins":
            if window_size is None:
                raise ValueError("`window_size` must be provided when `threshold_method` is 'bins'.")
            # Sliding window binning
            min_val = 0
            max_val = 5e8
            thresholds = np.linspace(min_val + (window_size/2), max_val - (window_size/2), n_steps)
        else:
            # Fixed floor thresholding
            # thresholds = np.linspace(0, dataset[dataset["Proxy"] > 0]["Proxy"].quantile(0.9), n_steps)
            raise ValueError("Fixed floor thresholding not supported for linspace")
    elif space == "quantile":
        if threshold_method == "bins":
            if window_size is None:
                raise ValueError("`window_size` must be provided when `threshold_method` is 'bins'.")
            # Sliding window binning based on quantiles
            quantiles = np.linspace(0, 1, n_steps + 1)
            thresholds = np.quantile(dataset[dataset["Proxy"] > 0]["Proxy"], quantiles)
        else:
            # Fixed floor thresholding based on quantiles
            quantiles = np.linspace(0, 0.9, n_steps)
            thresholds = np.quantile(dataset[dataset["Proxy"] > 0]["Proxy"], quantiles)
    else:
        raise ValueError("Space not recognized")

    # Generate thresholded DataFrames
    if threshold_method == "floor":
        proxy_dfs = [
            dataset[(dataset['Proxy'] >= thr) | (dataset['Proxy'] == 0)] 
            for thr in thresholds
        ]
    elif threshold_method == "bins":
        # Sliding window binning
        proxy_dfs = [
            dataset[dataset['Proxy'].between(mid - window_size/2, mid + window_size/2) | (dataset['Proxy'] == 0)] 
            for mid in thresholds
        ]
    else:
        raise ValueError("Threshold method not recognized")

    # Determine the minimum number of samples per class across all thresholded DataFrames
    min_len_zero = min([len(proxy_df[proxy_df['ship_y'] == 0]) for proxy_df in proxy_dfs])
    min_len_one = min([len(proxy_df[proxy_df['ship_y'] == 1]) for proxy_df in proxy_dfs])
    min_len = min(min_len_zero, min_len_one)

    # Balance the dataset by sampling
    for i in range(len(proxy_dfs)):
        proxy_df = proxy_dfs[i]
        if len(proxy_df[proxy_df['ship_y'] == 0]) < min_len or len(proxy_df[proxy_df['ship_y'] == 1]) < min_len:
            raise ValueError(f"Not enough samples to balance at threshold index {i}.")
        proxy_df_balanced = pd.concat([
            proxy_df[proxy_df['ship_y'] == 0].sample(n=min_len, replace=False, random_state=i*seed),
            proxy_df[proxy_df['ship_y'] == 1].sample(n=min_len, replace=False, random_state=i*seed)
        ])
        proxy_dfs[i] = proxy_df_balanced

    return {threshold: proxy_df for threshold, proxy_df in zip(thresholds, proxy_dfs)}
    

def all_classes_present(X, Y, group, cv):
    """
    Efficiently checks if all classes are present in both train and test sets for each fold of a GroupKFold.

    Args:
        X: Features
        Y: Target labels.
        group: Group identifiers for GroupKFold.
        cv: GroupKFold object.

    Returns:
        True if all classes are present in every train/test split, False otherwise.
    """
    unique_classes = set(Y)
    
    # Precompute sets of unique classes per group
    group_class_sets = {}
    for group_id, label in zip(group, Y):
        group_class_sets.setdefault(group_id, set()).add(label)
        

    for _, test_index in cv.split(X, Y, groups=group):
        test_group_ids = set(group.iloc[test_index])
        test_classes = set().union(*(group_class_sets[gid] for gid in test_group_ids))

        # If not all unique classes are in the test fold, fail early
        if len(test_classes) < len(unique_classes):
            return False

        remaining_group_ids = set(group_class_sets) - test_group_ids
        remaining_classes = set().union(*(group_class_sets[gid] for gid in remaining_group_ids))
        
        # If not all unique classes are in the remaining train folds, fail early
        if len(remaining_classes) < len(unique_classes):
            return False

    # If all tests pass, all classes are present
    return True


def get_features(combination):
    """
    Get the features for a given combination of gases.
    """
    if combination:
        features = [[f"{gas}_scd_min", f"{gas}_scd_max", f"{gas}_scd_mean", f"{gas}_scd_median", f"{gas}_scd_std"] for gas in combination]
        features = [item for sublist in features for item in sublist]
    else:
        features = []
    features += ['wind_mer_mean', 'wind_zon_mean', 'sensor_zenith_angle', 'sensor_azimuth_angle', 'solar_azimuth_angle', 'solar_zenith_angle']
    return features

def gasses_to_string(gasses):
    """
    Convert a string of gasses seperated by an underscore to a well-formatted string using subscripts.
    """
    gasses = gasses.split("_")
    # replace numbers with subscripts
    gasses = [gas.replace("2", "₂") for gas in gasses]
    return ", ".join(gasses)

def get_margin_of_error(data, axis=0, confidence_level=0.95):
    """
    Calculate the margin of error for a given dataset.
    """
    n_observations = data.shape[axis]
    std_scores = np.std(data, axis=axis)
    sem_scores = std_scores / np.sqrt(n_observations)
    degrees_freedom = n_observations - 1
    critical_value = t.ppf((1+confidence_level)/2, degrees_freedom)
    margin_of_error = sem_scores * critical_value
    return margin_of_error

def generate_filename(prefix, parameters, directory="results"):
    """
    Generate a standardized filename based on parameters.

    Args:
    - prefix (str): A prefix for the filename (e.g., experiment name).
    - parameters (dict): A dictionary of parameters to include in the filename.
    - directory (str): The directory in which to save the file.

    Returns:
    - str: The generated filename including the full path.
    """
    # Create a string representation of the parameters
    param_str = "_".join(f"{key}-{value}" for key, value in parameters.items())
    # Create a hash for long parameter sets to avoid overly long filenames
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:6]
    filename = f"{prefix}_{param_hash}.pkl"

    return os.path.join(directory, filename)
