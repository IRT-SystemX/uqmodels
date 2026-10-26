from importlib.util import find_spec
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import tensorflow as tf
#import torch
import numpy as np 
import polars as pl 

TORCH_AVAILABLE = find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import IterableDataset

    class PytorchIterableDataset(IterableDataset):
        """Adapt a custom loader to the PyTorch IterableDataset interface."""

        def __init__(self, custom_loader):
            super().__init__()
            self.custom_loader = custom_loader

        def __iter__(self):
            for batch in self.custom_loader:
                batch_dict = {}

                for column in batch.columns:
                    series = batch[column]

                    if series.dtype in {
                        pl.Datetime,
                        pl.Date,
                        pl.Categorical,
                        pl.String,
                    }:
                        continue

                    first_value = series[0]

                    if isinstance(first_value, (list, np.ndarray, pl.Series)):
                        values = series.to_list()
                    else:
                        values = series.to_numpy()

                    batch_dict[column] = torch.as_tensor(
                        values,
                        dtype=torch.float32,
                    )

                yield batch_dict


def create_tf_dataset(custom_loader):
    # On prend un échantillon pour la signature
    first_batch = next(iter(custom_loader))
    
    output_signature = {}
    for col in first_batch.columns:
        # On vérifie si c'est une colonne de listes (séquences)
        if isinstance(first_batch[col][0], (list, np.ndarray, pl.Series)):
            # Forme : (Batch, Longueur_Séquence)
            seq_len = len(first_batch[col][0])
            output_signature[col] = tf.TensorSpec(shape=(None, seq_len), dtype=tf.float32)
        else:
            # Forme : (Batch,)
            output_signature[col] = tf.TensorSpec(shape=(None,), dtype=tf.float32)

    def generator():
        for batch in custom_loader:
            data_dict = {}
            for col in batch.columns:
                val = batch[col]
                # SI c'est une colonne de listes, on doit la "stacker" pour faire une matrice 2D
                if isinstance(val[0], (list, np.ndarray, pl.Series)):
                    # np.vstack ou np.array(val.to_list()) transforme [ [1..], [1..] ] en matrix
                    data_dict[col] = np.array(val.to_list(), dtype=np.float32)
                else:
                    data_dict[col] = val.to_numpy().astype(np.float32)
            yield data_dict

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

####### Fonction de split 

def split_dataset_with_context_balance(dataset, categorical_columns, float_columns=None, bins=10, test_size=0.2, random_state=None):
    """
    Splits a dataset into train and test sets while preserving the distribution
    of categorical and floating-point context variables.

    :param dataset: DataFrame containing the dataset to split.
    :param categorical_columns: List of categorical columns to consider for stratification.
    :param float_columns: List of floating-point columns to analyze (optional for stratification).
    :param bins: Number of bins for the floating-point columns.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Seed for reproducibility.
    :return: Two DataFrames, train and test.
    """
    if float_columns is None:
        float_columns = []

    for float_col in float_columns:
        # Bin the floating-point column
        dataset['binned_' + float_col] = pd.cut(dataset[float_col], bins=bins, labels=False)

        # Debug: Check the distribution of the binned float column
        print(f"Distribution of binned {float_col}:")
        print(dataset['binned_' + float_col].value_counts())

        # If only one bin is present, log a warning
        if dataset['binned_' + float_col].nunique() <= 1:
            print(f"Warning: Only one bin detected for {float_col}. Excluding it from stratification.")
            dataset.drop(columns=['binned_' + float_col], inplace=True)

    # Use categorical columns and optionally the binned float columns for stratification
    stratify_columns = categorical_columns + [f'binned_{col}' for col in float_columns if f'binned_{col}' in dataset]

    # Create a composite stratification key
    dataset['stratify_key'] = dataset[stratify_columns].astype(str).agg('-'.join, axis=1)

    # Debug: Check the distribution of the stratification key
    print("Distribution of stratification key:")
    print(dataset['stratify_key'].value_counts())

    # Check for sparse classes in the stratification key
    value_counts = dataset['stratify_key'].value_counts()

    # Handle sparse classes by grouping them into an 'other' category
    sparse_classes = value_counts[value_counts <= 1].index
    if len(sparse_classes) > 0:
        print(f"Warning: Found {len(sparse_classes)} sparse stratification classes. Grouping them into 'other'.")
        dataset['stratify_key'] = dataset['stratify_key'].replace(sparse_classes, 'other')

    # Debug: Check the updated distribution of the stratification key
    print("Updated distribution of stratification key after handling sparse classes:")
    print(dataset['stratify_key'].value_counts())

    # Split the dataset while stratifying on the composite key
    train, test = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset['stratify_key']
    )

    # Drop only the composite stratify_key column (optional)
    train = train.drop(columns=['stratify_key'], errors='ignore')
    test = test.drop(columns=['stratify_key'], errors='ignore')

    # Return train and test DataFrames (with binned columns preserved)
    return train, test

def validate_split_distribution(train, test, categorical_columns, float_columns=None, bins=10, output_path=None, diff_threshold=0.05):
    """
    Validates the distribution of categorical and floating-point variables between train and test splits.
    """
    results = []
    warnings = []

    # Validate categorical columns
    for col in categorical_columns:
        train_dist = train[col].value_counts(normalize=True)
        test_dist = test[col].value_counts(normalize=True)

        diff = (train_dist - test_dist).abs().fillna(0)

        # Include total counts for clarity
        train_counts = train[col].value_counts()
        test_counts = test[col].value_counts()

        # Flag significant differences
        significant_diffs = diff[diff > diff_threshold]
        if not significant_diffs.empty:
            warnings.append(f"Significant differences found in column '{col}': {significant_diffs.to_dict()}")

        results.append(pd.DataFrame({
            'Category': diff.index,
            f'Train_{col}_Normalized': train_dist,
            f'Test_{col}_Normalized': test_dist,
            f'Train_{col}_Count': train_counts,
            f'Test_{col}_Count': test_counts,
            f'Difference_{col}': diff
        }))

        # Plot the distributions side by side
        combined = pd.DataFrame({'Train': train_dist, 'Test': test_dist}).fillna(0)
        combined.plot(kind='bar', width=0.8)
        plt.title(f'Distribution Comparison: {col}')
        plt.xlabel('Categories')
        plt.ylabel('Proportion')
        plt.legend(['Train', 'Test'])
        plt.tight_layout()
        plt.show()

    # Validate floating-point columns if provided
    if float_columns:
        for float_col in float_columns:
            bins_edges = pd.cut(pd.concat([train[float_col], test[float_col]]), bins=bins)
            train_binned = pd.cut(train[float_col], bins=bins_edges.cat.categories).value_counts(normalize=True)
            test_binned = pd.cut(test[float_col], bins=bins_edges.cat.categories).value_counts(normalize=True)

            diff = (train_binned - test_binned).abs().fillna(0)

            # Include total counts for clarity
            train_binned_counts = pd.cut(train[float_col], bins=bins_edges.cat.categories).value_counts()
            test_binned_counts = pd.cut(test[float_col], bins=bins_edges.cat.categories).value_counts()

            # Flag significant differences
            significant_diffs = diff[diff > diff_threshold]
            if not significant_diffs.empty:
                warnings.append(f"Significant differences found in column '{float_col}' (binned): {significant_diffs.to_dict()}")

            results.append(pd.DataFrame({
                'Bin': diff.index.astype(str),
                f'Train_{float_col}_Normalized': train_binned,
                f'Test_{float_col}_Normalized': test_binned,
                f'Train_{float_col}_Count': train_binned_counts,
                f'Test_{float_col}_Count': test_binned_counts,
                f'Difference_{float_col}': diff
            }))

            # Plot the distributions side by side
            combined = pd.DataFrame({'Train': train_binned, 'Test': test_binned}).fillna(0)
            combined.plot(kind='bar', width=0.8)
            plt.title(f'Distribution Comparison: {float_col}')
            plt.xlabel('Bins')
            plt.ylabel('Proportion')
            plt.legend(['Train', 'Test'])
            plt.tight_layout()
            plt.show()

    # Combine results into a single summary
    summary = pd.concat(results, keys=[r.columns[-1] for r in results], axis=0)

    # Save to JSON if output_path is provided
    if output_path:
        summary.reset_index(drop=True).to_json(output_path, orient='records', indent=4)
        print(f"Validation summary saved to {output_path}")

    # Output warnings
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    return summary

def validate_joint_distribution(train, test, welding_seam, variables):
    """
    Validates the joint distribution of specified variables for a given welding seam.
    """
    # Filter datasets for the specific welding seam
    train_filtered = train[train['welding-seams'] == welding_seam].copy()
    test_filtered = test[test['welding-seams'] == welding_seam].copy()

    # Create a column for joint combination
    train_filtered['joint_combination'] = train_filtered[variables].astype(str).agg(', '.join, axis=1)
    test_filtered['joint_combination'] = test_filtered[variables].astype(str).agg(', '.join, axis=1)

    # Calculate proportions
    train_joint_dist = train_filtered['joint_combination'].value_counts(normalize=True)
    test_joint_dist = test_filtered['joint_combination'].value_counts(normalize=True)

    # Align indices to compare proportions
    all_combinations = train_joint_dist.index.union(test_joint_dist.index)
    train_joint_dist = train_joint_dist.reindex(all_combinations, fill_value=0)
    test_joint_dist = test_joint_dist.reindex(all_combinations, fill_value=0)

    # Plot the joint distribution comparison
    bar_width = 0.35
    index = range(len(all_combinations))

    plt.bar(index, train_joint_dist.values, bar_width, label='Train', alpha=0.7, color='blue')
    plt.bar([i + bar_width for i in index], test_joint_dist.values, bar_width, label='Test', alpha=0.7, color='orange')

    plt.xticks(index, all_combinations, rotation=90)
    plt.xlabel('Joint Combinations')
    plt.ylabel('Proportion')
    plt.title(f'Joint Distribution Comparison for Welding Seam: {welding_seam}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def validate_welding_seam_distributions(train, test, welding_seam):
    """
    Validates distributions for a specific welding seam across multiple variables,
    and plots all comparisons in a single squared figure.

    :param train: DataFrame containing the training set.
    :param test: DataFrame containing the test set.
    :param welding_seam: Welding seam to filter on (e.g., 'c19').
    """
    # Filter datasets for the specified welding seam
    train_filtered = train[train['welding-seams'] == welding_seam]
    test_filtered = test[test['welding-seams'] == welding_seam]

    variables = ['class', 'blur_class', 'binned_blur_level', 'binned_luminosity_level']

    # Check if the required variables exist in train and test
    for var in variables:
        if var not in train_filtered.columns or var not in test_filtered.columns:
            raise KeyError(f"{var} column missing in train or test data. Please check your preprocessing steps.")

    # Calculate the grid size for square-like layout
    num_vars = len(variables)
    cols = math.ceil(math.sqrt(num_vars))  # Number of columns
    rows = math.ceil(num_vars / cols)     # Number of rows

    # Set up subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()  # Flatten to iterate easily if rows/cols > 1
    fig.suptitle(f'Distribution Comparisons for Welding Seam: {welding_seam}', fontsize=16)

    for i, var in enumerate(variables):
        # Compute normalized distributions
        train_dist = train_filtered[var].value_counts(normalize=True)
        test_dist = test_filtered[var].value_counts(normalize=True)

        # Combine distributions into a single DataFrame
        combined = pd.DataFrame({'Train': train_dist, 'Test': test_dist}).fillna(0)

        # Plot distributions in the subplot
        combined.plot(kind='bar', ax=axes[i], width=0.8)

        axes[i].set_title(f'Distribution Comparison - {var}')
        axes[i].set_xlabel('Categories/Bins')
        axes[i].set_ylabel('Proportion')
        axes[i].legend(['Train', 'Test'])
        axes[i].grid(axis='y')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout for better readability
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def validate_joint_distributions(train, test, welding_seam, variables):
    """
    Validates the joint distribution of specified variables for a given welding seam.

    :param train: DataFrame containing the training set.
    :param test: DataFrame containing the test set.
    :param welding_seam: Welding seam value to filter on (e.g., 'c19').
    :param variables: List of variables to include in the joint distribution.
    """
    # Filter datasets for the specific welding seam
    train_filtered = train[train['welding-seams'] == welding_seam].copy()
    test_filtered = test[test['welding-seams'] == welding_seam].copy()

    # Check if all variables are present
    for var in variables:
        if var not in train_filtered.columns or var not in test_filtered.columns:
            raise KeyError(f"{var} column missing in train or test data. Please check your preprocessing steps.")

    # Create joint combination columns in both datasets
    train_filtered['joint_combination'] = train_filtered[variables].astype(str).agg(', '.join, axis=1)
    test_filtered['joint_combination'] = test_filtered[variables].astype(str).agg(', '.join, axis=1)

    # Calculate joint distributions
    train_joint_dist = train_filtered['joint_combination'].value_counts(normalize=True)
    test_joint_dist = test_filtered['joint_combination'].value_counts(normalize=True)

    # Align indices to ensure comparability
    all_combinations = train_joint_dist.index.union(test_joint_dist.index)
    train_joint_dist = train_joint_dist.reindex(all_combinations, fill_value=0)
    test_joint_dist = test_joint_dist.reindex(all_combinations, fill_value=0)

    # Combine distributions into a single DataFrame
    combined = pd.DataFrame({
        'Train': train_joint_dist,
        'Test': test_joint_dist
    }).fillna(0)

    # Calculate absolute differences for debugging purposes
    combined['Difference'] = (combined['Train'] - combined['Test']).abs()

    # Plot the joint distribution comparison
    bar_width = 0.35
    index = range(len(all_combinations))

    plt.figure(figsize=(15, 7))
    plt.bar(index, combined['Train'], bar_width, label='Train', alpha=0.7, color='blue')
    plt.bar([i + bar_width for i in index], combined['Test'], bar_width, label='Test', alpha=0.7, color='orange')

    plt.xticks(index, all_combinations, rotation=90)
    plt.xlabel('Joint Combinations')
    plt.ylabel('Proportion')
    plt.title(f'Joint Distribution Comparison for Welding Seam: {welding_seam}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return combined
