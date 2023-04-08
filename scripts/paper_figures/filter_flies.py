import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import medfilt
from concurrent.futures import ProcessPoolExecutor
import numpy as np


class FilterData:
    def __init__(self, behavior_df, likelihood_df):
        self.behavior_df = behavior_df
        self.likelihood_df = likelihood_df

    def filter_by_likelihood(self, threshold):
        valid_expt_names = self.likelihood_df[
            self.likelihood_df.mean(axis=1) >= threshold
        ]["ExptNames"].unique()
        filtered_df = self.behavior_df[
            self.behavior_df["ExptNames"].isin(valid_expt_names)
        ]
        return filtered_df

    def calculate_mean_likelihood(self, *body_parts):
        if not body_parts:
            return self.likelihood_df.mean(axis=1)
        else:
            selected_columns = [
                column
                for column in self.likelihood_df.columns
                if any(part in column for part in body_parts)
            ]
            return self.likelihood_df[selected_columns].mean(axis=1)

    def plot_mean_likelihood(self):
        body_part_columns = [
            column for column in self.likelihood_df.columns if column != "ExptNames"
        ]
        n_body_parts = len(body_part_columns)
        fig, axes = plt.subplots(
            nrows=n_body_parts, ncols=1, figsize=(10, n_body_parts * 4), sharex=True
        )

        for ax, body_part in zip(axes, body_part_columns):
            mean_likelihood = self.likelihood_df.groupby("ExptNames")[body_part].mean()
            mean_likelihood.plot(kind="bar", ax=ax)
            ax.set_title(body_part)
            ax.set_ylabel("Mean Likelihood")

        plt.xlabel("ExptNames")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_binary_masks(df, col_dict, threshold):
        mask_data = {}
        for key, values in col_dict.items():
            if not isinstance(values, list):
                values = [values]
            avg_values = df[values].mean(axis=1)
            mask = avg_values >= threshold
            mask_data[key] = mask

        binary_masks_df = pd.DataFrame(mask_data)
        binary_masks_df["ExptNames"] = df["ExptNames"]
        return binary_masks_df

    @staticmethod
    def apply_mask_to_group(group_data):
        binary_masks_df, data_df = group_data
        shared_columns = binary_masks_df.columns.intersection(data_df.columns)
        binary_masks_df = binary_masks_df[shared_columns]
        data_df = data_df[shared_columns]

        binary_masks_df = binary_masks_df.set_index("ExptNames")
        masked_data_df = data_df.set_index("ExptNames")

        for column in binary_masks_df.columns:
            masked_data_df[column] = masked_data_df[column].where(
                binary_masks_df[column], other=None
            )

        masked_data_df.reset_index(inplace=True)
        return masked_data_df

    @staticmethod
    def apply_binary_masks(binary_masks_df, data_df, n_workers=None):
        # Group the DataFrames by unique ExptNames
        binary_masks_groups = binary_masks_df.groupby("ExptNames")
        data_groups = data_df.groupby("ExptNames")

        # Zip the groups together for parallel processing
        groups = [
            (binary_masks_groups.get_group(name), data_groups.get_group(name))
            for name in binary_masks_groups.groups.keys()
        ]

        # Use ThreadPoolExecutor to parallelize the process
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            masked_data_dfs = list(executor.map(FilterData.apply_mask_to_group, groups))

        # Concatenate the resulting DataFrames
        masked_data_df = pd.concat(masked_data_dfs)
        return masked_data_df

    @staticmethod
    def _apply_median_filter_resample(args):
        column_name, data, filter_size, resampling_ratio = args
        # Apply median filter
        filtered_data = medfilt(data, kernel_size=filter_size)

        # Resample the data
        resampled_data = filtered_data[::resampling_ratio]

        return column_name, resampled_data

    @staticmethod
    def median_filter_resample(
        df, filter_size, resampling_ratio, sampling_rate, n_workers=None
    ):
        # Prepare inputs for parallel processing
        inputs = [
            (col, df[col], filter_size, resampling_ratio)
            for col in df.columns
            if col != "ExptNames"
        ]

        # Apply median filter and resample in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            resampled_columns = list(
                executor.map(FilterData._apply_median_filter_resample, inputs)
            )

        # Create a new DataFrame with resampled data
        resampled_df = pd.DataFrame(dict(resampled_columns))
        resampled_df.index = pd.to_timedelta(
            np.arange(0, len(resampled_df)) / sampling_rate, unit="s"
        )

        # Add 'ExptNames' column if it exists in the original DataFrame
        if "ExptNames" in df.columns:
            resampled_df["ExptNames"] = df["ExptNames"]

        return resampled_df
