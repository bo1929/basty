import numpy as np
from scipy.signal import medfilt
from scipy import signal
import pandas as pd
import multiprocessing as mp

class ProboscisAnalysis:
    def __init__(self, data_dict):
        # Initialize with a dictionary of dataframes
        self.data_dict = data_dict

    def preprocess_data(self, df, feature_name):
        # Get the data from the specific feature
        sigs = df[feature_name]

        processed_sigs = []
        for sig in sigs:
            # Apply median filter
            sig = medfilt(sig, kernel_size=11)

            if np.abs(np.max(sig * -1)) == np.min(sig):
                sig = -sig + np.max(sig) * 2

            # Normalize to [0, 1]
            sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig))

            processed_sigs.append(sig)

        return processed_sigs

    def preprocess_all_data(self):
        # Create an empty dictionary to hold the preprocessed data
        preprocessed_dict = {}

        # Loop over each experiment in the data dictionary
        for expt_name, df in self.data_dict.items():
            # Create a copy of df to avoid changing original dataframe
            df_copy = df.copy()

            # Loop over each column (feature) in the dataframe, excluding "start", "stop", "region"
            for feature_name in df.columns:
                if feature_name not in ["start_index", "stop_index", "region"]:
                    # Preprocess the data
                    sig = self.preprocess_data(df, feature_name)
                    df_copy[feature_name] = sig  # replace the original data with preprocessed data

            # Add the preprocessed dataframe to the preprocessed_dict
            preprocessed_dict[expt_name] = df_copy

        return preprocessed_dict

    @staticmethod
    def segment_timeseries(series, length):
        """Divide a series into multiple segments of a given length.

        Args:
            series (pd.Series): The time series to divide.
            length (int): The length of each segment.

        Returns:
            list: A list of segments, each segment is a pd.Series object.
        """
        segments = [series[i:i + length] for i in range(0, len(series) - length + 1, 15)]
        return segments

    def segment_timeseries_column(self, df, column_name, length):
        """Segment all time series in a given column of a dataframe and add a new column with segmented series.

        Args:
            df (pd.DataFrame): The dataframe with the time series.
            column_name (str): The name of the column with the time series.
            length (int): The length of each segment.

        Returns:
            pd.DataFrame: A new dataframe with the segmented time series.
        """
        segmented_series = [self.segment_timeseries(series, length) for series in df[column_name]]
        df[f'{column_name}_segmented'] = segmented_series  # create a new column for segmented series
        return df

    def find_shortest_series(self, column_name):
        """Find the length of the shortest series in a dictionary of dataframes.

        Args:
            column_name (str): The name of the column to consider.

        Returns:
            The length of the shortest series in the data.
        """
        min_length = float('inf')

        for df in self.data_dict.values():
            if column_name in df.columns:
                for series in df[column_name]:
                    series_length = len(series)
                    if series_length < min_length:
                        min_length = series_length

        return min_length
    @staticmethod
    def calc_spectral_density(series, fs=30):
        """Calculate the spectral density of a time series.

        Args:
            series (pd.Series or np.array): The time series.
            fs (int, optional): The sampling frequency. Defaults to 30.

        Returns:
            tuple: A tuple containing the frequencies and corresponding power spectral density.
        """
        frequencies, PSD = signal.periodogram(series, fs)
        return frequencies, PSD

    @staticmethod
    def calculate_spectral_density_for_dataframe(df_chunk):
        """Calculate spectral density for all time series in a specific column for a dataframe.

        Args:
            df_chunk (pd.DataFrame): The dataframe chunk with the time series.

        Returns:
            pd.DataFrame: A dataframe with the original data and an additional column with the PSDs for each row.
        """
        if "time_series" not in df_chunk.columns:
            return df_chunk

        df_chunk['PSDs'] = df_chunk['time_series'].apply(ProboscisAnalysis.calc_spectral_density)
        return df_chunk

    @staticmethod
    def parallelize_dataframe(df, func, n_cores=32):
        """Split dataframe into chunks and apply function in parallel.

        Args:
            df (pd.DataFrame): The dataframe to process.
            func (function): The function to apply.
            n_cores (int, optional): The number of cores to use. Defaults to 4.

        Returns:
            pd.DataFrame: The processed dataframe.
        """
        df_split = np.array_split(df, n_cores)
        pool = mp.Pool(n_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df

    def unpack_dataframes(self, column_name):
        """Unpack time series from a specific column into a single dataframe.

        Args:
            column_name (str): The name of the column with the lists of lists.

        Returns:
            pd.DataFrame: A new dataframe with one time series per row, along with the associated key, original row index, and list index.
        """
        unpacked_data = []
        for key, df in self.data_dict.items():
            for row_idx, series_list in df[column_name].iteritems():
                for list_idx, series in enumerate(series_list):
                    unpacked_data.append({
                        "time_series": series,
                        "key": key,
                        "original_row": row_idx,
                        "list_index": list_idx
                    })
        return pd.DataFrame(unpacked_data)


