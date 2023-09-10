import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.ndimage import median_filter as medfilt
import concurrent.futures
import pandas as pd
import pickle
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from input import Input
import multiprocessing


class BehaviorData:
    def __init__(
        self,
        data,
        fps=30,
        behaviors=None,
        binary_mask_threshold=0.8,
        window_size_median_filter=6,
    ):
        self.FPS = fps
        self.BEHAVIORS = behaviors if behaviors else []
        self.data = data
        self.binary_mask_threshold = binary_mask_threshold
        self.window_size_median_filter = window_size_median_filter

    @staticmethod
    def get_time_stamp(
        idx,
        date="2022-01-01",
        FPS=30,
    ):
        sec_total = idx // FPS
        second = sec_total % 60
        minute = (sec_total // 60) % 60
        hour = sec_total // 3600
        stamp = str(int(hour)) + ":" + str(int(minute)) + ":" + str(int(second))
        return f"{date} {stamp}"

    def plot_raw_behavior(self, data, behavior, sd):
        g = sns.relplot(
            data=data,
            x="Idx",
            y=behavior,
            row="ExptNames",
            kind="line",
            palette="crest",
            height=2,
            aspect=10,
        )

        for expt_name, ax in g.axes_dict.items():
            ax.text(0.8, 0.85, expt_name, transform=ax.transAxes, fontweight="bold")
            ax.xaxis.grid(True)
        if sd == False:
            xticks = np.arange(
                start=0, stop=self.FPS * 60 * 60 * 16 + 1, step=self.FPS * 60 * 60 * 2
            )
        else:
            xticks = np.arange(
                start=0, stop=self.FPS * 60 * 60 * 6 + 1, step=self.FPS * 60 * 60 * 2
            )
        ax.set_xticks(xticks)
        zt_list = [
            "ZT" + str((tick + 10) % 24) for tick in range(0, len(xticks) * 2, 2)
        ]
        ax.set_xticklabels(zt_list)
        ax.set_xlabel("ZT Time")
        g.set_titles("")
        g.tight_layout()
        g.fig.subplots_adjust(top=0.97)
        g.fig.suptitle("Behavior: " + behavior)

    @staticmethod
    def _save_fig(name, behavior, fig_path):
        fig_name = os.path.join(fig_path, name + "behavior_" + behavior + ".pdf")
        plt.savefig(fig_name, dpi=300)


    def plot_all_behaviors(self, data, name, fig_path):
        for behavior in self.BEHAVIORS:
            self.plot_raw_behavior(data, behavior, False)
            self.save_fig(name, behavior, fig_path)

    @staticmethod
    def pivot_and_plot(data, name, fig_path, rate, BEHAVIORS):
        if rate[-1] == "S":
            td = pd.Timedelta(rate)
            seconds = td.total_seconds()
        elif rate[-1] == "T":
            td = pd.Timedelta(rate[-1] + "m")
            seconds = td.total_seconds()

        for behavior in BEHAVIORS:
            df_pivoted = data.pivot(
                index="ExptNames", columns="TimeStamp", values=behavior
            )
            a4_dims = (25.7, 5.27)
            fig, ax = plt.subplots(figsize=a4_dims)
            plt.title(behavior)
            ax = sns.heatmap(df_pivoted, cmap="YlGnBu")

            locator = matplotlib.ticker.IndexLocator(
                base=(2 * 60 * 60) / seconds, offset=0
            )
            ax.xaxis.set_major_locator(locator)
            _, ZT_ticklabels = BehaviorData.generate_tick_data()
            ax.set_xticklabels(ZT_ticklabels)
            BehaviorData._save_fig(name, behavior, fig_path)

    @staticmethod
    def resample_df(data, rate, BEHAVIORS):
        data_df_list = []
        for expt_name in data["ExptNames"].unique():
            sub_data = data[data["ExptNames"] == expt_name]
            data_ind_rs = sub_data[BEHAVIORS].resample(rate).mean()
            data_ind_rs["ExptNames"] = expt_name
            data_df_list.append(data_ind_rs)

            data_df_all_rs = pd.concat(data_df_list)
        return data_df_all_rs

    @staticmethod
    def generate_tick_data(FPS=30, sd=False):

        if sd == False:
            xticks = np.arange(
                start=0, stop=FPS * 60 * 60 * 16 + 1, step=FPS * 60 * 60 * 2
            )
        else:
            xticks = np.arange(
                start=0, stop=FPS * 60 * 60 * 6 + 1, step=FPS * 60 * 60 * 2
            )
        ZT_ticks = xticks
        ZT_ticklabels = [
            "ZT" + str((tick + 10) % 24) for tick in range(0, len(xticks) * 2, 2)
        ]
        return ZT_ticks, ZT_ticklabels

    @staticmethod
    def process_expt_name(
        expt_name, data, likelihood_data, threshold, window_size_median_filter, folder
    ):
        # Create a filename with ExptName, median_window_size, and threshold
        filename = (
            f"{expt_name}_median{window_size_median_filter}_threshold{threshold}.pkl"
        )
        file_path = os.path.join(folder, filename)

        # Check if the file already exists; if so, skip the calculation and return None
        if os.path.exists(file_path):
            print(f"File {filename} already exists. Skipping calculation.")
            return None

        # Perform the calculations as before
        sub_behavior_data = data[data["ExptNames"] == expt_name]
        sub_likelihood_data = likelihood_data[likelihood_data["ExptNames"] == expt_name]

        # Apply median filter to the data
        llh_filtered = BehaviorData._median_filter(sub_likelihood_data.prob.to_numpy(), window_size_median_filter)
        sub_filtered = BehaviorData._median_filter(sub_behavior_data.ProboscisPumping.to_numpy(), window_size_median_filter)

        # Resample the filtered data
        llh_filtered_resampled = BehaviorData._resample(llh_filtered, 1)
        sub_filtered_resampled = BehaviorData._resample(sub_filtered, 1)

        binary_mask = BehaviorData._create_binary_mask(
            llh_filtered_resampled, threshold
        )
        masked_filtered_resampled = sub_filtered_resampled * binary_mask

        temp_df = pd.DataFrame(
            {
                f"{expt_name}_unmasked": sub_filtered_resampled,
                f"{expt_name}_masked": masked_filtered_resampled,
            }
        )

        # Save the resulting DataFrame as a pickle file
        with open(file_path, "wb") as file:
            pickle.dump(temp_df, file)

        return temp_df

    def process_expt_names_parallel(self, likelihood_data, folder):
        unique_expt_names = self.data["ExptNames"].unique()
        result = pd.DataFrame()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    BehaviorData.process_expt_name,
                    expt_name,
                    self.data,
                    likelihood_data,
                    self.binary_mask_threshold,
                    self.window_size_median_filter,
                    folder,
                )
                for expt_name in unique_expt_names
            ]

        # for future in concurrent.futures.as_completed(futures):
        # temp_df = future.result()
        # if temp_df is not None:  # Only add the DataFrame to the result if it is not None (i.e., a new calculation)
        # result = pd.concat([result, temp_df], axis=1)

        # return result

    @staticmethod
    def _median_filter(array, window_size):
        filtered_array = medfilt(array, size=window_size)
        return filtered_array

    @staticmethod
    def _resample(array, resampling_factor):
        resampled_array = array[::resampling_factor]
        return resampled_array

    @staticmethod
    def _create_binary_mask(array, threshold):
        array = np.asarray(array)
        if array.ndim > 1:
            raise ValueError("Input array should be 1-dimensional.")
        binary_mask = np.where(array >= threshold, 1, 0)
        return binary_mask

    def create_binary_mask_from_behaviors(self, behaviors, target_behavior):
        if target_behavior not in behaviors:
            raise ValueError("Target behavior must be in the list of behaviors.")

        # Filter the data to only include the specified behaviors and the ExptNames column
        filtered_data = self.data[behaviors + ["ExptNames"]]

        # Group the data by ExptName
        grouped_data = filtered_data.groupby("ExptNames")

        # Initialize a dictionary to store the binary masks for each ExptName
        binary_masks = {}

        for expt_name, group in grouped_data:
            # Determine the column label with the highest value for each row (frame)
            max_column_label = group[behaviors].idxmax(axis=1)

            # Create a binary mask based on the target_behavior
            binary_mask = max_column_label.apply(
                lambda x: 1 if x == target_behavior else 0
            )

            # Reset the index of the binary mask and drop the original index
            binary_mask = binary_mask.reset_index(drop=True)

            # Add the binary mask to the dictionary with the key as the ExptName
            binary_masks[expt_name] = binary_mask

        return binary_masks

    @staticmethod
    def update_dictionary_with_final_masked(dict_of_dfs, mask_dict):
        updated_dict = dict_of_dfs.copy()

        for expt_name in updated_dict.keys():
            # Get the corresponding mask for the current ExptName
            mask = mask_dict[expt_name]

            # Multiply the '_masked' column with the mask and create a new '_final_masked' column
            updated_dict[expt_name][f"{expt_name}_final_masked"] = (
                updated_dict[expt_name][f"{expt_name}_masked"] * mask
            )

        return updated_dict

    @staticmethod
    def find_consecutive_bouts(data_dict, filter_size=60, padding=0, window_size=180):
        bouts_dict = {}

        for expt_name, df in data_dict.items():
            # Get the column ending with "_final_masked"
            final_masked_col = [col for col in df.columns if col.endswith("_final_masked")][0]

            # Apply median filter to smooth out the binary signal
            filtered_signal = medfilt(df[final_masked_col], size=filter_size)

            # Find the bouts of consecutive 1s with a maximum break of up to window_size
            bouts = []
            start_index = None
            zero_count = 0
            for index, value in enumerate(filtered_signal):
                if value >= 0.1:
                    if start_index is None:
                        start_index = index
                    zero_count = 0
                else:
                    zero_count += 1
                    if zero_count > window_size:
                        if start_index is not None:
                            bouts.append((start_index, index - zero_count))
                            start_index = None
                            zero_count = 0

            if start_index is not None:
                bouts.append((start_index, index))

            # Add padding to the bouts if specified
            if padding > 0:
                padded_bouts = [(max(0, start - padding), min(df.index[-1], end + padding)) for start, end in bouts]
            else:
                padded_bouts = bouts

            # Merge close intervals
            merged_bouts = []
            current_start = padded_bouts[0][0]
            current_stop = padded_bouts[0][1]

            for i in range(1, len(padded_bouts)):
                next_start = padded_bouts[i][0]
                next_stop = padded_bouts[i][1]

                if next_start - current_stop <= window_size:
                    current_stop = next_stop
                else:
                    merged_bouts.append((current_start, current_stop))
                    current_start = next_start
                    current_stop = next_stop

            merged_bouts.append((current_start, current_stop))

            # Create a DataFrame for the bouts with start index, stop index, and region
            bouts_df = pd.DataFrame(merged_bouts, columns=["start_index", "stop_index"])
            bouts_df["region"] = expt_name

            # Add the DataFrame to the output dictionary
            bouts_dict[expt_name] = bouts_df

        return bouts_dict

    @staticmethod
    def find_consecutive_bouts_and_snap_fts(data_dict, io_process, behavior_name, filter_size=3, padding=0,
                                            num_workers=60, force_recalculate=False):
        bouts_dict = {}
        main_output_folder = os.path.join(io_process.project.project_path, 'connected_bouts_w_snap_fts', behavior_name)

        if not os.path.exists(main_output_folder):
            os.makedirs(main_output_folder)

        def process_expt_name(expt_name):
            output_path = os.path.join(main_output_folder, f"{expt_name}.pkl")

            if not force_recalculate and os.path.exists(output_path):
                with open(output_path, "rb") as file:
                    bouts_df = pickle.load(file)
            else:
                # Load the snap_stft data and column names for the current expt_name
                snap_data, col_names = Input.load_snap_fts(io_process, expt_name)

                snap_df = pd.DataFrame(snap_data)
                snap_df.rename(columns=col_names, inplace=True)

                # Find consecutive bouts for the current expt_name in data_dict
                all_bouts = BehaviorData.find_consecutive_bouts({expt_name: data_dict[expt_name]},
                                                                filter_size=filter_size, padding=padding)

                # Get the consecutive bouts for the current expt_name
                bouts_df = all_bouts[expt_name]

                # Initialize new columns in bouts_df for each column in snap_df
                for col in snap_df.columns:
                    bouts_df[col] = None

                # For each row in bouts_df, extract data from snap_df for the start and stop indexes
                for index, row in bouts_df.iterrows():
                    start_index, stop_index = row["start_index"], row["stop_index"]
                    snap_data = snap_df.loc[start_index:stop_index]

                    # Store the extracted data in the new columns of bouts_df
                    for col in snap_df.columns:
                        bouts_df.at[index, col] = snap_data[col].values

                # Save the resulting bouts_df as a pickle file in the specified folder
                with open(output_path, "wb") as file:
                    pickle.dump(bouts_df, file)

            return expt_name, bouts_df

        # Check the number of available CPUs
        available_cpus = multiprocessing.cpu_count()

        # Use the default number of workers if it's less than or equal to the available CPUs
        # Otherwise, use half of the available CPUs
        num_workers = num_workers if num_workers <= available_cpus else available_cpus // 2

        # Process expt_names in parallel and update the bouts_dict
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_expt_name, expt_name): expt_name for expt_name in data_dict.keys()}

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc="Processing Experiments"):
                expt_name, bouts_df = future.result()
                bouts_dict[expt_name] = bouts_df

        return bouts_dict

    @staticmethod
    def merge_close_intervals(data_dict, distance_threshold):
        merged_data_dict = {}

        for key, df in data_dict.items():
            merged_df = pd.DataFrame(columns=df.columns)
            current_start = df.loc[0, 'start_index']
            current_stop = df.loc[0, 'stop_index']

            for i in range(1, len(df)):
                next_start = df.loc[i, 'start_index']
                next_stop = df.loc[i, 'stop_index']

                if next_start - current_stop <= distance_threshold:
                    current_stop = next_stop
                else:
                    merged_df = merged_df.append({'start_index': current_start, 'stop_index': current_stop},
                                                 ignore_index=True)
                    current_start = next_start
                    current_stop = next_stop

            merged_df = merged_df.append({'start_index': current_start, 'stop_index': current_stop}, ignore_index=True)
            merged_data_dict[key] = merged_df

        return merged_data_dict

