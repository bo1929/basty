import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.ndimage import median_filter as medfilt
import concurrent.futures
import pickle

class BehaviorData:
    def __init__(self, data, fps=30, behaviors=None, binary_mask_threshold=0.8, window_size_median_filter=6):
        self.FPS = fps
        self.BEHAVIORS = behaviors if behaviors else []
        self.data = data
        self.binary_mask_threshold = binary_mask_threshold
        self.window_size_median_filter = 6
    @staticmethod
    def get_time_stamp(idx, date="2022-01-01",FPS=30,):
        sec_total = idx // FPS
        second = sec_total % 60
        minute = (sec_total // 60) % 60
        hour = sec_total // 3600
        stamp = str(int(hour)) + ":" + str(int(minute)) + ":" + str(int(second))
        return f"{date} {stamp}"

    def plot_raw_behavior(self, data, behavior, sd):
        g = sns.relplot(
            data=data,
            x='Idx',
            y=behavior,
            row='ExptNames',
            kind='line',
            palette='crest',
            height=2,
            aspect=10
        )

        for expt_name, ax in g.axes_dict.items():
            ax.text(.8, .85, expt_name, transform=ax.transAxes, fontweight='bold')
            ax.xaxis.grid(True)
        if sd == False:
            xticks = np.arange(start=0, stop=self.FPS * 60 * 60 * 16 + 1, step=self.FPS * 60 * 60 * 2)
        else:
            xticks = np.arange(start=0, stop=self.FPS * 60 * 60 * 6 + 1, step=self.FPS * 60 * 60 * 2)
        ax.set_xticks(xticks)
        zt_list = ['ZT' + str((tick + 10) % 24) for tick in range(0, len(xticks) * 2, 2)]
        ax.set_xticklabels(zt_list)
        ax.set_xlabel('ZT Time')
        g.set_titles("")
        g.tight_layout()
        g.fig.subplots_adjust(top=0.97)
        g.fig.suptitle('Behavior: ' + behavior)

    @staticmethod
    def _save_fig(name, behavior, fig_path):
        fig_name = os.path.join(fig_path, name + 'behavior_' + behavior + '.pdf')
        #svg_name = os.path.join(fig_path, name + 'behavior_' + behavior + '.svg')
        plt.savefig(fig_name, dpi=150)
        #plt.savefig(svg_name)

    def plot_all_behaviors(self, data, name, fig_path):
        for behavior in self.BEHAVIORS:
            self.plot_raw_behavior(data, behavior, False)
            self.save_fig(name, behavior, fig_path)

    @staticmethod
    def pivot_and_plot( data, name, fig_path, rate, BEHAVIORS):
        if rate[-1] == 'S':
            td = pd.Timedelta(rate)
            seconds = td.total_seconds()
        elif rate[-1] == 'T':
            td = pd.Timedelta(rate[-1] + 'm')
            seconds = td.total_seconds()

        for behavior in BEHAVIORS:
            df_pivoted = data.pivot(index='ExptNames', columns='TimeStamp', values=behavior)
            a4_dims = (25.7, 5.27)
            fig, ax = plt.subplots(figsize=a4_dims)
            plt.title(behavior)
            ax = sns.heatmap(df_pivoted, cmap='YlGnBu')

            locator = matplotlib.ticker.IndexLocator(base=(2 * 60 * 60) / seconds, offset=0)
            ax.xaxis.set_major_locator(locator)
            _, ZT_ticklabels = BehaviorData.generate_tick_data()
            ax.set_xticklabels(ZT_ticklabels)
            BehaviorData._save_fig(name, behavior, fig_path)

    @staticmethod
    def resample_df(data, rate, BEHAVIORS):
        data_df_list = []
        for expt_name in data['ExptNames'].unique():
            sub_data = data[data['ExptNames'] == expt_name]
            data_ind_rs = sub_data[BEHAVIORS].resample(rate).mean()
            data_ind_rs['ExptNames'] = expt_name
            data_df_list.append(data_ind_rs)

            data_df_all_rs = pd.concat(data_df_list)
        return data_df_all_rs


    @staticmethod
    def generate_tick_data(FPS=30,sd=False):

        if sd == False:
            xticks = np.arange(start=0, stop=FPS * 60 * 60 * 16 + 1, step=FPS * 60 * 60 * 2)
        else:
            xticks = np.arange(start=0, stop=FPS * 60 * 60 * 6 + 1, step=FPS * 60 * 60 * 2)
        ZT_ticks = xticks
        ZT_ticklabels = ['ZT' + str((tick+10)%24) for tick in range(0,len(xticks)*2,2)]
        return ZT_ticks, ZT_ticklabels

    @staticmethod
    def process_expt_name(expt_name, data, likelihood_data, threshold, window_size_median_filter, folder):
        # Create a filename with ExptName, median_window_size, and threshold
        filename = f"{expt_name}_median{window_size_median_filter}_threshold{threshold}.pkl"
        file_path = os.path.join(folder, filename)

        # Check if the file already exists; if so, skip the calculation and return None
        if os.path.exists(file_path):
            print(f"File {filename} already exists. Skipping calculation.")
            return None

        # Perform the calculations as before
        sub_behavior_data = data[data['ExptNames'] == expt_name]
        sub_likelihood_data = likelihood_data[likelihood_data['ExptNames'] == expt_name]

        llh_filtered_resampled = BehaviorData._resample(sub_likelihood_data.prob.to_numpy(), window_size_median_filter,
                                                        1)
        sub_filtered_resampled = BehaviorData._resample(sub_behavior_data.ProboscisPumping.to_numpy(),
                                                        window_size_median_filter, 1)

        binary_mask = BehaviorData._create_binary_mask(llh_filtered_resampled, threshold)
        masked_filtered_resampled = sub_filtered_resampled * binary_mask

        temp_df = pd.DataFrame({
            f'{expt_name}_unmasked': sub_filtered_resampled,
            f'{expt_name}_masked': masked_filtered_resampled
        })

        # Save the resulting DataFrame as a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(temp_df, file)

        return temp_df

    def process_expt_names_parallel(self, likelihood_data, folder):
        unique_expt_names = self.data['ExptNames'].unique()
        result = pd.DataFrame()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(BehaviorData.process_expt_name, expt_name, self.data, likelihood_data,
                                       self.binary_mask_threshold, self.window_size_median_filter, folder) for expt_name
                       in
                       unique_expt_names]

            for future in concurrent.futures.as_completed(futures):
                temp_df = future.result()
                if temp_df is not None:  # Only add the DataFrame to the result if it is not None (i.e., a new calculation)
                    result = pd.concat([result, temp_df], axis=1)

        return result


    @staticmethod
    def _median_filter(array, window_size):
        filtered_array = medfilt(array, size=window_size)
        return filtered_array
    @staticmethod
    def _resample(array, window_size, resampling_factor):
        filtered_array = BehaviorData._median_filter(array, window_size)
        resampled_array = filtered_array[::resampling_factor]
        return resampled_array
    @staticmethod
    def _create_binary_mask(array, threshold):
        array = np.asarray(array)
        if array.ndim > 1:
            raise ValueError("Input array should be 1-dimensional.")
        binary_mask = np.where(array >= threshold, 1, 0)
        return binary_mask