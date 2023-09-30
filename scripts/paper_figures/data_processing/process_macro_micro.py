import os  # Added this import
import pandas as pd
import numpy as np
import joblib as jb  # Assuming this is how you're importing jb
from basty.utils import misc  # Assuming you have a module named misc with the cont_intvls function
import platform
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl


class ExperimentOutline:
    def __init__(self, project_path):
        self.project_path = project_path

    @staticmethod
    def _get_bout_df(labels, expt_name):
        cont_bouts = misc.cont_intvls(labels)
        outline_list = []
        for i in range(len(cont_bouts) - 1):
            behavior = labels[cont_bouts[i]]
            start_idx = cont_bouts[i]
            finish_idx = cont_bouts[i + 1]
            outline_list.append(
                {"Behavior": behavior, "Start": start_idx, "Finish": finish_idx, "Experiment Name": expt_name})
        bout_df = pd.DataFrame(outline_list)
        return bout_df

    @staticmethod
    def _get_outline_df(velocity, labels, expt_name):
        outline_df = pd.DataFrame({"Velocity": velocity, "Labels": labels})
        outline_df.reset_index(inplace=True)
        outline_df["Experiment Name"] = expt_name
        return outline_df
    @staticmethod
    def hard_mask(expt_name, velocity, labels):
        if expt_name in ["Fly11242021_F_B_SD_5D_9am", "Fly11192021_F_A_SD_5D_9am", "Fly11152020_F_SD","Fly11052021_F_A_SD_5D_8am"]:
            velocity, labels = velocity[1512000:], labels[1512000:]
        return velocity, labels

    def outline_expt(self, expt_name):
        # Check if the system is Windows
        is_windows = platform.system() == 'Windows'

        # Save the original PosixPath class if the system is Windows. This allows loading of data generated in Linux/OS system in Windows
        if is_windows:
            original_posix_path = pathlib.PosixPath

        try:
            # Apply the workaround only for this block of code and only if the system is Windows
            if is_windows:
                pathlib.PosixPath = pathlib.WindowsPath

            expt_path = os.path.join(self.project_path, expt_name)
            expt_record_path = os.path.join(expt_path, "expt_record.z")
            velocity_path = os.path.join(expt_path, "delta_stft.pkl")

            expt_record = jb.load(expt_record_path)
            velocity = pd.read_pickle(velocity_path).mean(axis=1).to_numpy()

            # Apply the hard mask here
            velocity, labels = ExperimentOutline.hard_mask(expt_name, velocity, expt_record.mask_active)

            outline_df = self._get_outline_df(velocity, labels, expt_name)

            # Also apply the hard mask for 'labels' that we use in bout_df
            labels = np.logical_not(expt_record.mask_dormant).astype(int)

            labels[labels == 1] = 1
            labels[np.logical_and(expt_record.mask_dormant, expt_record.mask_active)] = 2
            velocity, labels = ExperimentOutline.hard_mask(expt_name, velocity, labels)
            bout_df = self._get_bout_df(labels, expt_name)

        finally:
            # Restore the original PosixPath class if the system is Windows
            if is_windows:
                pathlib.PosixPath = original_posix_path

        return outline_df, bout_df

    # TODO Add resampling to this
    def outline_expt_all(self, expt_info_df):
        bout_df_list = []
        outline_df_list = []

        for expt_name in expt_info_df.ExptNames.unique():
            outline_df, bout_df = self.outline_expt(expt_name)

            outline_df_list.append(outline_df)
            bout_df_list.append(bout_df)

        bout_df_all = pd.concat(bout_df_list)
        outline_df_all = pd.concat(outline_df_list)
        return bout_df_all, outline_df_all

    import matplotlib.pyplot as plt
    import numpy as np

    @staticmethod
    def _plot_segments(ax, df, category_names, ZT_ticks, ZT_ticklabels):
        # Set Arial as the default font
        plt.rcParams['font.family'] = 'Arial'

        unique_experiments = [exp for exp in df['Experiment Name'].unique() if exp != "Fly06272022_M_5d_SD_A"]
        y_pos = np.arange(len(unique_experiments))

        ax.set_yticks(y_pos)
        ax.set_yticklabels(unique_experiments)

        # Define custom color map using three specific colors
        color_map = ['#377eb8', '#ff7f00', '#4daf4a']

        for i, experiment in enumerate(unique_experiments):
            sub_df = df[df['Experiment Name'] == experiment]
            for index, row in sub_df.iterrows():
                start, finish, behavior = row['Start'], row['Finish'], row['Behavior']
                color = color_map[category_names.index(behavior)]
                ax.barh(i, finish - start, left=start, color=color, edgecolor='none', rasterized=True)

        ax.legend(category_names, ncol=len(category_names), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
        ax.set_xticks(ZT_ticks)
        ax.set_xticklabels(ZT_ticklabels)

    @staticmethod
    def plot_ethogram(df, save_path=None):
        unique_combinations = df[['SD', 'Sex']].drop_duplicates().values.tolist()
        FPS = 30
        fig, axs = plt.subplots(len(unique_combinations), figsize=(15, 5 * len(unique_combinations)))

        for i, (sd_val, sex_val) in enumerate(unique_combinations):
            ZT_ticks, ZT_ticklabels = misc.generate_tick_data(FPS, sd=sd_val)
            ax = axs[i]
            sub_df = df[(df['SD'] == sd_val) & (df['Sex'] == sex_val)]
            ExperimentOutline._plot_segments(ax, sub_df, [0, 1, 2], ZT_ticks, ZT_ticklabels)
            ax.set_title(f"SD: {sd_val}, Sex: {sex_val}")

        if save_path:
            plt.savefig(save_path, format='pdf',dpi=150)
        else:
            plt.show()