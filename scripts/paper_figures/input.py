import os
import pickle
import yaml
import pandas as pd
import glob
import basty.project.experiment_processing as experiment_processing
from basty.utils import misc


class Input:
    def __init__(self, project, results_folder, tmp_result_folder):
        self.project = project
        self.results_folder = results_folder
        self.tmp_result_folder = tmp_result_folder

    def load_snap_fts(self, expt_name):
        snap_path = os.path.join(self.project.project_path, expt_name, "snap_stft.pkl")
        snap_col_name = os.path.join(
            self.project.project_path, expt_name, "ftname_to_snapft.yaml"
        )

        with open(snap_path, "rb") as file:
            snap_data = pickle.load(file)

        with open(snap_col_name, "r") as file:
            col_names = yaml.safe_load(file)

        return snap_data, col_names

    def load_predictions(self):
        csv_files = [
            file for file in os.listdir(self.results_folder) if file.endswith(".csv")
        ]
        dfs = list()
        for file in csv_files:
            data = pd.read_csv(os.path.join(self.results_folder, file))
            data["ExptNames"] = os.path.splitext(file)[0]
            dfs.append(data)

        df_data = pd.concat(dfs, ignore_index=True)
        df_data = df_data.rename(columns={"Unnamed: 0": "Idx"})

        return df_data

    def load_expt_info(self, flysexpath=r"Y:\DeepSleepPaperData\flysexinfo.yaml"):
        data_path_dict = self.project.data_path_dict
        df = misc.parse_experiment_names(data_path_dict)
        df = misc.update_expt_info_df(df, flysexpath)

        return df

    def create_binary_masks_subfolders(self, BEHAVIORS):
        binary_masks_folder = os.path.join(self.tmp_result_folder, "binary_masks")

        if not os.path.exists(binary_masks_folder):
            os.makedirs(binary_masks_folder)

        for behavior in BEHAVIORS:
            behavior_subfolder = os.path.join(binary_masks_folder, behavior)

            if not os.path.exists(behavior_subfolder):
                os.makedirs(behavior_subfolder)

    def get_binary_mask_subfolder(self, behavior):
        binary_masks_folder = os.path.join(self.tmp_result_folder, "binary_masks")
        behavior_subfolder = os.path.join(binary_masks_folder, behavior)

        if not os.path.exists(behavior_subfolder):
            raise ValueError(f"The subfolder for behavior '{behavior}' does not exist.")

        return behavior_subfolder

    def load_binary_mask(self, subfolder_name):
        # Get the path to the subfolder within the binary masks folder
        binary_masks_subfolder = self.get_binary_mask_subfolder(subfolder_name)

        # Find all the pickle files in the subfolder
        pickle_files = glob.glob(os.path.join(binary_masks_subfolder, "*.pkl"))

        # Initialize an empty dictionary to store the dataframes
        binary_mask_dataframes = {}

        # Load each binary mask dataframe and store it in the dictionary
        for file in pickle_files:
            with open(file, "rb") as f:
                df = pickle.load(f)

            # Extract the ExptName from the file name
            file_basename = os.path.basename(file)
            expt_name = file_basename[: file_basename.find("_median")]

            # Store the dataframe in the dictionary with the ExptName as the key
            binary_mask_dataframes[expt_name] = df

        return binary_mask_dataframes

    def get_prediction_result_folder(self, behavior):
        predictions_folder = os.path.join(self.tmp_result_folder, "predictions")
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)

        behavior_folder = os.path.join(predictions_folder, behavior)
        if not os.path.exists(behavior_folder):
            os.makedirs(behavior_folder)

        return behavior_folder



