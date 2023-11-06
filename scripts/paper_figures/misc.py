import pandas as pd
import os
import time
import moviepy.editor as mpy
from typing import List
import numpy as np
import joblib as jb
from basty.utils import misc


def process_files(expt_info_path, bouts_dict_path):
    """Creates a dataframe from predicted bouts and experimental info."""
    # Load the expt_info_df and bouts_dict from .pkl files
    expt_info_df = pd.read_pickle(expt_info_path)
    bouts_dict = pd.read_pickle(bouts_dict_path)

    # Initialize an empty list to store the results
    results = []

    # Go through each key in bouts_dict
    for key in bouts_dict.keys():
        # Grab the start_index, stop_index, and DataFrame index
        start_indices = bouts_dict[key]['start_index']
        stop_indices = bouts_dict[key]['stop_index']
        df_indices = bouts_dict[key].index

        # Grab the path from expt_info_df
        path = expt_info_df[expt_info_df['ExptNames'] == key]['Path'].values[0]

        # For each start and stop index, create a dictionary and append to results
        for start_index, stop_index, df_index in zip(start_indices, stop_indices, df_indices):
            result_dict = {'ExptName': key,
                           'start_index': start_index,
                           'stop_index': stop_index,
                           'df_index': df_index,
                           'Path': path}
            results.append(result_dict)

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Return the results
    return results_df


def slice_and_save_videos(results_df: pd.DataFrame, output_dir: str) -> List[str]:
    """Uses a dataframe with start stop and path information to slice predicted behavioral videos and saves them in the output folder"""
    saved_files = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, row in results_df.iterrows():
        video_files = [f for f in os.listdir(row['Path']) if f.endswith(('.mp4', '.avi'))]
        video_files.sort(key=lambda x: os.path.getsize(os.path.join(row['Path'], x)), reverse=True)

        if video_files:
            video_path = os.path.join(row['Path'], video_files[0])
            start_time = row['start_index'] / 30  # assuming 30fps
            stop_time = row['stop_index'] / 30  # assuming 30fps
            output_path = os.path.join(output_dir,
                                       f'{row["ExptName"]}_index_{row["df_index"]}_start_{row["start_index"]}_stop_{row["stop_index"]}.mp4')

            # Load video and slice it
            video = mpy.VideoFileClip(video_path)
            subclip = video.subclip(start_time, stop_time)

            # Measure the time taken to save the file
            start = time.time()
            subclip.write_videofile(output_path)
            end = time.time()

            print(f"Time taken to save the file {output_path}: {end - start} seconds")

            saved_files.append(output_path)

        else:
            print(f"No .mp4 or .avi files found in directory: {row['Path']}")

    return saved_files


def save_predicted_column_as_csv(data, column_name, output_folder):
    """This function reads a dataframe that contains predictions for a given behavior
    It then extracts the desired column to the output folder. Primarily used for generating
    training data
    # Example usage
    data = pd.read_pickle(r'Z:\mfk\basty-projects\bouts_dict.pkl')
    column_name = "distance.origin-prob"
    output_folder = os.path.dirname(r"Z:\mfk\basty-projects\bouts_to_csv")

    save_column_as_csv(data, column_name, output_folder)
"""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for key, df in data.items():
        output_path = os.path.join(output_folder, f"{key}.csv")
        if column_name in df.columns:
            time_series = df[column_name].apply(pd.Series)
            time_series = time_series.T

            time_series.to_csv(output_path, index=False)
        else:
            print(f"Column '{column_name}' does not exist in dataframe for key '{key}'")

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