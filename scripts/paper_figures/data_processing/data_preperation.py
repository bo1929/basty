import pandas as pd
from scripts.paper_figures.input import Input
import os

import basty.project.experiment_processing as experiment_processing

def prepare_data(args):
    project = experiment_processing.Project(args.CONFIG_PATH)
    args.FPS = project.fps

    print("Data Loading and Preprocessing...")
    # Data Loading and Preprocessing
    io_process = Input(project, args.RESULTS_FOLDER, args.TMP_RESULT_FOLDER)
    expt_info_df = io_process.load_expt_info()
    df_data = io_process.load_predictions()
    df_data.drop(["HaltereSwitch", "Noise"], axis=1, inplace=True)
    io_process.create_binary_masks_subfolders(args.BEHAVIORS)

    print("Saving Experiment Info...")
    # Saving intermediate result if it does not exist
    file_path = os.path.join(args.output_path,'expt_info_df.pkl')
    if not os.path.exists(file_path):
        pd.to_pickle(expt_info_df, file_path)
        print(f"Data saved to {file_path}")
    else:
        print("Data already exists. No data was saved.")

    return df_data, expt_info_df, project, io_process
