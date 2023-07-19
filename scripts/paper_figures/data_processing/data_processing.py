from scripts.paper_figures.data_processing.config import Args
from scripts.paper_figures.data_processing.data_preperation import prepare_data
from scripts.paper_figures.process_results import BehaviorData
from basty.utils import misc

def process_data(args, df_data, expt_info_df, project):
    print("Loading Likelihood data...")
    # Loading Likelihood data
    llh = misc.get_likelihood(project.data_path_dict, args.CONFIG_PATH)
    llh = llh[llh["ExptNames"].isin(df_data.ExptNames.unique())]


    # Process Data
    processed_data = BehaviorData(df_data, binary_mask_threshold=0.8)
    return llh, processed_data

def main():
    args = Args()
    df_data, expt_info_df, project, io_process = prepare_data(args)
    llh, processed_data= process_data(args, df_data, expt_info_df, project)
    return llh, processed_data, io_process

if __name__ == "__main__":
    llh, processed_data, io_process = main()
