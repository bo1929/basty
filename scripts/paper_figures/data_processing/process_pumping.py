from scripts.paper_figures.data_processing.config import Args
from scripts.paper_figures.process_results import BehaviorData
import os
import pickle


class ProboscisProcessing:
    def __init__(self, process_data, llh, io_process, force_recalculate=False, filter_size=60, padding=30 * 18,
                 num_workers=60):
        self.process_data = process_data
        self.llh = llh
        self.io_process = io_process
        self.force_recalculate = force_recalculate
        self.args = Args()
        self.filter_size = filter_size
        self.padding = padding
        self.num_workers = num_workers
        self.bouts_dict = []
        self.masked_data = []
        self.behavior = 'ProboscisPumping'

    def process_pumping(self):
        self.process_data.process_expt_names_parallel(
            self.llh, self.io_process.get_binary_mask_subfolder(self.behavior)
        )

        masks_based_on_likelihood = self.io_process.load_binary_mask(self.behavior)
        beh_masks = self.process_data.create_binary_mask_from_behaviors(self.args.BEHAVIORS, self.behavior)
        masked_data = self.process_data.update_dictionary_with_final_masked(masks_based_on_likelihood, beh_masks)

        bouts_dict = BehaviorData.find_consecutive_bouts_and_snap_fts(masked_data, self.io_process, self.behavior,
                                                                      self.filter_size,
                                                                      self.padding,
                                                                      self.num_workers,
                                                                      self.force_recalculate)

        self.bouts_dict = bouts_dict

    def save_predicted_bouts(self):
        folder_path = self.io_process.get_prediction_result_folder(self.behavior)

        if not self.bouts_dict:
            raise ValueError("Bouts are not processed")

        with open(os.path.join(folder_path, 'bouts_dict.pkl'), 'wb') as f:
            pickle.dump(self.bouts_dict, f)

    def generate_prob_bouts(self, load_bouts=False):
        folder_path = self.io_process.get_prediction_result_folder(self.behavior)
        bouts_file_path = os.path.join(folder_path, 'bouts_dict.pkl')

        if load_bouts:
            if os.path.exists(bouts_file_path):
                with open(bouts_file_path, 'rb') as f:
                    self.bouts_dict = pickle.load(f)
            else:
                raise FileNotFoundError(
                    f"The bouts_dict.pkl file does not exist in the specified folder: {folder_path}")
        else:
            self.process_pumping()

        return self.bouts_dict

