import os

#TODO: PASS FIG PATH TO DOWNSTREAM
class Args:
    def __init__(self):
        self.CONFIG_PATH = r"Z:\mfk\basty-projects\main_cfg.yaml"
        self.PROJECT_PATH = os.path.dirname(self.CONFIG_PATH)
        self._get_tmp_result_folder()
        self.FPS = 30
        self.FIG_PATH = r"C:\Users\Grover\Documents\GitHub\deepsleepfigs"
        self.RESULTS_FOLDER = r"Z:\mfk\basty-projects\backup-allnohaltere\results\semisupervised_pair_kNN\predictions.15NN.neighbor_weights-distance.neighbor_weights_norm-log_count.activation-standard.voting-soft.voting_weights-None\exports"
        self.BODY_PART_SETS = {"ProboscisPumping": "prob", "Feeding": "prob", "HaltereSwitch": "halt"},
        self.BEHAVIORS = [
            "Idle&Other",
            "PosturalAdjustment&Moving",
            "Feeding",
            "Grooming",
            "ProboscisPumping",
        ]
        self.output_path = r'Z:\mfk\basty-projects'
        self._get_tmp_result_folder()

    def _get_tmp_result_folder(self):
        tmp_result_folder = os.path.join(self.PROJECT_PATH,'tmp_results')

        if not os.path.exists(tmp_result_folder):
            os.makedirs(tmp_result_folder)

        self.TMP_RESULT_FOLDER = tmp_result_folder
