import matplotlib.pyplot as plt
import pandas as pd
class FilterData:
    def __init__(self, behavior_df, likelihood_df):
        self.behavior_df = behavior_df
        self.likelihood_df = likelihood_df

    def filter_by_likelihood(self, threshold):
        valid_expt_names = self.likelihood_df[self.likelihood_df.mean(axis=1) >= threshold]['ExptNames'].unique()
        filtered_df = self.behavior_df[self.behavior_df['ExptNames'].isin(valid_expt_names)]
        return filtered_df

    def calculate_mean_likelihood(self, *body_parts):
        if not body_parts:
            return self.likelihood_df.mean(axis=1)
        else:
            selected_columns = [column for column in self.likelihood_df.columns if any(part in column for part in body_parts)]
            return self.likelihood_df[selected_columns].mean(axis=1)

    def plot_mean_likelihood(self):
        body_part_columns = [column for column in self.likelihood_df.columns if column != 'ExptNames']
        n_body_parts = len(body_part_columns)
        fig, axes = plt.subplots(nrows=n_body_parts, ncols=1, figsize=(10, n_body_parts * 4), sharex=True)

        for ax, body_part in zip(axes, body_part_columns):
            mean_likelihood = self.likelihood_df.groupby('ExptNames')[body_part].mean()
            mean_likelihood.plot(kind='bar', ax=ax)
            ax.set_title(body_part)
            ax.set_ylabel('Mean Likelihood')

        plt.xlabel('ExptNames')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_binary_masks(df, col_dict, threshold):
        mask_data = {}
        for key, values in col_dict.items():
            if not isinstance(values, list):
                values = [values]
            avg_values = df[values].mean(axis=1)
            mask = avg_values >= threshold
            mask_data[key] = mask

        binary_masks_df = pd.DataFrame(mask_data)
        binary_masks_df['ExptNames'] = df['ExptNames']
        return binary_masks_df

    @staticmethod
    def apply_binary_masks(binary_masks_df, data_df):
        masked_data_df = data_df.copy()
        for column in binary_masks_df.columns:
            if column != 'ExptNames':
                masked_data_df[column] = data_df[column].where(binary_masks_df[column], other=None)
        return masked_data_df
