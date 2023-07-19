import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


class Plotter:
    @staticmethod
    def plot_histograms(binary_mask_dataframes, lower_threshold=0, bins=50, filename="histograms.pdf"):
        num_plots = len(binary_mask_dataframes)
        plots_per_page = 32
        num_pages = math.ceil(num_plots / plots_per_page)
        plots_per_row = 4
        plots_per_col = 8
        figsize = (8.27, 11.69)  # A4 size in inches

        with PdfPages(filename) as pdf:
            for page in range(num_pages):
                fig, axes = plt.subplots(plots_per_col, plots_per_row, figsize=figsize, constrained_layout=True)

                for i in range(plots_per_col):
                    for j in range(plots_per_row):
                        idx = page * plots_per_page + i * plots_per_row + j

                        if idx >= num_plots:
                            break

                        expt_name = list(binary_mask_dataframes.keys())[idx]
                        df = binary_mask_dataframes[expt_name]

                        # Get the values from the dataframe and filter them based on the lower_threshold
                        values = df.values.flatten()
                        filtered_values = values[values >= lower_threshold]

                        # Plot the histogram for the current dataframe
                        axes[i, j].hist(filtered_values, bins=bins, alpha=0.5, label=expt_name)
                        axes[i, j].set_title(expt_name, fontsize=8)
                        axes[i, j].tick_params(axis="both", which="major", labelsize=6)

                # Save the current page to the PDF
                pdf.savefig(fig)

                # Close the figure to free up memory
                plt.close(fig)

    @staticmethod
    def plot_time_series(dict_of_dfs, feature, filename="multipage_proced_pdf.pdf"):
        """Plot predicted time series. This is used in the context of proboscis pumps"""
        # Initiate the pdf file
        with PdfPages(filename) as pdf:
            # Initialize counter for plots
            plot_counter = 0
            # Create a new figure for every 50 rows (plots)
            fig, axs = plt.subplots(10, 5, figsize=(15, 20))
            # For each dataframe in the dictionary
            for key, df in dict_of_dfs.items():
                # For each row in the dataframe
                for i, row in df.iterrows():
                    # Get current axis
                    ax = axs[plot_counter // 5, plot_counter % 5]

                    # Create the plot for the specific feature
                    ax.plot(np.arange(len(row[feature])) / 30, row[feature])  # divide by 30 to get seconds

                    # Set the xticks
                    ax.set_xticks(np.arange(0, len(row[feature]) / 30, step=5))  # adjust the step as needed

                    # Set the title
                    ax.set_title(f'{key}: start={row["start_index"]}, stop={row["stop_index"]}')

                    plot_counter += 1

                    # If we've hit the limit of 50 plots per pdf page, save and create a new page
                    if plot_counter % 50 == 0:
                        # Tightly layout the plots
                        plt.tight_layout()

                        # Save the current figure to the pdf
                        pdf.savefig(fig)

                        # Close the figure to free up memory
                        plt.close(fig)

                        # Create a new figure for every 50 rows (plots)
                        fig, axs = plt.subplots(10, 5, figsize=(15, 20))

                        # Reset the plot counter
                        plot_counter = 0

                # If we're at the end of the dictionary, and it's not a full page, save the remaining plots
                if plot_counter % 50 != 0 and key == list(dict_of_dfs.keys())[-1]:
                    # Tightly layout the plots
                    plt.tight_layout()

                    # Save the current figure to the pdf
                    pdf.savefig(fig)

                    # Close the figure to free up memory
                    plt.close(fig)
