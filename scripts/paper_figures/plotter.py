import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Plotter:
    @staticmethod
    def plot_histograms(
        binary_mask_dataframes, lower_threshold=0, bins=50, filename="histograms.pdf"
    ):
        num_plots = len(binary_mask_dataframes)
        plots_per_page = 32
        num_pages = math.ceil(num_plots / plots_per_page)
        plots_per_row = 4
        plots_per_col = 8
        figsize = (8.27, 11.69)  # A4 size in inches

        with PdfPages(filename) as pdf:
            for page in range(num_pages):
                fig, axes = plt.subplots(
                    plots_per_col,
                    plots_per_row,
                    figsize=figsize,
                    constrained_layout=True,
                )

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
                        axes[i, j].hist(
                            filtered_values, bins=bins, alpha=0.5, label=expt_name
                        )
                        axes[i, j].set_title(expt_name, fontsize=8)
                        axes[i, j].tick_params(axis="both", which="major", labelsize=6)

                # Save the current page to the PDF
                pdf.savefig(fig)

                # Close the figure to free up memory
                plt.close(fig)
