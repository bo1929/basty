import argparse

import altair as alt
import basty.project.experiment_processing as experiment_processing
import basty.utils.misc as misc
import joblib as jl
import numpy as np
import pandas as pd
from altair_saver import save
from style import StyleEthogram

parser = argparse.ArgumentParser(
    description="Generate ethograms given categorical labels such as annotations and predictions."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
parser.add_argument("--extension", type=str, default="png", help="Desired file type.")
parser.add_argument(
    "--generate-annotation-ethograms",
    action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()


def plot_ethogram(expt_name, expt_record, y, y_col_name, date="2021-01-01"):
    label_to_behavior = expt_record.label_to_behavior
    domain_behaviors = list(label_to_behavior.values())

    cont_bouts = misc.cont_intvls(y)
    all_bouts = []
    for i in range(cont_bouts.shape[0] - 1):
        start = f"{date} {expt_record.get_hour_stamp(cont_bouts[i])}"
        finish = f"{date} {expt_record.get_hour_stamp(cont_bouts[i+1]+FPS*60)}"
        behavior = label_to_behavior[y[cont_bouts[i]]]
        all_bouts.append({y_col_name: behavior, "Start": start, "Finish": finish})
    for behavior in domain_behaviors:
        start = f"{date} {expt_record.get_hour_stamp(0)}"
        finish = f"{date} {expt_record.get_hour_stamp(0)}"
        all_bouts.append({y_col_name: behavior, "Start": start, "Finish": finish})
    df_bouts = pd.DataFrame(all_bouts)

    ethogram_chart = (
        alt.Chart(df_bouts)
        .mark_bar()
        .encode(
            x="hoursminutesseconds(Start):T",
            x2="hoursminutesseconds(Finish):T",
            y=f"{y_col_name}:N",
            color=alt.Color(
                f"{y_col_name}:N",
                scale=alt.Scale(
                    domain=domain_behaviors, scheme=StyleEthogram.colorscheme
                ),
            ),
        )
        .properties(
            title=f"Ethogram of Annotations for {expt_name}",
            width=1200,
            height=400,
        )
    )
    return ethogram_chart


def generate_annotation_ethograms(project_obj):
    alt.themes.register("ethograms_style", StyleEthogram.get_ethogram_style)
    alt.themes.enable("ethograms_style")

    for expt_name in project_obj.annotation_path_dict.keys():
        expt_path = project_obj.expt_path_dict[expt_name]
        expt_record = jl.load(expt_path / "expt_record.z")
        annotations = np.load(expt_path / "annotations.npy")

        ethogram_chart = plot_ethogram(expt_name, expt_record, annotations, "Behavior")
        save(
            ethogram_chart,
            str(
                expt_path
                / "figures"
                / f"{expt_name}_annotation_ethogram.{args.extension}"
            ),
        )
        print(expt_name)


if __name__ == "__main__":
    FPS = 30

    project = experiment_processing.Project(
        args.main_cfg_path,
    )

    if args.generate_annotation_ethograms:
        generate_annotation_ethograms(project)
