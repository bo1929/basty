import argparse

import altair as alt
import basty.project.experiment_processing as experiment_processing
import joblib as jl
import numpy as np
import pandas as pd
from altair_saver import save
from style import StyleEmbedding

alt.data_transformers.disable_max_rows()

parser = argparse.ArgumentParser(
    description="Generate scatter plots of behavioral embeddings."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
parser.add_argument("--extension", type=str, default="png", help="Desired file type.")
parser.add_argument(
    "--generate-unsupervised-disparate",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--generate-supervised-disparate",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--use-annotations-to-color",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--change-size-wrt-cardinality",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--change-opacity-wrt-cardinality",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--exclude-unannotated-frames",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--use-annotations-to-mask",
    action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()
args.use_annotations_to_visualize = (
    args.use_annotations_to_color
    or args.change_size_wrt_cardinality
    or args.change_opacity_wrt_cardinality
)


def plot_embedding(
    expt_name,
    expt_record,
    X,
    y,
    y_col_name,
):
    if expt_record.has_annotation:
        label_to_behavior = expt_record.label_to_behavior
        domain_behaviors = list(label_to_behavior.values())

    if expt_record.has_annotation and args.use_annotations_to_color:
        df_embedding = pd.DataFrame(
            {"UMAP-1": X[:, 0], "UMAP-2": X[:, 1], y_col_name: y}
        )
    else:
        df_embedding = pd.DataFrame({"UMAP-1": X[:, 0], "UMAP-2": X[:, 1]})

    chart_embedding = alt.Chart(df_embedding).mark_point(
        filled=StyleEmbedding.filled,
        size=StyleEmbedding.sizeDefault,
        opacity=StyleEmbedding.opacityDefault,
    )

    x_range = X[:, 0].max() - X[:, 0].min()
    y_range = X[:, 1].max() - X[:, 1].min()
    x_axis = alt.X(
        "UMAP-1:Q",
        axis=alt.Axis(
            tickMinStep=StyleEmbedding.tickMinStep,
            tickCount=x_range // StyleEmbedding.tickMinStep,
        ),
    )
    y_axis = alt.Y(
        "UMAP-2:Q",
        axis=alt.Axis(
            tickMinStep=StyleEmbedding.tickMinStep,
            tickCount=y_range // StyleEmbedding.tickMinStep,
        ),
    )
    encode_args = [x_axis, y_axis]

    if expt_record.has_annotation and args.use_annotations_to_color:
        encode_args.append(
            alt.Color(
                f"{y_col_name}:N",
                scale=alt.Scale(
                    domain=domain_behaviors, scheme=StyleEmbedding.colorscheme
                ),
            )
        )
    if expt_record.has_annotation and (
        args.change_size_wrt_cardinality or args.change_opacity_wrt_cardinality
    ):
        cardinality_field = alt.AggregatedFieldDef.from_dict(
            {"op": "count", "field": f"{y_col_name}", "as": "Cardinality"}
        )
        chart_embedding = chart_embedding.transform_joinaggregate(
            [cardinality_field],
            groupby=[f"{y_col_name}"],
        )
    if expt_record.has_annotation and args.change_size_wrt_cardinality:
        encode_args.append(
            alt.Size(
                "Cardinality:Q",
                scale=alt.Scale(
                    # type="log",
                    reverse=True,
                    rangeMax=StyleEmbedding.sizeMax,
                    rangeMin=StyleEmbedding.sizeMin,
                ),
                legend=None,
            )
        )
    if expt_record.has_annotation and args.change_opacity_wrt_cardinality:
        encode_args.append(
            alt.Opacity(
                "Cardinality:Q",
                scale=alt.Scale(
                    # type="log",
                    reverse=True,
                    rangeMax=StyleEmbedding.opacityMax,
                    rangeMin=StyleEmbedding.opacityMin,
                ),
                legend=None,
            )
        )

    chart_embedding = chart_embedding.encode(*encode_args).properties(
        title=f"{expt_name}",
        width=500,
        height=500,
    )
    return chart_embedding


def mask_and_get_values(X, annotations, expt_record):
    if args.use_annotations_to_mask:
        maskDAnn = np.logical_and(expt_record.mask_annotated, expt_record.mask_dormant)
        annotationsDAnn = annotations[maskDAnn]
        y_lbl = annotationsDAnn
    else:
        maskDA = np.logical_and(expt_record.mask_active, expt_record.mask_dormant)
        annotationsDA = annotations[maskDA]
        if args.exclude_unannotated_frames:
            X = X[annotationsDA != 0, :]
            annotationsDA = annotationsDA[annotationsDA != 0]
        y_lbl = annotationsDA
    if args.use_annotations_to_visualize:
        y = [expt_record.label_to_behavior[lbl] for lbl in y_lbl]
        y_col_name = "Behavior Annotation"
    return X, y, y_col_name


def generate_supervised_disparate(project_obj):
    alt.themes.register("embedding_style", StyleEmbedding.get_embedding_style)
    alt.themes.enable("embedding_style")
    for expt_name in project_obj.annotation_path_dict.keys():
        expt_path = project_obj.expt_path_dict[expt_name]
        expt_record = jl.load(expt_path / "expt_record.z")
        X = np.load(expt_path / "embeddings" / "supervised_disparate_embedding.npy")
        annotations = np.load(expt_path / "annotations.npy")

        y = None
        y_col_name = None

        if args.use_annotations_to_visualize or args.exclude_unannotated_frames:
            X, y, y_col_name = mask_and_get_values(X, annotations, expt_record)

        embedding_chart = plot_embedding(expt_name, expt_record, X, y, y_col_name)
        save(
            embedding_chart,
            str(
                expt_path
                / "figures"
                / f"{expt_name}_supervised-disparate-embedding.{args.extension}"
            ),
        )
        print(expt_name)


def generate_unsupervised_disparate(project_obj):
    alt.themes.register("embedding_style", StyleEmbedding.get_embedding_style)
    alt.themes.enable("embedding_style")

    for expt_name, expt_path in project_obj.expt_path_dict.items():
        expt_record = jl.load(expt_path / "expt_record.z")
        X = np.load(expt_path / "embeddings" / "unsupervised_disparate_embedding.npy")

        y = None
        y_col_name = None

        if (
            args.use_annotations_to_visualize or args.exclude_unannotated_frames
        ) and expt_record.has_annotation:
            annotations = np.load(expt_path / "annotations.npy")
            X, y, y_col_name = mask_and_get_values(X, annotations, expt_record)

        embedding_chart = plot_embedding(expt_name, expt_record, X, y, y_col_name)
        save(
            embedding_chart,
            str(
                expt_path
                / "figures"
                / f"{expt_name}_unsupervised-disparate-embedding.{args.extension}"
            ),
        )
        print(expt_name)


if __name__ == "__main__":
    FPS = 30

    project = experiment_processing.Project(
        args.main_cfg_path,
    )

    if args.generate_unsupervised_disparate:
        generate_unsupervised_disparate(project)
    if args.generate_supervised_disparate:
        generate_supervised_disparate(project)
