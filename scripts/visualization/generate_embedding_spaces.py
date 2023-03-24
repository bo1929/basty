import argparse

import altair as alt
import joblib as jl
import numpy as np
import pandas as pd
from altair_saver import save
from style import StyleEmbedding

import basty.project.experiment_processing as experiment_processing
import basty.utils.misc as misc

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
    "--generate-semisupervised-pair",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--generate-joint-charts",
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
    "--use-annotations-to-color",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--exclude-unannotated-frames",
    action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()
args.use_annotations_to_visualize = (
    args.use_annotations_to_color
    or args.change_size_wrt_cardinality
    or args.change_opacity_wrt_cardinality
)

# There is only a single fly in the behavioral space, so this is not possible.
if args.generate_supervised_disparate or args.generate_unsupervised_disparate:
    args.generate_joint_charts = False

name_suffix = ""
name_suffix += "-wAnnFrames" if args.exclude_unannotated_frames else ""
name_suffix += "-wAnnColor" if args.use_annotations_to_color else ""
name_suffix += "-wAnnOpacity" if args.change_opacity_wrt_cardinality else ""
name_suffix += "-wAnnSize" if args.change_size_wrt_cardinality else ""
name_suffix += "-JointChart" if args.generate_joint_charts else ""


def get_mask_suffix(embedding_name, expt_record):
    return "-DAnn" if expt_record.use_annotations_to_mask[embedding_name] else "-DA"


def mask_and_get_labels(X, expt_path, embedding_name, expt_record):
    y = None
    y_col_name = None
    if expt_record.has_annotation:
        annotations = np.load(expt_path / "annotations.npy")
        if expt_record.use_annotations_to_mask[embedding_name]:
            maskDAnn = np.logical_and(
                expt_record.mask_annotated, expt_record.mask_dormant
            )
            annotationsDAnn = annotations[maskDAnn]
            y_ann = annotationsDAnn
        else:
            maskDA = np.logical_and(expt_record.mask_active, expt_record.mask_dormant)
            annotationsDA = annotations[maskDA]
            y_ann = annotationsDA
            if args.exclude_unannotated_frames:
                X = X[annotationsDA != 0, :]
                y_ann = y_ann[annotationsDA != 0]
        if args.use_annotations_to_visualize:
            y = [expt_record.label_to_behavior[lbl] for lbl in y_ann]
            y_col_name = "Behavior Annotation"
    else:
        if args.use_annotations_to_visualize:
            y = ["Unannotated" for _ in y_ann]
            y_col_name = "Behavior Annotation"
            expt_record.has_annotation = True
            expt_record.label_to_behavior[-1] = "Unannotated"

    return X, y, y_col_name, expt_record


def plot_embedding(
    X,
    y,
    y_col_name,
    expt_record,
    title=None,
):
    if expt_record.has_annotation:
        label_to_behavior = expt_record.label_to_behavior
        domain_y = list(label_to_behavior.values())

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
                scale=alt.Scale(domain=domain_y, scheme=StyleEmbedding.colorscheme),
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
                    # reverse=True,
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
                    # reverse=True,
                    rangeMax=StyleEmbedding.opacityMax,
                    rangeMin=StyleEmbedding.opacityMin,
                ),
                legend=None,
            )
        )
    chart_embedding = chart_embedding.encode(*encode_args).properties(
        title=title,
        width=500,
        height=500,
    )
    return chart_embedding


def generate_semisupervised_pair(project_obj):
    alt.themes.register("embedding_style", StyleEmbedding.get_embedding_style)
    alt.themes.enable("embedding_style")

    annotated_expt_names = list(project_obj.annotation_path_dict.keys())
    all_expt_names = list(project_obj.expt_path_dict.keys())
    unannotated_expt_names = list(set(all_expt_names) - set(annotated_expt_names))
    if project_obj.evaluation_mode:
        unannotated_expt_names = annotated_expt_names
    pairs = [
        (name1, name2)
        for name1 in annotated_expt_names
        for name2 in unannotated_expt_names
        if name1 != name2
    ]
    for expt_name_ann, expt_name_unann in pairs:
        expt_path_unann = project_obj.expt_path_dict[expt_name_unann]
        expt_path_ann = project_obj.expt_path_dict[expt_name_ann]

        expt_record_unann = jl.load(expt_path_unann / "expt_record.z")
        expt_record_ann = jl.load(expt_path_ann / "expt_record.z")

        pair_name = f"{expt_name_ann}_{expt_name_unann}"

        ann_embedding_name = f"semisupervised_pair_embedding_{pair_name}"
        ann_embedding_dir = expt_path_ann / "embeddings"
        X_ann = np.load(ann_embedding_dir / f"{ann_embedding_name}.npy")
        X_ann, y_ann, y_col_name, expt_record_ann = mask_and_get_labels(
            X_ann, expt_path_ann, ann_embedding_name, expt_record_ann
        )

        unann_embedding_name = f"semisupervised_pair_embedding_{pair_name}"
        unann_embedding_dir = expt_path_unann / "embeddings"
        X_unann = np.load(unann_embedding_dir / f"{unann_embedding_name}.npy")
        X_unann, y_unann, y_col_name, expt_record_unann = mask_and_get_labels(
            X_unann, expt_path_unann, unann_embedding_name, expt_record_unann
        )

        if args.generate_joint_charts:
            expt_record_unann.label_to_behavior = {
                **expt_record_ann.label_to_behavior,
                **expt_record_unann.label_to_behavior,
            }
            expt_record_ann = expt_record_unann

            y_unann = [*y_unann, *y_ann]
            y_ann = y_unann

            X_unann = np.vstack((X_unann, X_ann))
            X_ann = X_unann

        embedding_chart_unann = plot_embedding(
            X_unann,
            y_unann,
            y_col_name,
            expt_record_unann,
            title={
                "text": f"{expt_name_unann} (unannotated)",
                "subtitle": f"Semisupervised with {expt_name_ann} (annotated)",
            },
        )
        embedding_chart_ann = plot_embedding(
            X_ann,
            y_ann,
            y_col_name,
            expt_record_ann,
            title={
                "text": f"{expt_name_ann} (annotated)",
                "subtitle": f"Semisupervised with {expt_name_unann} (unannotated)",
            },
        )

        chart_type_name = "semisupervised-pair-embedding"
        expt_name_unann += get_mask_suffix(unann_embedding_name, expt_record_unann)
        expt_name_ann += get_mask_suffix(ann_embedding_name, expt_record_ann)

        chart_name_unann = (
            f"{expt_name_unann}_{chart_type_name}_{expt_name_ann}{name_suffix}"
        )
        save(
            embedding_chart_unann,
            str(expt_path_unann / "figures" / f"{chart_name_unann}.{args.extension}"),
        )

        chart_name_ann = (
            f"{expt_name_ann}_{chart_type_name}_{expt_name_unann}{name_suffix}"
        )
        save(
            embedding_chart_ann,
            str(expt_path_ann / "figures" / f"{chart_name_ann}.{args.extension}"),
        )
        print(expt_name_ann, expt_name_unann)


def generate_supervised_disparate(project_obj):
    alt.themes.register("embedding_style", StyleEmbedding.get_embedding_style)
    alt.themes.enable("embedding_style")

    annotated_expt_names = list(project_obj.annotation_path_dict.keys())
    for expt_name in annotated_expt_names:
        expt_path = project_obj.expt_path_dict[expt_name]
        expt_record = jl.load(expt_path / "expt_record.z")
        embedding_name = "supervised_disparate_embedding"
        X = np.load(expt_path / "embeddings" / f"{embedding_name}.npy")

        X, y, y_col_name, expt_record = mask_and_get_labels(
            X, expt_path, embedding_name, expt_record
        )

        embedding_chart = plot_embedding(
            X,
            y,
            y_col_name,
            expt_record,
            title={
                "text": f"{expt_name}",
                "subtitle": "Supervised disparate",
            },
        )

        mask_suffix = get_mask_suffix(embedding_name, expt_record)
        chart_name = (
            f"{expt_name}_supervised-disparate-embedding{mask_suffix}{name_suffix}"
        )
        save(
            embedding_chart,
            str(expt_path / "figures" / f"{chart_name}.{args.extension}"),
        )
        print(expt_name)


def generate_unsupervised_disparate(project_obj):
    alt.themes.register("embedding_style", StyleEmbedding.get_embedding_style)
    alt.themes.enable("embedding_style")

    all_expt_names = list(project_obj.expt_path_dict.keys())
    for expt_name in all_expt_names:
        expt_path = project_obj.expt_path_dict[expt_name]
        expt_record = jl.load(expt_path / "expt_record.z")
        embedding_name = "unsupervised_disparate_embedding"
        X = np.load(expt_path / "embeddings" / f"{embedding_name}.npy")

        X, y, y_col_name, expt_record = mask_and_get_labels(
            X, expt_path, embedding_name, expt_record
        )

        embedding_chart = plot_embedding(
            X,
            y,
            y_col_name,
            expt_record,
            title={
                "text": f"{expt_name}",
                "subtitle": "Unsupervised disparate",
            },
        )

        mask_suffix = get_mask_suffix(embedding_name, expt_record)
        chart_name = (
            f"{expt_name}_unsupervised-disparate-embedding{mask_suffix}{name_suffix}"
        )
        save(
            embedding_chart,
            str(expt_path / "figures" / f"{chart_name}.{args.extension}"),
        )
        print(expt_name)


if __name__ == "__main__":
    project = experiment_processing.Project(
        args.main_cfg_path,
    )

    if args.generate_unsupervised_disparate:
        generate_unsupervised_disparate(project)
    if args.generate_supervised_disparate:
        generate_supervised_disparate(project)
    if args.generate_semisupervised_pair:
        generate_semisupervised_pair(project)
