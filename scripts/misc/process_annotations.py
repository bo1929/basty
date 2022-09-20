import argparse
import pathlib

import pandas as pd

LABEL_UNANNOTATED = "Idle&Other"

parser = argparse.ArgumentParser(prog="PROG")
parser.add_argument(
    "--expt-name",
    required=True,
    type=str,
)
parser.add_argument(
    "--num-of-frames",
    required=True,
    type=int,
)
parser.add_argument(
    "--annotation-directory",
    required=False,
    type=str,
    default="./",
)

args = parser.parse_args()

expt_name = args.expt_name
num_of_frames = args.num_of_frames
annotation_directory = pathlib.Path(args.annotation_directory)

annotations = {}
missing_annotations = {}
for path in pathlib.Path(annotation_directory / f"{expt_name}/").glob("*.csv"):
    print(path)
    try:
        bhv = "-".join(path.stem.split("-")[-1].split("_"))
        df_tmp = pd.read_csv(path, header=None)
        df_tmp = df_tmp.rename(columns={0: "Beginning", 1: "End"})
        df_tmp = df_tmp.assign(
            Behavior=pd.Series([bhv for _ in range(df_tmp.shape[0])]).values
        )
        annotations[bhv] = df_tmp
    except pd.errors.EmptyDataError:
        df_tmp = pd.DataFrame.from_dict(
            {"Beginning": [-1], "End": [-1], "Behavior": [bhv]}
        )
        missing_annotations[bhv] = df_tmp

df_ann = pd.concat(annotations.values()).sort_values("Beginning").reset_index(drop=True)
ann_stop_dict = {"Beginning": [], "End": [], "Behavior": []}
for i in range(df_ann.shape[0]):
    ann_stop_dict["Behavior"].append(LABEL_UNANNOTATED)
    ann_stop_dict["Beginning"].append(df_ann["End"].iloc[i])
    if i == df_ann.shape[0] - 1:
        ann_stop_dict["End"].append(num_of_frames)
    else:
        ann_stop_dict["End"].append(df_ann["Beginning"].iloc[i + 1])

if df_ann["Beginning"].iloc[0] != 0:
    ann_stop_dict["Behavior"].append(LABEL_UNANNOTATED)
    ann_stop_dict["Beginning"].append(0)
    ann_stop_dict["End"].append(df_ann["Beginning"].iloc[0])

if missing_annotations:
    df_missing_ann = pd.concat(missing_annotations.values()).reset_index(drop=True)
else:
    df_missing_ann = pd.DataFrame()
df_ann = (
    pd.concat((df_missing_ann, df_ann, pd.DataFrame.from_dict(ann_stop_dict)))
    .sort_values("Beginning")
    .reset_index(drop=True)
)
df_ann.to_csv(annotation_directory / f"{expt_name}.csv")
