import argparse
import pathlib
import pandas as pd

LABEL_INACTIVE = "Idle"

parser = argparse.ArgumentParser(prog="PROG")
parser.add_argument(
    "--expt-name",
    required=True,
    type=str,
)
parser.add_argument(
    "--expt-length",
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
total_frame = args.expt_length
annotation_directory = pathlib.Path(args.annotation_directory)

annotations = {}
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
        annotations[bhv] = pd.DataFrame()

df_ann = pd.concat(annotations.values()).sort_values("Beginning").reset_index(drop=True)
ann_stop_dict = {"Beginning": [], "End": [], "Behavior": []}
for i in range(df_ann.shape[0]):
    ann_stop_dict["Behavior"].append(LABEL_INACTIVE)
    ann_stop_dict["Beginning"].append(df_ann["End"].iloc[i])
    if i == df_ann.shape[0] - 1:
        ann_stop_dict["End"].append(total_frame)
    else:
        ann_stop_dict["End"].append(df_ann["Beginning"].iloc[i + 1])

if df_ann["Beginning"].iloc[0] != 0:
    ann_stop_dict["Behavior"].append(LABEL_INACTIVE)
    ann_stop_dict["Beginning"].append(0)
    ann_stop_dict["End"].append(df_ann["Beginning"].iloc[0])

df_ann = (
    pd.concat((df_ann, pd.DataFrame.from_dict(ann_stop_dict)))
    .sort_values("Beginning")
    .reset_index(drop=True)
)
df_ann.to_csv(annotation_directory / f"{expt_name}.csv")
