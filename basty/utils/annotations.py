import pathlib
import numpy as np
import pandas as pd

import basty.utils.misc as misc


class HumanAnnotations:
    def __init__(self, ann_path=None, inactive_behavior="Idle"):
        # Label of the inactive behavior is always zero.
        self.inactive_behavior = inactive_behavior
        self.behavior_to_label = {}
        self.multibehavior_to_label = {}

        self.behavior_to_label[self.inactive_behavior] = 0
        self.multibehavior_to_label[self.inactive_behavior] = 0

        if ann_path is not None:
            self.df_ann = pd.read_csv(ann_path)
            behaviors_uniq = [
                behavior
                for behavior in self.df_ann["Behavior"].unique()
                if behavior != self.inactive_behavior
            ]
            for i, bhv in enumerate(sorted(behaviors_uniq)):
                self.behavior_to_label[bhv] = i + 1
                self.multibehavior_to_label[bhv] = i + 1
            self.df_ann = self.df_ann.query("Beginning != End")
        else:
            self.df_ann = pd.DataFrame()
            behaviors_uniq = []

        self.label_to_behavior = misc.reverse_dict(self.behavior_to_label)
        self.label_to_multibehavior = misc.reverse_dict(self.multibehavior_to_label)

    def annotation_details(self, y_ann):
        assert isinstance(y_ann, np.ndarray)
        ann_uniqs, ann_counts = np.unique(y_ann, return_counts=True)
        detal_dict = {}
        for idx, lbl_ann in enumerate(ann_uniqs):
            try:
                detal_dict[self.label_to_behavior[lbl_ann]] = ann_counts[idx]
            except KeyError:
                try:
                    detal_dict[self.label_to_multibehavior[lbl_ann]] = ann_counts[idx]
                except KeyError:
                    raise KeyError("Given annotation label is unkown.")
        return detal_dict

    def compile_annotations(self, ann_dir, ann_out_path, num_of_frames):
        annotations = {}
        missing_annotations = {}
        for path in pathlib.Path(ann_dir).glob("*.csv"):
            bhv = "-".join(path.stem.split("-")[-1].split("_"))
            try:
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

        df_missing_ann = pd.concat(missing_annotations.values()).reset_index(drop=True)
        df_ann = (
            pd.concat(annotations.values())
            .sort_values("Beginning")
            .reset_index(drop=True)
        )
        ann_stop_dict = {"Beginning": [], "End": [], "Behavior": []}
        for i in range(df_ann.shape[0]):
            ann_stop_dict["Behavior"].append(self.inactive_behavior)
            ann_stop_dict["Beginning"].append(df_ann["End"].iloc[i] + 1)
            if i == df_ann.shape[0] - 1:
                ann_stop_dict["End"].append(num_of_frames)
            else:
                ann_stop_dict["End"].append(df_ann["Beginning"].iloc[i + 1] - 1)

        if df_ann["Beginning"].iloc[0] != 0:
            ann_stop_dict["Behavior"].append(self.inactive_behavior)
            ann_stop_dict["Beginning"].append(0)
            ann_stop_dict["End"].append(df_ann["Beginning"].iloc[0] - 1)

        df_ann = (
            pd.concat((df_missing_ann, df_ann, pd.DataFrame.from_dict(ann_stop_dict)))
            .sort_values("Beginning")
            .reset_index(drop=True)
        )
        df_ann.to_csv(ann_out_path, index=False)

    def get_annotations(self):
        y_ann_list = [[] for _ in range(self.df_ann["End"].max())]
        for i in range(self.df_ann.shape[0]):
            ann_name = self.df_ann.iloc[i]["Behavior"]
            begining = self.df_ann.iloc[i]["Beginning"]
            end = self.df_ann.iloc[i]["End"]
            for i in range(begining, end):
                y_ann_list[i].append(self.behavior_to_label[ann_name])
        return y_ann_list

    def label_converter(self, y_ann_list, priority_order=[]):
        def get_label(y_ann_i):
            assert len(y_ann_i) > 0
            lbl_ann = y_ann_i[-1]
            for ann_name in priority_order:
                lbl = self.behavior_to_label[ann_name]
                if lbl in y_ann_i:
                    lbl_ann = lbl
                    break
            return lbl_ann

        y_ann = np.array(
            [get_label(y_ann_list[i]) for i in range(len(y_ann_list))], dtype=int
        )
        return y_ann

    def multilabel_converter(self, y_ann_list, sep="&"):
        def get_label(y_ann_i):
            assert len(y_ann_i) > 0
            if len(y_ann_i) > 1:
                name_ann = sep.join(
                    [
                        self.label_to_multibehavior[y_ann_i[j]]
                        for j in range(len(y_ann_i))
                        if y_ann_i[j] != self.behavior_to_label[self.inactive_behavior]
                    ]
                )
                lbl_ann = self.multibehavior_to_label.get(name_ann, False)
                if lbl_ann:
                    pass
                else:
                    lbl_ann = max(self.multibehavior_to_label.values()) + 1
                    self.multibehavior_to_label[name_ann] = lbl_ann
                    self.label_to_multibehavior[lbl_ann] = name_ann
            else:
                lbl_ann = y_ann_i[0]
            return lbl_ann

        y_ann = np.array(
            [get_label(y_ann_list[i]) for i in range(len(y_ann_list))], dtype=int
        )
        return y_ann
