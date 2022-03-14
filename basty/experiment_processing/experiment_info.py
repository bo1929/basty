import numpy as np
import pandas as pd

from collections import defaultdict

import basty.utils.misc as misc


class AnnotationInfo:
    def __init__(self):
        # no annotation by default
        self.has_annotation = False
        self._inactive_annotation = None
        self._noise_annotation = None
        self._arouse_annotation = None
        self._label_to_behavior = None
        self._behavior_to_label = None
        self._mask_annotated = None

    @property
    def noise_annotation(self):
        if self._noise_annotation is None:
            raise ValueError("'noise_annotation' is not set.")
        return self._noise_annotation

    @noise_annotation.setter
    def noise_annotation(self, value):
        if not isinstance(value, str):
            raise ValueError("'noise_annotation' must have type 'str'.")
        self._noise_annotation = value

    @property
    def inactive_annotation(self):
        if self._inactive_annotation is None:
            raise ValueError("'inactive_annotation' is not set.")
        return self._inactive_annotation

    @inactive_annotation.setter
    def inactive_annotation(self, value):
        if not isinstance(value, str):
            raise ValueError("'inactive_annotation' must have type 'str'.")
        self._inactive_annotation = value

    @property
    def arouse_annotation(self):
        if self._arouse_annotation is None:
            raise ValueError("'arouse_annotation' is not set.")
        return self._arouse_annotation

    @arouse_annotation.setter
    def arouse_annotation(self, value):
        if not isinstance(value, str):
            raise ValueError("'arouse_annotation' must have type 'str'.")
        self._arouse_annotation = value

    @property
    def mask_annotated(self):
        if self._mask_annotated is None:
            raise ValueError("'mask_annotated' is not set.")
        return self._mask_annotated

    @mask_annotated.setter
    def mask_annotated(self, value):
        if not isinstance(value, np.ndarray) and value.ndim == 1:
            raise ValueError("'mask_annotated' must have type '1D numpy.ndarray'.")
        self._mask_annotated = value

    @property
    def label_to_behavior(self):
        if self._label_to_behavior is None:
            raise ValueError("'label_to_behavior' is not set.")
        return self._label_to_behavior

    @property
    def behavior_to_label(self):
        if self._behavior_to_label is None:
            raise ValueError("'behavior_to_label' is not set.")
        return self._behavior_to_label

    @label_to_behavior.setter
    def label_to_behavior(self, value):
        if not isinstance(value, dict):
            raise ValueError("'label_to_behavior' must have type 'dict'.")
        if self._behavior_to_label is not None:
            assert misc.reverse_dict(self._behavior_to_label) == value
        self._label_to_behavior = misc.sort_dict(value)

    @behavior_to_label.setter
    def behavior_to_label(self, value):
        if not isinstance(value, dict):
            raise ValueError("'behavior_to_label' must have type 'dict'.")
        if self._label_to_behavior is not None:
            assert misc.reverse_dict(self._label_to_behavior) == value
        self._behavior_to_label = misc.sort_dict(value)


class ExptInfo:
    def __init__(self, fps):
        self.fps = fps
        self.expt_frame_count = None
        self._part_to_frame_count = None
        self._mask_dormant = None
        self._mask_active = None
        self._mask_noise = None
        self._ftname_to_snapft = None
        self._ftname_to_deltaft = None
        self._snapft_to_ftname = None
        self._deltaft_to_ftname = None

    @property
    def part_to_frame_count(self):
        if self._part_to_frame_count is None:
            raise ValueError("'part_to_frame_count' is not set.")
        return self._part_to_frame_count

    @part_to_frame_count.setter
    def part_to_frame_count(self, value):
        if not isinstance(value, dict):
            raise ValueError("'part_to_frame_count' must have type 'dict'.")
        self.expt_frame_count = sum(value.values())
        self._part_to_frame_count = value

    @property
    def mask_noise(self):
        if self._mask_noise is None:
            raise ValueError("'mask_noise' is not set.")
        return self._mask_noise

    @mask_noise.setter
    def mask_noise(self, value):
        if not isinstance(value, np.ndarray) and value.ndim == 1:
            raise ValueError("'mask_noise' must have type '1D numpy.ndarray'.")
        self._mask_noise = value

    @property
    def mask_dormant(self):
        if self._mask_dormant is None:
            raise ValueError("'mask_dormant' is not set.")
        if self._mask_noise is not None:
            mask_dormant = np.logical_and(
                self._mask_dormant, np.logical_not(self._mask_noise)
            )
        else:
            mask_dormant = self._mask_dormant
        return mask_dormant

    @mask_dormant.setter
    def mask_dormant(self, value):
        if not isinstance(value, np.ndarray) and value.ndim == 1:
            raise ValueError("'mask_dormant' must have type '1D numpy.ndarray'.")
        self._mask_dormant = value

    @property
    def mask_active(self):
        if self._mask_active is None:
            raise ValueError("'mask_active' is not set.")
        return self._mask_active

    @mask_active.setter
    def mask_active(self, value):
        if not isinstance(value, np.ndarray) and value.ndim == 1:
            raise ValueError("'mask_active' must have type '1D numpy.ndarray'.")
        self._mask_active = value

    @property
    def ftname_to_snapft(self):
        if self._ftname_to_snapft is None:
            raise ValueError("'ftname_to_snapft' is not set.")
        return self._ftname_to_snapft

    @property
    def snapft_to_ftname(self):
        if self._snapft_to_ftname is None:
            raise ValueError("'behavior_to_label' is not set.")
        return self._snapft_to_ftname

    @property
    def ftname_to_deltaft(self):
        if self._ftname_to_deltaft is None:
            raise ValueError("'ftname_to_deltaft' is not set.")
        return self._ftname_to_deltaft

    @property
    def deltaft_to_ftname(self):
        if self._deltaft_to_ftname is None:
            raise ValueError("'behavior_to_label' is not set.")
        return self._deltaft_to_ftname

    @snapft_to_ftname.setter
    def snapft_to_ftname(self, value):
        if not isinstance(value, dict):
            raise ValueError("'snapft_to_ftname' must have type 'dict'.")
        if self._ftname_to_snapft is not None:
            assert misc.reverse_dict(self._ftname_to_snapft) == value
        else:
            self.ftname_to_snapft = misc.reverse_dict(value)
        self._snapft_to_ftname = misc.sort_dict(value)

    @ftname_to_snapft.setter
    def ftname_to_snapft(self, value):
        if not isinstance(value, dict):
            raise ValueError("'ftname_to_snapft' must have type 'dict'.")
        if self._snapft_to_ftname is not None:
            assert misc.reverse_dict(self._snapft_to_ftname) == value
        else:
            self._snapft_to_ftname = misc.reverse_dict(value)
        self._ftname_to_snapft = misc.sort_dict(value)

    @deltaft_to_ftname.setter
    def deltaft_to_ftname(self, value):
        if not isinstance(value, dict):
            raise ValueError("'deltaft_to_ftname' must have type 'dict'.")
        if self._ftname_to_deltaft is not None:
            assert misc.reverse_dict(self._ftname_to_deltaft) == value
        else:
            self.ftname_to_deltaft = misc.reverse_dict(value)
        self._deltaft_to_ftname = misc.sort_dict(value)

    @ftname_to_deltaft.setter
    def ftname_to_deltaft(self, value):
        if not isinstance(value, dict):
            raise ValueError("'ftname_to_deltaft' must have type 'dict'.")
        if self._deltaft_to_ftname is not None:
            assert misc.reverse_dict(self._deltaft_to_ftname) == value
        else:
            self._deltaft_to_ftname = misc.reverse_dict(value)
        self._ftname_to_deltaft = misc.sort_dict(value)

    def _get_sec(self, idx):
        sec_total = idx // self.fps
        second = sec_total % 60
        return sec_total, second

    def get_hour_stamp(self, idx):
        if idx is np.nan:
            stamp = np.nan
        else:
            sec_total, second = self._get_sec(idx)
            minute = (sec_total // 60) % 60
            hour = sec_total // 3600
            stamp = str(int(hour)) + ":" + str(int(minute)) + ":" + str(int(second))
        return stamp

    def get_minute_stamp(self, idx):
        if idx is np.nan:
            stamp = np.nan
        else:
            sec_total, second = self._get_sec(idx)
            minute = sec_total // 60
            stamp = str(int(minute)) + ":" + str(int(second))
        return stamp


class ExptRecord(ExptInfo, AnnotationInfo):
    def __init__(self, name, data_path, expt_path, fps=30):
        self.name = name
        self.data_path = data_path
        self.expt_path = expt_path
        ExptInfo.__init__(self, fps)
        AnnotationInfo.__init__(self)

    def generate_report(self, labels, is_behavior=False, use_time_stamps=False):
        assert isinstance(labels, np.ndarray)
        assert np.issubdtype(labels.dtype, np.integer)

        intvls = misc.cont_intvls(labels)

        if is_behavior:
            try:
                labels = [self.label_to_behavior[lbl] for lbl in labels]
            except KeyError:
                raise ValueError(
                    "Given labels are not defined for label to behavior mapping."
                )

        report_dict = defaultdict(list)
        for i in range(intvls.shape[0] - 1):
            dur = intvls[i + 1] - intvls[i]
            if use_time_stamps:
                report_dict["Duration"].append(self.get_minute_stamp(dur))
                report_dict["Beginning"].append(self.get_hour_stamp(intvls[i]))
                report_dict["End"].append(self.get_hour_stamp(intvls[i + 1]))
            else:
                report_dict["Duration"].append(dur)
                report_dict["Beginning"].append(intvls[i])
                report_dict["End"].append(intvls[i + 1])
            report_dict["Label"].append(labels[intvls[i]])
        df_report = pd.DataFrame.from_dict(report_dict)

        return df_report
