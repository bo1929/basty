import itertools
import logging
import os
from pathlib import Path
import subprocess
import textwrap
import time
from collections import Counter
import yaml
import basty.feature_extraction.body_pose as body_pose
import re

import numpy as np
import pandas as pd


def list_cartesian_product(*args):
    assert all([isinstance(list_arg, list) for list_arg in args])
    return itertools.product(*args)


def dict_check_and_sort(x):
    if x is not None:
        assert isinstance(x, dict)
        x = dict(sorted(x.items()))
    return x


def txt_wrap(x, lw=72):
    return textwrap.fill(x, lw)


def sort_dict(x):
    if x is not None:
        # Sorts dictionary by key.
        return dict(sorted(x.items()))
    return x


def parse_experiment_names(data_path_dict):
    """Takes in the name of the experiments and returns a dataframe containing
    sex,age and sleep deprivation info"""
    df = pd.DataFrame(data_path_dict.items(), columns=['ExptNames', 'Path'])
    df['SD'] = df['ExptNames'].str.contains('_SD')
    df['Age'] = df['ExptNames'].str.extract(re.compile("(_[1-9]|10)[dD]"))
    df['Age'] = pd.to_numeric(df['Age'].str[1])
    df['Sex'] = df['ExptNames'].str.extract(r"_([FM])")
    # Assign 5 to 'NaN' age (if age is not listed, then it is 5 days old per expt protocols)
    df['Age'] = df['Age'].fillna(5)

    def check_folders(folderpath):
        txt_file = [file for file in os.listdir(folderpath) if 'CB' in file and file.endswith('.txt')]
        if txt_file:
            if 'annotated' not in txt_file[0]:
                experimenter_name = os.path.splitext(txt_file[0])[0]
            else:
                experimenter_name = 'MK'
        else:
            experimenter_name = 'MK'
        return experimenter_name

    # Finaly check if each folder contains CB.txt files which indicates the experimenter. Assume MK if it is not existent
    df['Experimenter'] = df['Path'].apply(lambda x: check_folders(x))
    return df


def split_video_by_annotation(idxpath, vidname):
    """
    Takes in a file containing list of indices and a video file, returns
    smaller videos corresponding to the list.
    """
    df = pd.read_csv(idxpath)
    bout = 0
    fps = 30
    targetfolder = os.path.join(os.path.dirname(idxpath), "SplitVids")
    os.mkdir(targetfolder)

    for start, finish, behavior in zip(df.Start, df.Finish, df.Behavior):
        idx1 = start / fps
        idx2 = finish / fps
        tempname = behavior + "_" + str(bout) + "_" + str(start) + "_" + str(finish)
        outputname = os.path.join(targetfolder, tempname + ".avi")
        bout += 1
        command = (
            f"ffmpeg -n -i {vidname} -ss {idx1} -to {idx2} " f"-c:v copy {outputname}"
        )
        subprocess.call(command, shell=True)


def organize_file_structure(path):
    """
    Modifies the filenames of .csv files to match it with videos and returns a dictionary to be incorporated in the main_cfg.yaml
    """
    new_csv_names = []
    folder_dict = {}

    # Iterate over folders
    for dirs in os.listdir(path):
        avi_names = ([f for f in os.listdir(os.path.join(path, dirs)) if f.endswith('.avi') or f.endswith('.mp4')])
        # Get the file in each folder that ends with .csv
        csv_names = ([f for f in os.listdir(os.path.join(path, dirs)) if f.endswith('.csv')])

        if len(avi_names) == 1 and len(csv_names) == 1:
            new_csv_name = os.path.splitext(avi_names[0])[0] + '.csv'
            new_csv_names.extend(new_csv_name)
            os.rename(os.path.join(path, dirs, csv_names[0]), os.path.join(path, dirs, new_csv_name))
            folder_dict[os.path.splitext(avi_names[0])[0]] = os.path.join(path, dirs)
    return folder_dict


def update_main_cfg(main_path, new_dict):
    with open(main_path, 'r') as f:
        config = yaml.safe_load(f)

    config['experiment_data_paths'].update(new_dict)

    with open(main_path, 'w') as f:
        yaml.safe_dump(config, f)


def update_expt_info_df(df, yaml_path):
    """Updates the df containing experimental info with a supplemental configuration file that has the sex info
    for flies without it """
    with open(yaml_path, 'r') as f:
        info = yaml.safe_load(f)
    update_dict = info['sexlist']
    for key in list(update_dict.keys()):
        df['Sex'][df['ExptNames'] == key] = update_dict[key]
    return df


def get_likelihood(data_path_dict):
    """Loops through the data and loads likelihood scores for the desired body parts"""
    out_df_list = []
    for keys, values in data_path_dict.items():
        file_path = list(Path(values).glob(keys + '.csv'))[0]
        ind = 'likelihood'
        #save likelihood
        llh_path = os.path.join(values,keys + '_llh.pkl')
        if not os.path.exists(llh_path):
            # grab the header to create a new header for the df
            header = pd.read_csv(file_path, nrows=3)
            col_list = header.columns[header.loc[1] == ind]
            head_list = header[col_list].loc[0].tolist()
            df = pd.read_csv(file_path, skiprows=[0, 1, 2], header=None)
            df = df.iloc[:, 3::3]
            df.columns = head_list
            df.to_pickle(llh_path)
            df['ExptNames'] = keys
        else:
            df = pd.read_pickle(llh_path)
            df['ExptNames'] = keys
        out_df_list.append(df)

    out_df = pd.concat(out_df_list)
    return out_df


def convert_hour_to_HM(hour):
    return time.strftime("%H:%M", time.gmtime(float(hour) * 3600))


def change_points(labels):
    labels_diff = np.diff(labels)
    return (np.nonzero(labels_diff != 0)[0] + 1).astype(int)


def cont_intvls(labels):
    return np.append(
        np.insert(change_points(labels), 0, 0),
        labels.shape[0],
    ).astype(int)


def generate_bout_report(arr, label_to_name={}, filter_names=[]):
    bout_intervals = cont_intvls(arr)

    bout_start_list = []
    bout_finish_list = []
    name_list = []

    for i in range(bout_intervals.shape[0] - 1):
        start, finish = bout_intervals[i], bout_intervals[i + 1] - 1
        name = label_to_name[arr[bout_intervals[i]]]

        if not filter_names or name in filter_names:
            bout_start_list.append(start)
            bout_finish_list.append(finish)
            name_list.append(name)

    report_df = pd.DataFrame(
        {
            "Start": bout_start_list,
            "Finish": bout_finish_list,
            "Name": name_list,
        }
    )
    return report_df


def reverse_dict(x):
    if x is not None:
        reversed_dict = {j: i for i, j in x.items()}
    return reversed_dict


def inner_dict_sum(x):
    return {key: sum(val.values()) for key, val in x.items()}


def sliding_window(seq, winsize=3, stepsize=1):
    assert winsize > 1
    assert stepsize >= 1
    if winsize % 2 == 0:
        winsize = winsize + 1
    hw = (winsize - 1) // 2
    p1 = [seq[0: i + hw] for i in range(0, hw, stepsize)]
    p2 = [seq[i - hw: i + hw] for i in range(hw, len(seq), stepsize)]
    return p1 + p2


def most_common(lst):
    if not lst:
        return -1
    data = Counter(lst)
    return data.most_common(1)[0][0]


def flatten(items, seqtypes=(list, tuple)):
    for i, _ in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i: i + 1] = items[i]
    return items


def multi_append(*args):
    for list_, item in args:
        list_.append(item)


def merge_two_dicts(x, y):
    z = x.copy()  # Start with x's keys and values.
    z.update(y)  # Modifies z with y's keys and values & returns None.
    return z


# Return the first nested list which contains the item.
# If there is no such list, return [].
def in_nested_list(my_list, item):
    is_in = []
    if item in my_list:
        is_in = my_list
    else:
        for sublist in my_list:
            if isinstance(sublist, list):
                is_in = in_nested_list(sublist, item)
            if len(is_in) > 0:
                break
    return is_in


def is_all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def make_funcname_dict(*args):
    dict_ = {}
    for item in args:
        dict_.update({item.__name__: item})
    return dict_


def timeit(f):
    """
    Timing decorator for functions. Just add @timeit to start and function
    will be timed. It will write starting and ending times

    Parameters
    ----------
    f : function
        decorator takes the function as parameter
    Returns
    -------
    mixed
        return value of function itself
    Raises
    ------
    Error
        when any error thrown by called function does not catch it
    """

    def wrapper(*args, **kwargs):
        logger = logging.getLogger("main")
        logging.info("...")
        logging.basicConfig()
        logger.info(f"Started: {f.__qualname__}.")
        t = time.time()
        res = f(*args, **kwargs)
        logger.info(f"Finished: {f.__qualname__} elapsed: {time.time() - t:.2f}s.")
        logging.info("...")
        return res

    return wrapper
