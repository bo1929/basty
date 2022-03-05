import time
import itertools
import textwrap
import logging
import numpy as np

from collections import Counter


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
    p1 = [seq[0 : i + hw] for i in range(0, hw, stepsize)]
    p2 = [seq[i - hw : i + hw] for i in range(hw, len(seq), stepsize)]
    return p1 + p2


def most_common(lst):
    if not lst:
        return -1
    data = Counter(lst)
    return data.most_common(1)[0][0]


def flatten(items, seqtypes=(list, tuple)):
    for i, _ in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i : i + 1] = items[i]
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
