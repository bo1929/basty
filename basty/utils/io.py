import logging
import pickle
from pathlib import Path

import yaml

import basty.utils.misc as misc


def safe_create_dir(d: Path):
    if not d.exists():
        logger = logging.getLogger("main")
        logger.info(f"Directory is not found, creating: {str(d)}")
        d.mkdir(parents=True)


def ensure_file_dir(file_path: Path):
    safe_create_dir(file_path.parent)


def read_config(file_path):
    file_path = Path(file_path)
    file_extension = file_path.suffix
    if file_extension in [".yaml", ".yml"]:
        return read_yaml(file_path)
    else:
        raise ValueError("Unkown configuration file type!")


def read_pickle(file_path, **kwargs):
    with open(str(file_path), "rb") as file:
        return pickle.read(file)


def write_pickle(data, file_path, **kwargs):
    with open(str(file_path), "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def write_txt(data, file_path, **kwargs):
    with open(str(file_path), "w", encoding="utf-8") as file:
        file.write(data)


def read_txt(file_path, **kwargs):
    with open(str(file_path), "r", encoding="utf-8") as file:
        return file.read()


def read_yaml(path, **kwargs):
    with open(str(path), "r") as file:
        return yaml.load(file, Loader=yaml.SafeLoader)


def dump_yaml(data, file_path, **kwargs):
    with open(str(file_path), "w") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def multi_read(destination_dir, names, function):
    read_func = read_reference_dict[function]
    data_all = []
    destination_dir = Path(destination_dir)
    for i in range(names):
        file_path = destination_dir / names[i]
        data_all.append(read_func(file_path))
    return data_all


def multi_write(destination_dir, names, data_all, function):
    if len(names) != len(data_all):
        raise ValueError("Lengt of given name list and data list are not same!")
    destination_dir = Path(destination_dir)
    write_func = write_reference_dict[function]
    for i in range(names):
        file_path = destination_dir / names[i]
        write_func(file_path, data_all[i])


read_reference_dict = misc.make_funcname_dict(
    read_pickle,
    read_txt,
    read_yaml,
)
write_reference_dict = misc.make_funcname_dict(
    write_pickle,
    write_txt,
    dump_yaml,
)
