from pathlib import Path

import basty.utils.io as io


def backup_old_project(main_cfg_path):
    project_path = io.read_config(main_cfg_path).get("project_path")
    old_proj_dir = Path(project_path)
    if old_proj_dir.exists():
        suffix = 1
        while old_proj_dir.with_suffix(f".{str(suffix)}").exists():
            suffix += 1
        old_proj_dir.replace(old_proj_dir.with_suffix(f".{str(suffix)}"))


def log_params(main_cfg_path, name_params, params):
    project_path = io.read_config(main_cfg_path).get("project_path")
    params_path = Path(project_path) / "parameters" / f"{name_params}.yaml"
    io.ensure_file_dir(params_path)
    io.dump_yaml(params, params_path)
