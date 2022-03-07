import argparse

from pathlib import Path

from basty.utils.io import read_config
from basty.project.experiment_processing import Project

parser = argparse.ArgumentParser(
    description="Initialize project  based on a given main configuration."
)
parser.add_argument(
    "--main-cfg-path",
    type=str,
    required=True,
    help="Path to the main configuration file.",
)
parser.add_argument(
    "--reset-project",
    default=True,
    type=bool,
    help="Option to create a new project.",
)

args = parser.parse_args()
main_cfg_path = args.main_cfg_path
reset_project = args.reset_project


def backup_old_project(main_cfg_path):
    main_cfg = read_config(main_cfg_path)
    project_path = main_cfg.get("project_path")
    old_proj_dir = Path(project_path)
    if old_proj_dir.exists():
        suffix = ".old"
        while old_proj_dir.with_suffix(suffix).exists():
            suffix = suffix + ".old"
        old_proj_dir.replace(old_proj_dir.with_suffix(suffix))


def test_project_init(main_cfg_path, **kwargs):
    assert proj is not None

    assert Path(proj.main_cfg["configuration_paths"]["pose_cfg"]).exists()
    assert Path(proj.main_cfg["configuration_paths"]["feature_cfg"]).exists()
    assert Path(proj.main_cfg["configuration_paths"]["temporal_cfg"]).exists()
    assert Path(proj.project_path / "main_cfg.yaml").exists()

    for expt_name, expt_path in proj.expt_path_dict.items():
        assert (expt_path / "expt_record.z").exists()
        assert (expt_path / "embeddings").exists()
        expt_record = proj._load_joblib_object(expt_path, "expt_record.z")
        if expt_record.has_annotation:
            assert expt_name in list(proj.annotation_path_dict.keys())
            assert (expt_path / "annotations.npy").exists()
            assert expt_record.mask_annotated is not None


if __name__ == "__main__":
    """
    Add suffix '.old' to the project created by previous tests.
    """
    if reset_project:
        backup_old_project(main_cfg_path)
    """
    Only test if the intended files and directories created.
    Uses default arguments.
    Test starts.
    """

    proj = Project(
        main_cfg_path,
        annotation_priority=[],
        inactive_annotation="Idle",
        noise_annotation="Noise",
        arouse_annotation="Moving",
    )
