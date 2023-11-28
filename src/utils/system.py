import os
from pathlib import Path
from typing import Union


RC_DICT = {
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
}


def insert_rc_args(rc_args: dict):
    rc_dict = RC_DICT.copy()
    if rc_args is not None:
        for key in rc_args:
            if key in rc_dict.keys():
                rc_dict[key] = rc_args[key]
            else:
                print(key + " is not a valid argument")
    return rc_dict


def find_max_version(path: Union[Path, str]):
    path = Path(path)

    version_dirs = list(path.glob("version_*"))

    if len(version_dirs) == 0:
        return -1

    version_dirs = [entry for entry in version_dirs if entry.is_dir()]

    versions = [int(str(dir_name).split("_")[-1]) for dir_name in version_dirs]
    max_version = max(versions)
    return max_version
