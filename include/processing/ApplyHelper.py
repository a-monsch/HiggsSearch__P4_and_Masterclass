# -*- coding: UTF-8 -*-
import os
from collections import namedtuple

import numpy as np
import pandas as pd


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray


class ProcessHelper(object):
    """
    Class with useful functions for the Apply class.
    """
    
    @staticmethod
    def create_tuple(file_path):
        """
        Creates a list of namedtuple consisting of the respective names and the corresponding leptons
        ("muon", "lelectron" or "both").

        :param file_path: str
        :return: list
                 [Pair("name", "particle"), (...), ...]
        """
        data_type = [("ru", "FourElectron", "FourMuon"), ("mc", "_4el_", "_4mu_")]
        filename = [file_path + item for item in os.listdir(file_path)]
        filetuple = []
        used_names = data_type[0]
        pair = namedtuple("Pair", "name, particle")
        if "mc_" in file_path:
            used_names = data_type[1]
        for item in filename:
            if used_names[1] in item:
                filetuple.append(pair(item, "electron"))
                continue
            if used_names[2] in item:
                filetuple.append(pair(item, "muon"))
                continue
            else:
                filetuple.append(pair(item, "both"))
        return filetuple
    
    @staticmethod
    def strip_affix(name):
        """
        Removes the affix separated by "_" from the folder and the filename.

        :param name: str
        :return: str
        """
        name, name_format = name.split(".")[0], name.split(".")[-1]
        name = "_".join(name.split("_")[:-1])
        return f"{name}.{name_format}"
    
    @staticmethod
    def change_on_affix(name, affix):
        """
        Changes the affix of the folder and the name of a given file (separated by "_").

        :param name: str
        :param affix: str
        :return: str
        """
        to_place_path, name_ = os.path.split(name)
        name_ = ProcessHelper.strip_affix(name_)
        name_ = f"{name_.split('.')[0]}_{affix}.{name_.split('.')[-1]}"
        to_place_path = f"{'_'.join(to_place_path.split('_')[:-1])}_{affix}"
        return os.path.join(to_place_path, name_).replace("\\", "/")
    
    @staticmethod
    def print_status_bar(i, tot_, flush=True, end="\r"):
        """
        Status bar printed for loops during the loops.

        Note: Best to replace it with tqdm and remove it at the end.

        :param i: int
        :param tot_: list or int
        :param flush: bool
        :param end: str
        """
        if type(tot_).__name__ == "int":
            print(f"[{'#' * (i + 1): <{tot_}}] {round((i + 1) / tot_) * 100}%", end=end, flush=flush)
        else:
            print(f"[{'#' * (i + 1): <{len(tot_)}}] {round((i + 1) / len(tot_) * 100)}%", end=end, flush=flush)
    
    @staticmethod
    def print_possible_variables(pass_item):
        """
        Shows which sizes can be histographed in the given pd.DataFrame or pd.DataFrame.columns.

        :param pass_item: list
        :return: list
        """
        pit = []
        if isinstance(pass_item, list): pit = pass_item
        if isinstance(pass_item, str): pit = list(pd.read_csv(pass_item, sep=";").columns)
        if not isinstance(pass_item, str) and not isinstance(pass_item, list): pit = list(pass_item.columns)
        return list(set([it.replace("muon_", "").replace("electron_", "").replace("_el", "").replace("_mu", "") for it in pit]))
