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
    def strip_affix(name):
        """
        Removes the affix separated by "_" from the folder and the file.

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
    def print_possible_variables(pass_item):
        """
        Shows which quantities can be displayed in form of a histogram in the given
        pd.DataFrame or pd.DataFrame.columns.

        :param pass_item: list
        :return: list
        """
        pit = []
        if isinstance(pass_item, list):
            pit = pass_item
        if isinstance(pass_item, str):
            pit = list(pd.read_csv(pass_item, sep=";").columns)
        if not isinstance(pass_item, str) and not isinstance(pass_item, list):
            pit = list(pass_item.columns)
        
        pit = [item for item in pit if not any(item in it or it in item for it in ["index", "tag"])]
        
        return list(set([it.replace("muon_", "").replace("electron_", "").replace("_el", "").replace("_mu", "") for it in pit]))
