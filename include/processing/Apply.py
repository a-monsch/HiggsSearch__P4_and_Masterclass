# -*- coding: UTF-8 -*-
import errno
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from .CalcAndAllowerInit import CalcInit, AllowedInit
from .FilterRecoAdd import FilterStr, Reconstruct, AddVariable
from ..histogramm.Hist import Hist


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray



class Apply(object):
    """
    Class for applying filters or reconstruction steps, as well as
    adding different sizes or visualizing current distributions.
    """
    calc_instance = CalcInit
    allowed_instance = AllowedInit
    reconstruction_instance = Reconstruct
    filter_instance = FilterStr
    add_variable_instance = AddVariable
    
    def __init__(self, input_, particle_type, use_n_rows=None,
                 filter_instance=FilterStr,
                 add_variable_instance=AddVariable,
                 reconstruction_instance=Reconstruct,
                 calc_instance=CalcInit,
                 allowed_instance=AllowedInit,
                 multi_cpu=True,
                 n_cpu=mp.cpu_count()):
        
        self.particle_type = particle_type
        self.n_cpu = n_cpu
        
        if type(input_) is str:
            self.data, self.name = Apply.from_string(string_name=input_, n_rows=use_n_rows)
        
        else:
            self.data = input_
        
        self.multi_cpu = multi_cpu
        
        self.calculated_dump = {}
        Apply.set_instance(calc_instance=calc_instance, allowed_instance=allowed_instance,
                           filter_instance=filter_instance, reconstruct_instance=reconstruction_instance,
                           add_variable_instance=add_variable_instance)
        
        Apply.filter_instance.set_instance(calc_instance=calc_instance, allowed_instance=allowed_instance)
        Apply.reconstruction_instance.set_instance(calc_instance=calc_instance, allowed_instance=allowed_instance)
        Apply.add_variable_instance.set_instance(calc_instance=calc_instance, allowed_instance=allowed_instance)
    
    @staticmethod
    def from_string(string_name, n_rows=None):
        """
        Load the pandas.DataFrame from a file.

        :param string_name: str
        :param n_rows: int
        :return: pd.DataFrame, str
        """
        
        if not os.path.exists(string_name):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), string_name)
        
        print("Load " + string_name)
        data_frame = pd.read_csv(string_name, sep=";", nrows=n_rows)
        print(f"{string_name} Successfully loaded")
        return data_frame, string_name
    
    @classmethod
    def set_instance(cls, **kwargs):
        """
        Sets the class instances used in the following.

        :param kwargs: class instances
        """
        if any("calc" in it for it in kwargs.keys()): Apply.calc_instance = kwargs["calc_instance"]
        if any("allowed" in it for it in kwargs.keys()): Apply.allowed_instance = kwargs["allowed_instance"]
        if any("run" in it for it in kwargs.keys()): Apply.filter_instance = kwargs["filter_instance"]
        if any("run" in it for it in kwargs.keys()):
            Apply.reconstruction_instance = kwargs["reconstruct_instance"]
        if any("run" in it for it in kwargs.keys()):
            Apply.add_variable_instance = kwargs["add_variable_instance"]
    
    def _verbose(self, msg_, cls_, fnc_, shape_, flush_=False):
        """
        Function to print the performed operation.
        
        :param msg_: str
        :param cls_: python class
        :param fnc_: str
        :param shape_: tuple
        :param flush_: bool
        """
        if flush_:
            print(f"{msg_: <{7}} {cls_.__name__: <{12}}: {fnc_: <{35}}; shape: {str(shape_): <{15}}", end="\r", flush=True)
        if not flush_:
            print(f"{msg_: <{7}} {cls_.__name__: <{12}}: {fnc_: <{35}}; shape: {str(shape_): <{15}}")
    
    def get_partial(self, arg_tuple, **kwargs):
        """
        Partially constructs and evaluates the pandas run function using the given arguments.

        :param arg_tuple: tuple
                          (class_name, used_name, data_frame verbose)
                          used_name is the used method name of the provided class
        :param kwargs: see tuple, except without verbose
        :return: pd.DataFrame
        """
        cls_, fnc_, df, vb_ = arg_tuple if arg_tuple else (kwargs["used_class"], kwargs["used_name"], kwargs["data_frame"], True)
        
        if vb_: self._verbose("Done", cls_, fnc_, df.shape, True)
        
        if not df.empty:
            if self.multi_cpu:
                df = df.apply(lambda x: getattr(cls_(x, list(df)), fnc_)(look_for=self.particle_type), axis=1)
            if not self.multi_cpu:
                tqdm.pandas()
                df = df.progress_apply(lambda x: getattr(cls_(x, list(df)), fnc_)(look_for=self.particle_type), axis=1)
            df = df.dropna()
            if vb_: self._verbose("Done", cls_, fnc_, df.shape)
            return df
        if df.empty:
            return df
    
    def __multiprocessing(self, **kwargs):
        """
        Executes the Apply.get_partial in a pool process with self.n_cpu number of processes.

        :param kwargs: see Apply.get_partial
        :return: pd.DataFrame
        """
        if not kwargs["data_frame"].empty:
            df_split = [frame_ for frame_ in np.array_split(kwargs["data_frame"], self.n_cpu) if not frame_.empty]
            
            pass_args = [(kwargs["used_class"], kwargs["used_name"], frame_, False) for frame_ in df_split]
            
            self._verbose("Do", kwargs["used_class"], kwargs["used_name"], kwargs["data_frame"].shape, True)

            with mp.Pool(processes=self.n_cpu) as p:
                results_ = p.map(self.get_partial, pass_args)
                collected_frame = pd.concat([item for item in results_])

            self._verbose("Done", kwargs["used_class"], kwargs["used_name"], collected_frame.shape)
            
            return collected_frame
        else:
            return kwargs["data_frame"]
    
    def __do_quicksave(self, name):
        """
        Saves the pd.DataFrame in a .csv file.

        :param name: str
        """
        if type(name).__name__ == "str":
            if not self.data.empty:
                to_place_dir, _ = os.path.split(name)
                if not os.path.isdir(to_place_dir):
                    os.makedirs(to_place_dir, exist_ok=True)
                self.data.to_csv(name, index=False, sep=";")
    
    @staticmethod
    def help():
        """
        Shows all possible reconstruction, filtering processes as well as the sizes that can be added.
        """

        print("\nPossible Filter:")
        print("\n".join(reversed([f" - {item}" for item in dir(Apply.filter_instance) if "filter_" in item])))
        
        print("\nPossible Reconstructions:")
        print("\n".join(reversed([f" - {item}" for item in dir(Apply.reconstruction_instance) if "reco" in item])))
        
        print("\nAdding possible variables:")
        print("\n".join(reversed([f" - {item}" for item in dir(Apply.add_variable_instance) if "add" in item and "to_row" not in item])))

    def _raise_error(self, method, args):
        if method == self.run:
            raise NameError(f"{args} operation does not exist. Use Apply.help() to list all possible methods")
        
    def run(self, name, quicksave=None):

        _used_class = self.filter_instance if "filter" in name else(
                        self.reconstruction_instance if "reco" in name else(
                            self.add_variable_instance if "add" in name else(
                                self._raise_error(self.run, name))))
        _used_func = name
        
        if not self.multi_cpu:
            self.data = self.get_partial(arg_tuple=False, used_class=_used_class, used_name=name, data_frame=self.data)
        if self.multi_cpu:
            self.data = self.__multiprocessing(used_class=_used_class, used_name=name, data_frame=self.data)
        
        self.__do_quicksave(quicksave)
    
    def hist(self, variable, bins, hist_range, filter_=None, **kwargs):
        """
        Draws the current unscaled distribution of the variable "variable".

        :param variable: str
        :param bins: int
        :param hist_range: tuple
                           (float, float); (lower_hist_range, upper_hist_range)
        :param filter_: list
                        [str, (float, float)]; [filter_based_on_row_name, (lower_value, upper_value)]
        :param kwargs: matplotlib drawing kwargs
        """
        _hist = Hist(bins=bins, hist_range=hist_range)
        
        col__ = []
        if isinstance(variable, str): col__ = [it for it in self.data.columns if variable in it or it in variable]
        
        if filter_ is not None: col__.append(filter_[0])
        
        _hist.fill_hist(name="undefined", array_of_interest=_hist.convert_column(col_=self.data.run(col__), filter_=filter_))
        _hist.draw(pass_name="undefined", **kwargs)
