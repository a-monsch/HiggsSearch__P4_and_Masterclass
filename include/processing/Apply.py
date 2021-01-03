# -*- coding: UTF-8 -*-
import errno
import multiprocessing as mp
import os
import types
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from .CalcAndAllowerInit import CalcInit, FilterInit
from .FilterRecoAdd import FilterStr, Reconstruct, AddVariable
from .ProcessingRow import ProcessingRow
from ..RandomHelper import ReformatText, in_notebook
from ..histogramm.Hist import Hist

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def copy_func(func, name=None):
    _func = types.FunctionType(func.__code__, func.__globals__, name or func.__name__, func.__defaults__, func.__closure__)
    _func.__dict__.update(func.__dict__)
    return _func


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
    filter_instance = FilterInit
    reconstruction_instance = Reconstruct
    filter_steps_instance = FilterStr
    add_variable_instance = AddVariable
    
    def __init__(self, file, nrows=None,
                 calc_instance=CalcInit,
                 filter_instance=FilterInit,
                 multi_cpu=False,
                 n_cpu=mp.cpu_count()):
        
        self.n_cpu = n_cpu
        self.multi_cpu = multi_cpu
        
        if type(file) is str:
            self.data, self.name = Apply.from_file(file=file, nrows=nrows)
            self.directory, self.filename = os.path.split(file)
        
        else:
            self.data = file
        
        self.particle_type = self.set_particle_type()  # particle_type
        
        self.calculated_dump = {}
        Apply.set_instance(calc_instance=calc_instance, filter_instance=filter_instance)
        Apply.filter_steps_instance.set_instance(calc_instance=calc_instance, filter_instance=filter_instance)
        Apply.reconstruction_instance.set_instance(calc_instance=calc_instance, filter_instance=filter_instance)
        Apply.add_variable_instance.set_instance(calc_instance=calc_instance, filter_instance=filter_instance)
        
        # Here's a little lesson in trickery...
        self.help = self._instance_help
    
    def set_particle_type(self):
        _cols = list(self.data.columns)
        if any("muon_" in it for it in _cols) and any("electron_" in it for it in _cols):
            return "both"
        if any("muon_" in it for it in _cols) and not any("electron_" in it for it in _cols):
            return "muon"
        if not any("muon_" in it for it in _cols) and any("electron_" in it for it in _cols):
            return "electron"
    
    @staticmethod
    def from_file(file, nrows=None):
        """
        Load the pandas.DataFrame from a file.

        :param file: str
        :param nrows: int
        :return: pd.DataFrame, str
        """
        
        if not os.path.exists(file):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)
        
        print(f"Loading {file} ...", end="\r", flush=True)
        data_frame = pd.read_csv(file, sep=";", nrows=nrows)
        print(f"Loading {file} done.")
        return data_frame, file
    
    @classmethod
    def set_instance(cls, **kwargs):
        """
        Sets the class instances used in the following.

        :param kwargs: class instances
        """
        if any("calc" in it for it in kwargs.keys()):
            Apply.calc_instance = kwargs["calc_instance"]
        if any("filter" in it for it in kwargs.keys()):
            Apply.filter_instance = kwargs["filter_instance"]
    
    @staticmethod
    def _verbose(msg_, cls_, fnc_, shape_, flush_=False):
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
    
    def do_partial(self, arg_tuple, **kwargs):
        """
        Partially constructs and evaluates the pandas run function using the given arguments.

        :param arg_tuple: tuple
                          (class_name, used_name, data_frame verbose)
                          used_name is the used method name of the provided class
        :param kwargs: see tuple, except without verbose
        :return: pd.DataFrame
        """
        cls_, fnc_, df, vb_ = arg_tuple if arg_tuple else (kwargs["used_class"], kwargs["used_name"], kwargs["data_frame"], True)
        
        if vb_:
            self._verbose("Do", cls_, fnc_, df.shape, True)
        
        if not df.empty:
            if self.multi_cpu:
                df = df.apply(lambda x: getattr(cls_(x, list(df), look_for=self.particle_type), fnc_)(), axis=1)
            if not self.multi_cpu:
                tqdm.pandas()
                df = df.progress_apply(lambda x: getattr(cls_(x, list(df), look_for=self.particle_type), fnc_)(), axis=1)
            df = df.dropna()
            if vb_:
                self._verbose("Done", cls_, fnc_, df.shape)
            return df
        if df.empty:
            return df
    
    def __multiprocessing(self, **kwargs):
        """
        Executes the Apply.do_partial in a pool process with self.n_cpu number of processes.

        :param kwargs: see Apply.do_partial
        :return: pd.DataFrame
        """
        if not kwargs["data_frame"].empty:
            df_split = [frame_ for frame_ in np.array_split(kwargs["data_frame"], self.n_cpu) if not frame_.empty]
            
            pass_args = [(kwargs["used_class"], kwargs["used_name"], frame_, False) for frame_ in df_split]
            
            self._verbose("Do", kwargs["used_class"], kwargs["used_name"], kwargs["data_frame"].shape, True)
            
            with mp.Pool(processes=self.n_cpu) as p:
                results_ = p.map(self.do_partial, pass_args)
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
    def _build_help_parts(msg1, inst_, key):
        _ignore = [it for it in dir(ProcessingRow)]
        _ignore.extend(["instance", "type", "misshit", "lepton_detector_classification"])
        print(f"\n\n{msg1} ('{key}'):")
        print("\n".join([f" - {item}" for item in dir(inst_) if not any(it in item for it in _ignore)]))
    
    def _instance_help(self, workflow=None, method=None):
        """
        Shows all possible reconstruction, filtering processes as well as the sizes that can be added in this instance.
        """
        
        if method is None:
            Apply._build_help_parts("Possible Filter", self.filter_steps_instance, "filter")
            Apply._build_help_parts("Possible Reconstructions", self.reconstruction_instance, "reconstruct")
            Apply._build_help_parts("Adding possible variables", self.add_variable_instance, "add")
        
        if method is not None:
            _cls = self.filter_steps_instance if "filter" in workflow else (
                self.add_variable_instance if "add" in workflow else (
                    self.reconstruction_instance if "reconstruct" in workflow else None))
            if _cls:
                _tmp = copy_func(getattr(_cls, method))
                if in_notebook() or True:
                    _tmp.__doc__ = ReformatText.color_docs(_tmp.__doc__, calc_instance=self.calc_instance, filter_instance=self.filter_instance)
                return help(_tmp)
    
    @staticmethod
    def help(workflow=None, method=None):
        """
        Shows all possible reconstruction, filtering processes as well as the sizes that can be added.
        """
        
        if method is not None:
            
            _cls = Apply.filter_steps_instance if "filter" in workflow else (
                Apply.add_variable_instance if "add" in workflow else (
                    Apply.reconstruction_instance if "reconstruct" in workflow else None))
            
            if _cls:
                _tmp = copy_func(getattr(_cls, method))
                if in_notebook():
                    _tmp.__doc__ = ReformatText.color_docs(_tmp.__doc__, calc_instance=None, filter_instance=None)
                return help(_tmp)
        
        if method is None:
            Apply._build_help_parts("Possible Filter", Apply.filter_steps_instance, "filter")
            Apply._build_help_parts("Possible Reconstructions", Apply.reconstruction_instance, "reconstruct")
            Apply._build_help_parts("Adding possible variables", Apply.add_variable_instance, "add")
    
    def filter(self, name, save_path=None):
        
        if not self.multi_cpu:
            self.data = self.do_partial(arg_tuple=False, used_class=self.filter_steps_instance, used_name=name, data_frame=self.data)
        if self.multi_cpu:
            self.data = self.__multiprocessing(used_class=self.filter_steps_instance, used_name=name, data_frame=self.data)
        
        self.__do_quicksave(save_path)
    
    def add(self, name, save_path=None):
        
        if not self.multi_cpu:
            self.data = self.do_partial(arg_tuple=False, used_class=self.add_variable_instance, used_name=name, data_frame=self.data)
        if self.multi_cpu:
            self.data = self.__multiprocessing(used_class=self.add_variable_instance, used_name=name, data_frame=self.data)
        
        self.__do_quicksave(save_path)
    
    def reconstruct(self, name, save_path=None):
        
        if not self.multi_cpu:
            self.data = self.do_partial(arg_tuple=False, used_class=self.reconstruction_instance, used_name=name, data_frame=self.data)
        if self.multi_cpu:
            self.data = self.__multiprocessing(used_class=self.reconstruction_instance, used_name=name, data_frame=self.data)
        
        self.__do_quicksave(save_path)
    
    def save(self, save_path):
        self.__do_quicksave(name=save_path)
    
    def hist(self, variable, bins, hist_range, lepton_number=None, filter_by=None, **kwargs):
        """
        Draws the current unscaled distribution of the variable "variable".

        :param variable: str
        :param bins: int
        :param hist_range: tuple
                           (float, float); (lower_hist_range, upper_hist_range)
        :param lepton_number: int or list containing the desired leptons
        :param filter_by: list
                        [str, (float, float)]; [filter_based_on_row_name, (lower_value, upper_value)]
        :param kwargs: matplotlib drawing kwargs
        """
        hist_ = Hist(bins=bins, hist_range=hist_range)
        
        col__ = []
        if isinstance(variable, str):
            col__ = [it for it in self.data.columns if variable in it or it in variable]
        
        if filter_by is not None:
            col__.append(filter_by[0])
        
        hist_.fill(name="undefined", array_of_interest=hist_.convert_column(col_=self.data.filter(col__),
                                                                            filter_by=filter_by, lepton_number=lepton_number))
        hist_.draw(pass_name="undefined", **kwargs)
