# -*- coding: UTF-8 -*-
import ast
import errno
import multiprocessing as mp
import os
import types
import warnings
from copy import deepcopy

import awkward1 as awk
import numpy as np
import pandas as pd
from tqdm import tqdm

from .CalcAndAllowerInit import CalcInit, FilterInit
from .FilterRecoAdd import FilterStr, Reconstruct, AddVariable
from .ProcessingBasicAwk import ProcessingBasicAwk
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
                 calc_instance=CalcInit, filter_instance=FilterInit,
                 multi_cpu=False, n_cpu=mp.cpu_count(), verbose=True):
        
        self.verbose = verbose
        self.n_cpu = n_cpu
        self.multi_cpu = multi_cpu
        
        self.filename = file
        
        self.particle_type = None
        
        self.nrows = nrows
        self.awk_data = self.get_awk_from_file()
        self.directory, self.filename = os.path.split(file)
        
        Apply.set_instance(calc_instance=calc_instance, filter_instance=filter_instance)
        Apply.filter_steps_instance.set_instance(calc_instance=calc_instance, filter_instance=filter_instance)
        Apply.reconstruction_instance.set_instance(calc_instance=calc_instance, filter_instance=filter_instance)
        Apply.add_variable_instance.set_instance(calc_instance=calc_instance, filter_instance=filter_instance)
        
        # Here's a little lesson in trickery...
        self.help = self._instance_help
        
        self._data = None
        self._data_changed = True
    
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
        
        _str = f"{msg_: <{7}} {cls_.__name__: <{12}}: {fnc_: <{35}}; shape: {str(shape_): <{15}}"
        print(_str, end="\r" if flush_ else "\n", flush=flush_)
    
    @staticmethod
    def _get_record_shape(list_of_awk_array):
        if isinstance(list_of_awk_array, list):
            _columns = list_of_awk_array[0].fields
            _len = sum(len(awk_a[_columns[0]]) for awk_a in list_of_awk_array)
            return _len, len(_columns)
    
    @staticmethod
    def _empty_awk(awk_array):
        if len(awk_array[awk_array.fields[-1]]) == 0:
            return True
        else:
            return False
    
    @staticmethod
    def _build_help_parts(msg1, inst_, key):
        _ignore = [it for it in dir(ProcessingBasicAwk)]
        _ignore.extend(["instance", "type", "misshit", "lepton_detector_classification", "__"])
        print(f"\n\n{msg1} ('{key}'):")
        print("\n".join([f" - {item}" for item in dir(inst_) if not any(it in item for it in _ignore)]))
    
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
    
    def set_multi_cpu(self, multi_cpu=None, n_cpu=None):
        self.multi_cpu = multi_cpu if multi_cpu is not None else self.multi_cpu
        self.n_cpu = n_cpu if n_cpu is not None else self.n_cpu
    
    def set_particle_type(self, columns, particle=None):
        if particle is None:
            if any("muon_" in it for it in columns) and any("electron_" in it for it in columns):
                self.particle_type = "both"
            if any("muon_" in it for it in columns) and not any("electron_" in it for it in columns):
                self.particle_type = "muon"
            if not any("muon_" in it for it in columns) and any("electron_" in it for it in columns):
                self.particle_type = "electron"
        if isinstance(particle, str) and any(particle == it for it in ["electron", "muon", "both"]):
            self.particle_type = particle
    
    def _get_dataframe_string_chuncks(self, chunks=10):
        _length = pd.read_csv(self.filename, sep=";", nrows=self.nrows).shape[0]
        return [chunk for chunk in pd.read_csv(self.filename, sep=";", nrows=self.nrows, chunksize=int(_length / (chunks - 0.1)))]
    
    @staticmethod
    def _get_converter_func(df):
        
        _cont = lambda x: x
        _list = lambda column_: True if "," in f"{df[column_][0]}" else False
        
        return {c: lambda x: (np.array if _list(c) else _cont)(ast.literal_eval(f"[{x}]" if _list(c) else x)) for c in df.columns}
    
    def _get_awk_from_chunk(self, df_chunk):
        
        _converters = self._get_converter_func(pd.read_csv(self.filename, sep=";", nrows=1))
        
        for name in df_chunk.columns:
            df_chunk[name] = df_chunk[name].apply(_converters[name])
        
        _dict = {name: df_chunk[name].values for name in df_chunk.columns}
        
        _1d_vars = ["mass", "tag", "event", "run", "lumi"]
        
        for k in _dict.keys():
            if any(it in k for it in _1d_vars):
                _dict[k] = np.concatenate(_dict[k])
        
        return awk.from_iter([dict(zip(_dict, item)) for item in zip(*_dict.values())])
        # return awk.from_iter(_dict)
    
    def get_awk_from_file(self):
        """
        Load the pandas.DataFrame from a file.

        :return: awk.Array
        """
        
        if not os.path.exists(self.filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.filename)
        
        string_chunks = self._get_dataframe_string_chuncks(chunks=self.n_cpu if self.multi_cpu else 10)
        self.set_particle_type(pd.read_csv(self.filename, sep=";", nrows=1).columns)
        
        print(f"Reading {self.filename} and converting to processable format ...")
        
        if not self.multi_cpu:
            awk_data = [self._get_awk_from_chunk(chunk) for chunk in tqdm(string_chunks)]
        
        if self.multi_cpu:
            with mp.Pool(self.n_cpu) as p:
                awk_data = p.map(self._get_awk_from_chunk, string_chunks)
        
        print(f"Reading {self.filename} and converting to processable format done.")
        
        return awk_data
    
    @staticmethod
    def _dummy_for_multiprocessing(arg):
        return awk.to_list(arg[0](awk.from_iter(arg[1])))
    
    def do_multiprocess(self, cls_, fnc_, vb_):
        if self.awk_data:
            if vb_:
                self._verbose("Do", cls_, fnc_, self._get_record_shape(self.awk_data))
            
            with mp.Pool(self.n_cpu) as p:
                _results = p.map(self._dummy_for_multiprocessing, [(getattr(cls_(look_for=self.particle_type), fnc_),
                                                                    deepcopy(awk.to_list(item))) for item in self.awk_data])
            
            self.awk_data = [awk.from_iter(item) for item in _results]
            self.awk_data = [awk_array for awk_array in self.awk_data if not self._empty_awk(awk_array)]
            
            if vb_:
                self._verbose("Done", cls_, fnc_, self._get_record_shape(self.awk_data))
    
    def do_sequential(self, cls_, fnc_, vb_, progress_vb_=False):
        
        if self.awk_data:
            if vb_:
                self._verbose("Do", cls_, fnc_, self._get_record_shape(self.awk_data), ~progress_vb_)
            
            for i in tqdm(range(len(self.awk_data))) if progress_vb_ else range(len(self.awk_data)):
                if not self._empty_awk(self.awk_data[i]):
                    _tmp = getattr(cls_(look_for=self.particle_type), fnc_)(awk_array=self.awk_data[i])
                    self.awk_data[i] = _tmp
            
            self.awk_data = [awk_array for awk_array in self.awk_data if not self._empty_awk(awk_array)]
            
            if vb_:
                self._verbose("Done", cls_, fnc_, self._get_record_shape(self.awk_data))
    
    def filter(self, name, save_path=None):
        
        if not self.multi_cpu:
            self.do_sequential(cls_=self.filter_steps_instance, fnc_=name, vb_=self.verbose, progress_vb_=self.verbose)
        
        if self.multi_cpu:
            self.do_multiprocess(cls_=self.filter_steps_instance, fnc_=name, vb_=self.verbose)
        
        self._data_changed = True
        
        self.save(save_path)
    
    def add(self, name, save_path=None):
        
        if not self.multi_cpu:
            self.do_sequential(cls_=self.add_variable_instance, fnc_=name, vb_=self.verbose, progress_vb_=self.verbose)
        
        if self.multi_cpu:
            self.do_multiprocess(cls_=self.add_variable_instance, fnc_=name, vb_=self.verbose)
        
        self._data_changed = True
        
        self.save(save_path)
    
    def reconstruct(self, name, save_path=None):
        
        if not self.multi_cpu:
            self.do_sequential(cls_=self.reconstruction_instance, fnc_=name, vb_=self.verbose, progress_vb_=self.verbose)
        
        if self.multi_cpu:
            self.do_multiprocess(cls_=self.reconstruction_instance, fnc_=name, vb_=self.verbose)
        
        self._data_changed = True
        
        self.save(save_path)
    
    def save(self, save_path):
        if type(save_path).__name__ == "str":
            if not self.data.empty:
                to_place_dir, _ = os.path.split(save_path)
                if not os.path.isdir(to_place_dir):
                    os.makedirs(to_place_dir, exist_ok=True)
                self.data.to_csv(save_path, index=False, sep=";")
    
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
    
    @property
    def data(self):
        
        _1d_vars = ["event", "run", "lumi", "mass"]
        
        if self._data_changed or self._data is None:
            _awk_data = deepcopy(self.awk_data)
            
            for item in tqdm(_awk_data):
                for name in item.fields:
                    try:
                        item[name] = awk.from_iter([",".join([f'{subit}' if "tag" not in name else f'"{subit}"' for subit in it])
                                                    for it in awk.to_list(item[name])])
                    except TypeError:
                        try:
                            item[name] = awk.concatenate(item[name])
                        except ValueError:
                            item[name] = awk.from_iter(awk.to_list(item[name]))
            
            _df = pd.DataFrame().append([awk.to_pandas(item) for item in _awk_data], ignore_index=True)
            
            self._data_changed = False
            
            self._data = _df
            
            return _df
        else:
            return self._data
