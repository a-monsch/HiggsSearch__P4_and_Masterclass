# -*- coding: UTF-8 -*-
import multiprocessing as mp
import os
import errno

import numpy as np
import pandas as pd
import swifter

from .CalcAndAllowerInit import CalcInit, AllowedInit
from .FilterRecoAdd import FilterStr, Reconstruct, AddVariable
from ..histogramm.Hist import Hist


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray

sw_temp_ = swifter


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
                 n_cpu=mp.cpu_count(),
                 use_swifter=False):
        
        self.particle_type = particle_type
        self.use_swifter = use_swifter
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
        if any("filter" in it for it in kwargs.keys()): Apply.filter_instance = kwargs["filter_instance"]
        if any("reconstruct" in it for it in kwargs.keys()):
            Apply.reconstruction_instance = kwargs["reconstruct_instance"]
        if any("add_variable" in it for it in kwargs.keys()):
            Apply.add_variable_instance = kwargs["add_variable_instance"]
    
    def get_partial(self, arg_tuple=False, **kwargs):
        """
        Partially constructs and evaluates the pandas apply function using the given arguments.

        :param arg_tuple: tuple
                          (class_name, used_name, data_frame verbose)
                          used_name is the used method name of the provided class
        :param kwargs: see tuple, except without verbose
        :return: pd.DataFrame
        """
        cls_, fnc_, df, vb_ = arg_tuple if arg_tuple else (kwargs["used_class"], kwargs["used_name"], kwargs["data_frame"], True)
        
        if vb_:
            print(f"Do   {cls_.__name__: <{12}}: {fnc_: <{20}}; shape: {str(df.shape): <{15}}", end="\r", flush=True)
        if not df.empty:
            if not self.use_swifter:
                df = df.apply(lambda x: getattr(cls_(x, list(df)), fnc_)(look_for=self.particle_type), axis=1)
            if self.use_swifter:
                df = df.swifter.apply(lambda x: getattr(cls_(x, list(df)), fnc_)(look_for=self.particle_type), axis=1)
            df = df.dropna()
            if vb_:
                print(f"Done {cls_.__name__: <{12}}: {fnc_: <{15}}; shape: {str(df.shape): <{15}}")
            return df
    
    def __multiprocessing(self, **kwargs):
        """
        Executes the Apply.get_partial in a pool process with self.n_cpu number of processes.

        :param kwargs: see Apply.get_partial
        :return: pd.DataFrame
        """
        if not kwargs["data_frame"].empty:
            df_split = np.array_split(kwargs["data_frame"], self.n_cpu)
            pass_args = [(kwargs["used_class"], kwargs["used_name"], frame_, False) for frame_ in df_split]
            pool = mp.Pool(processes=self.n_cpu)
            print(
                f'Do   {kwargs["used_class"].__name__: <{12}}: {kwargs["used_name"]: <{20}}; shape: {str(kwargs["data_frame"].shape): <{15}}',
                end="\r", flush=True)
            results = pool.map(self.get_partial, pass_args)
            pool.close()
            pool.join()
            collected_frame = pd.concat([item for item in results])
            print(
                f'Done {kwargs["used_class"].__name__: <{12}}: {kwargs["used_name"]: <{20}}; shape: {str(collected_frame.shape): <{15}}')
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
        print("Possible Filter:\n" +
              "- 'check_type'\n- 'check_q'\n- 'check_min_pt'\n- 'check_eta'\n- 'check_misshit'\n" +
              "- 'check_rel_iso'\n- 'check_impact_param'\n- 'check_exact_pt'\n- 'check_m_2l'\n- 'check_m_4l'\n" +
              "\n")
        
        print("Possible Reconstructions:\n" +
              "- 'zz'\n" +
              "- 'mass_4l_out_zz'\n" +
              "\n")
        
        print("Adding possible variables: \n" +
              "- 'pt'\n" +
              "- 'eta'\n" +
              "- 'phi'\n")
    
    def add_variable(self, variable_name, quicksave=None):
        """
        Adds the size variable_name to the pd.DataFrame.

        :param variable_name: str
        :param quicksave: str
        :return: pd.DataFrame
        """
        if not self.multi_cpu:
            self.data = self.get_partial(used_class=self.add_variable_instance, used_name=variable_name,
                                         data_frame=self.data)
        if self.multi_cpu:
            self.data = self.__multiprocessing(used_class=self.add_variable_instance, used_name=variable_name,
                                               data_frame=self.data)
        
        self.__do_quicksave(name=quicksave)
    
    def filter(self, filter_name, quicksave=None):
        """
        Applies the filter filter_name to the pd.DataFrame.

        :param filter_name: str
        :param quicksave: str
        :return: pd.DataFrame
        """
        if not self.multi_cpu:
            self.data = self.get_partial(used_class=self.filter_instance, used_name=filter_name, data_frame=self.data)
        
        if self.multi_cpu:
            self.data = self.__multiprocessing(used_class=self.filter_instance, used_name=filter_name,
                                               data_frame=self.data)
        
        self.__do_quicksave(quicksave)
    
    def reconstruct(self, reco_name, quicksave=None):
        """
        Applies the reconstruction step reco_name to the pd.DataFrame.

        :param reco_name: str
        :param quicksave: str
        :return: pd.DataFrame
        """
        if not self.multi_cpu:
            self.data = self.get_partial(used_class=self.reconstruction_instance, used_name=reco_name,
                                         data_frame=self.data)
        if self.multi_cpu:
            self.data = self.__multiprocessing(used_class=self.reconstruction_instance, used_name=reco_name,
                                               data_frame=self.data)
        
        self.__do_quicksave(quicksave)
    
    def hist_of_variable(self, variable, bins, hist_range, filter_=None, **kwargs):
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
        hist = Hist(bins=bins, hist_range=hist_range)
        
        col__ = []
        if isinstance(variable, str): col__ = [it for it in self.data.columns if variable in it or it in variable]
        
        if filter_ is not None: col__.append(filter_[0])
        
        hist.fill_hist(name="undefined", array_of_interest=hist.convert_column(col_=self.data.filter(col__), filter_=filter_))
        hist.draw(pass_name="undefined", **kwargs)
