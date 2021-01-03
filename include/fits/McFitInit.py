import os

import kafe2 as K2
import matplotlib.pyplot as plt
import numpy as np

from ..RandomHelper import ToSortHelper as TSH
from ..histogramm.HistDataGetter import HistDataGetter as HDG


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray


class McFitInit(object):
    """
    Class that performs an adjustment of the distribution of the MC simulations using the kafe2 module.
    """
    
    def __init__(self, bins=15, hist_range=(106, 151), tag="background",
                 to_chi2_one=False,
                 mc_dir="./data/mc_aftH", ru_dir="./data/ru_aftH",
                 verbose=True, info=None, error_type_model="relative", save_dir="./mc_fits"):
        self.verbose = verbose
        self.error_type_model = error_type_model
        self.to_chi2_one = to_chi2_one
        self.bins = bins
        self.hist_range = hist_range
        self.tag = tag
        self.info = [["2012"], ["B", "C"]] if info is None else info
        self.mc_dir = mc_dir
        self.ru_dir = ru_dir
        self.drfd = {}  # DataRawFitDict
        self.func_com = {}
        self.data_raw = self.set_data_raw()
        self.data_com = self.set_data_com()
        self.combined_errors = self.create_combined_errors()
        self.save_dir = save_dir
        self.create_dir()
    
    def create_dir(self):
        """
        Creates folder to save.
        """
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_data_raw(self):
        """
        Get raw_date as Hist object from HistDataGetter.

        Note: Used for calculation of uncertainties
        """
        d = HDG(bins=self.bins, hist_range=self.hist_range, info=self.info, mc_dir=self.mc_dir, ru_dir=self.ru_dir)
        return d.get_mc_raw(tag=self.tag)
    
    def set_data_com(self):
        """
        Creates a Hist object from HistDataGetter.
        Contains the already scaled histogram with all channels.
        """
        d = HDG(bins=self.bins, hist_range=self.hist_range, info=self.info, mc_dir=self.mc_dir, ru_dir=self.ru_dir)
        return d.get_mc_com()
    
    def create_combined_errors(self):
        """
        Creates an array of seld.data_raw after gaussian error propagation.

        :return: ndarray
                 1D array containing data with "float" type.
        """
        combined_errors = np.zeros(len(self.data_com.x_range))
        for i in range(len(self.data_raw.x_range)):
            pass_y_error_temp = 0
            for key in list(self.data_raw.data.keys()):
                corr_fac_square = (self.data_raw.data[key]["corr_fac"]) ** 2
                pass_y_error_temp += (self.data_raw.data[key]["raw_hist"][i] * corr_fac_square)
            combined_errors[i] = np.sqrt(pass_y_error_temp)
        return combined_errors
    
    def create_raw_fit(self, used_func):
        """
        Fits the function to the distribution.
        Repeats the step with scaled uncertainties if to_chi2_one=True.
        The kafe2 object is cached.

        :param used_func: function
        Function to create a kafe2 fit object
        
        :param used_func: function
        """
        # may be implemented by students
    
    def plot_fit(self, used_func):
        """
        Draws the result of the performed kafe2 fit using the kafe2 plot method.

        :param used_func: list
                          1D list containing functions
        """
        
        # may be implemented by students
    
    def get_results(self, used_func):
        """
        Collects and passes the parameters and uncertainties
        to the used_func as a tuple if the fit was performed.

        :param used_func: functon
        :return: tuple
                 (parameter_values, parameter_errors)
        """
        
        # may be implemented by students
    
    @property
    def variables_for_xy_fit(self):
        """
        Passes all necessary variables for a standalone xy fit
        (the uncertainties on the data are already calculated).

        :return: tuple
                 (x_data, y_data, y_errors)
        """
        to_pass_tuple = []
        
        to_pass_tuple.append(self.data_com.x_range)
        if self.tag.lower() == "background": to_pass_tuple.append(self.data_com.data["mc_bac"])
        if self.tag.lower() == "signal": to_pass_tuple.append(self.data_com.data["mc_sig"])
        to_pass_tuple.append(self.combined_errors)
        
        return tuple(to_pass_tuple)
    
    @property
    def variables_for_hist_fit(self):
        """
        Passes all necessary variables for a standalone histogram fit
        (the uncertainties on the data are already calculated).

        :return: tuple
                 (bins, histogramm_range, histogramm_data, histogramm_data_errors)
        """
        to_pass_tuple = []
        to_pass_tuple.append(self.bins)
        to_pass_tuple.append(self.hist_range)
        if self.tag.lower() == "background": to_pass_tuple.append(self.data_com.data["mc_bac"])
        if self.tag.lower() == "signal": to_pass_tuple.append(self.data_com.data["mc_sig"])
        to_pass_tuple.append(self.combined_errors)
        return tuple(to_pass_tuple)
