# -*- coding: UTF-8 -*-

import contextlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .HistHelper import HistHelper


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray


class Hist(HistHelper):
    """
    Class that creates a histogram and, if necessary, corrects and draws or saves it according to the conditions.
    """
    
    def __init__(self, bins, hist_range, signal_mass_mc=None, input_dict_data=None, save_dir="."):
        self.__cross_sec = {"2011": {"ZZ_4mu": 66.09, "ZZ_4el": 66.09, "ZZ_2el2mu": 152, "H_ZZ": 5.7},
                            "2012": {"ZZ_4mu": 76.91, "ZZ_4el": 76.91, "ZZ_2el2mu": 176.7, "H_ZZ": 6.5}}
        self.__k_factor = {"2011": {"ZZ_4mu": 1.386, "ZZ_4el": 1.386, "ZZ_2el2mu": 1.386, "H_ZZ": 1.0},
                           "2012": {"ZZ_4mu": 1.386, "ZZ_4el": 1.386, "ZZ_2el2mu": 1.386, "H_ZZ": 1.0}}
        self.__lumen = {"2011": {"A": 4.7499}, "2012": {"A": np.longdouble(0.889391999043448687),
                                                        "B": np.longdouble(4.429375295985512733),
                                                        "C": np.longdouble(7.152728016920716286),
                                                        "D": np.longdouble(7.318301466596649170),
                                                        "A-D": np.longdouble(19.789796778546325684)}}
        
        self.__event_num_mc = {"2011": {"ZZ_4mu": 1447136, "ZZ_4el": 1493308, "ZZ_2el2mu": 1479879, "H_ZZ": 299683},
                               "2012": {"ZZ_4mu": 1499064, "ZZ_4el": 1499093, "ZZ_2el2mu": 1497445, "H_ZZ": 299973}}
        self.set_signal_mc_mass(mh_mass=signal_mass_mc)
        
        self.save_dir = save_dir
        self.create_dir()
        
        self.bins = bins
        self.hist_range = hist_range
        self.bin_edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
        self.x_range = self.bin_edges[:-1] + np.abs(self.bin_edges[0] - self.bin_edges[1]) / 2.
        self.bin_width = np.abs(self.x_range[0] - self.x_range[1])
        
        self.data = {} if input_dict_data is None else input_dict_data
    
    def create_dir(self, dir_="./histogramms"):
        """
        Creates directory.

        :param dir_: str
        """
        if not os.path.exists(dir_):
            os.mkdir(dir_)
            self.save_dir = dir_
    
    @classmethod
    def from_dataframe_with_bins(cls, data_frame, column, bins, hist_range, filter_=None):
        """
        Alternative constructor for a pd.DataFrame that not cointains bin information.

        :param data_frame: pd.DataFrame
        :param column: str
        :param bins: int
        :param hist_range: tuple
                           (lower_value_lim, upper_value_lim)
        :param filter_: list
                        ["column_name", (lower_value_lim, upper_value_lim)]
        """
        to_pass_data = {column, HistHelper.convert_column(data_frame[column], filter_=filter_)}
        return cls(bins=bins, hist_range=hist_range, input_dict_data=to_pass_data)
    
    @classmethod
    def from_dataframe(cls, file_):
        """
        Alternative constructor for a pd.DataFrame from file that cointains bin information.

        :param file_: str
        """
        try:
            df_ = pd.read_csv(file_)
        except:
            df_ = file_
        
        dd_, bin_num_, hist_range_ = {}, 0, 0
        for name_ in list(df_):
            if "x_range" in name_:
                bin_num_ = len(df_[name_].values)
                width_ = abs(df_[name_].values[0] - df_[name_].values[1])
                hist_range_ = (np.amin(df_[name_].values) - width_ / 2., np.amax(df_[name_].values) + width_ / 2.)
                continue
            dd_[name_] = df_[name_].values
        return cls(bins=bin_num_, hist_range=hist_range_, input_dict_data=dd_)
    
    def set_signal_mc_mass(self, mh_mass=None):
        """
        Sets the used mass of the Higgs boson and the corresponding
        number of events in the MC simulation.

        :param mh_mass: int
        """
        mass_event_dict = {115: 299971, 120: 295473, 122: 299970, 124: 276573, 128: 267274,
                           130: 299976, 135: 299971, 140: 299971, 145: 287375, 150: 299973}
        if mh_mass is not None:
            self.__event_num_mc["2012"]["H_ZZ"] = mass_event_dict[mh_mass]
    
    def set_empty_bin(self, name):
        """
        Creates empty bins for name.

        :param name: str or list
        """
        try:
            self.data[name] = np.zeros(len(self.x_range))
        except TypeError:
            for it in name:
                self.data[it] = np.zeros(len(self.x_range))
    
    def corr_fac(self, year, run, process):
        """
        Calculation of the correction factor for the histograms of the individual channels.

        :param year: str
        :param run: str or list
                    containing "A", "B", "C", "D" or "A-D" if all at once.
        :param process: str
                        "ZZ_4mu", "ZZ_4el", "ZZ_2el2mu" or "H_ZZ"
        :return: ndarray
                 1D array containing correction factor with "float" type.
        """
        if isinstance(run, str): run = [run]
        tot_luminosity = np.sum(lum for run_, lum in self.__lumen[year].items() if run_ in run)
        
        fac_ = self.__k_factor[year][process] * self.__cross_sec[year][process] / self.__event_num_mc[year][process]
        
        return tot_luminosity * fac_
    
    def fill_hist(self, name, array_of_interest, info=None, get_raw=False):
        """
        Fills the created histogram "name" with the array "array_of_interests".
        If "track" is included in the name, all intermediate steps and the corresponding
        correction factor are also stored separately in a dict.

        :param name: str
        :param array_of_interest: ndarray
                                  1D array containing data with "float" type.
        :param info: list
                     [["year"], ["run"], ["process"]]
        :param get_raw: bool
                        for HistDataGetter only
        """
        if name not in self.data.keys():
            self.set_empty_bin(name=name)
        
        pile_array = np.zeros(len(self.x_range))
        year, run, process = (None, None, None) if "mc" not in name else (info[0], info[1], info[2])
        
        if "data" in name or "undefined" in name:
            pile_array, _ = np.histogram(a=array_of_interest, bins=self.bins, range=self.hist_range)
        
        if "mc" in name:
            pile_array, _ = np.histogram(a=array_of_interest, bins=self.bins, range=self.hist_range)
            pile_array = pile_array * self.corr_fac(year, run, process)
        
        if not get_raw:
            self.data[name] += pile_array
        
        if get_raw and "mc" in name or "track" in name:
            pile_array, _ = np.histogram(a=array_of_interest, bins=self.bins, range=self.hist_range)
            self.data[name] = {"raw_hist": pile_array, "corr_fac": self.corr_fac(year, run, process),
                               "raw_data": array_of_interest}
    
    def fill_hist_from_dir(self, col_, dir_,
                           info=None, filter_=None, name_data_=None,
                           name_sig_mc_=None, name_bac_mc_=None,
                           file_list_=None, **kwargs):
        """
        Fills the histogram according to the used files in dir_ or file_list_ and
        saves all intermediate steps if get_raw is passed.
        See self.fill_hist(...).


        :param col_: str
        :param dir_: str
                     folder from which all files are read
        :param info: list
                     [["year"], ["run"]]
        :param filter_: list
                        ["column_name", (lower_value_limit, upper_value_limit)]
        :param name_data_: list
                           1D list containing all histogramm enrty names of data len(file_list) (optional)
        :param name_sig_mc_: list
                             1D list containing all histogramm enrty names of signal processes len(file_list) (optional)
        :param name_bac_mc_: list
                             1D list containing all histogramm enrty names of background processes len(file_list) (optional)
        :param file_list_: list
                           1D list containing all files that contribute to the histogramm.
        :param kwargs: kwargs of fill_hist
        """
        if file_list_ is None: file_list_ = [os.path.join(dir_, item) for item in os.listdir(dir_)]
        
        years, runs_ = (["2012"], ["A-D"]) if info is None else (info[0], info[1])
        
        temp_d_ = "data_" if not [k for k in self.data if "data" in k] else [k for k in self.data if "data" in k][0]
        name_data_ = [temp_d_ for _ in file_list_] if name_data_ is None else name_data_
        
        temp_mc_sig_ = "mc_sig" if not [k for k in self.data if "mc_sig" in k] else [k for k in self.data if "mc_sig" in k][0]
        name_sig_mc_ = [temp_mc_sig_ for _ in file_list_] if name_sig_mc_ is None else name_sig_mc_
        
        temp_mc_bac_ = "mc_bac" if not [k for k in self.data if "mc_bac" in k] else [k for k in self.data if "mc_bac" in k][0]
        name_bac_mc_ = [temp_mc_bac_ for _ in file_list_] if name_bac_mc_ is None else name_bac_mc_
        
        for item, n_data_, n_sig_mc_, n_bac_mc_ in zip(file_list_, name_data_, name_sig_mc_, name_bac_mc_):
            for year in years:
                if year in item:
                    df_ = pd.read_csv(item, sep=";")
                    if not df_.empty:
                        
                        col__ = []
                        if isinstance(col_, str): col__ = [it for it in df_.columns if col_ in it or it in col_]
                        
                        if filter_ is not None: col__.append(filter_[0])
                        
                        array_to_fill = self.convert_column(df_[col__], filter_)
                        # if "CMS_Run" in item:
                        #     if runs_ == "A-D" and ("2012" in item and "2012" in years):
                        #         self.fill_hist(name=n_data_, array_of_interest=array_to_fill, **kwargs)
                        if "CMS_Run" in item:
                            for run in runs_:
                                if f"{year}{run}" in item or run == "A-D":
                                    self.fill_hist(name=n_data_, array_of_interest=array_to_fill, **kwargs)
                        
                        if "MC_20" in item:
                            if "_H_to" in item:
                                self.fill_hist(name=n_sig_mc_, array_of_interest=array_to_fill,
                                               info=[year, runs_, "H_ZZ"], **kwargs)
                            if "_H_to" not in item:
                                self.fill_hist(name=n_bac_mc_, array_of_interest=array_to_fill,
                                               info=[year, runs_, "ZZ_" + str(item).split("_")[-3]], **kwargs)
    
    def draw(self, pass_name, **kwargs):
        """
        Draws the histogram.
        If "data" is drawn, the kwargs as well as the
        pass_name should contain "data" in the first place.

        :param pass_name: list
                          similar to ["data", "mc_bac", "mc_sig"]
        :param kwargs: dict
                       dict of lists for matplotlib.
                       Similar to dict(alpha=[0.5, 0.5, 0.5], color=["black", "green", "red"]...)
        """
        
        def append_zeros(array_):
            return np.append([0], np.append(array_, 0))
        
        fig, ax = (kwargs["figure"], kwargs["ax"]) if "figure" in kwargs else plt.subplots(1, 1, figsize=(10, 6))
        with contextlib.suppress(KeyError):
            kwargs.pop("figure")
            kwargs.pop("ax")
        
        if "undefined" in self.data.keys():
            ax.fill_between(self.x_range, self.data["undefined"], step="mid", **kwargs)
            return fig, ax
        
        kwlist = [{key: val[i] for key, val in kwargs.items()} for i in range(max(map(len, kwargs.values())))]
        
        if "data" in self.data.keys():
            
            if np.count_nonzero(self.data["data"]) > 0:
                
                kwargs_data_ = {"color": "black", "marker": "o", "fmt": "o"}
                kwargs_data_.update(kwlist[0])
                
                pass_x, pass_y = np.array([]), np.array([])
                for i in range(len(self.x_range)):
                    if self.data["data"][i] != 0:
                        pass_x, pass_y = np.append(pass_x, self.x_range[i]), np.append(pass_y, self.data["data"][i])
                ax.errorbar(pass_x, pass_y, xerr=0, yerr=self.calc_errors_poisson_near_cont(pass_y), **kwargs_data_)
        
        if len(list(self.data.keys())) >= 2:
            plt_xr = np.append([self.x_range[0] - self.bin_width], np.append(self.x_range, self.x_range[-1] + self.bin_width))
            
            ax.fill_between(plt_xr, append_zeros(self.data[pass_name[1]]), step="mid", linewidth=0.0, **kwlist[1])
            if len(list(self.data.keys())) >= 3:
                temp_pileup = 0
                for i in range(2, len(pass_name)):
                    temp_pileup += append_zeros(self.data[pass_name[i - 1]])
                    ax.fill_between(plt_xr, append_zeros(self.data[pass_name[i]]) + temp_pileup, temp_pileup,
                                    step="mid", linewidth=0.0, **kwlist[i])
        ax.legend()
        
        return fig, ax
    
    def save_hist_plot(self, name):
        """
        Saves the created figure.
        
        :param name: str
        """
        plt.savefig(os.path.join(self.save_dir, name))
    
    def get_data_frame(self):
        """
        Creates and transfers a pd.DataFrame containing the reduced information of the histogram.7
        
        :return: pd.DataFrame
        """
        df_ = pd.DataFrame(self.x_range, columns=["x_range"])
        for name in list(self.data.keys()):
            df_[name] = self.data[name]
        return df_

