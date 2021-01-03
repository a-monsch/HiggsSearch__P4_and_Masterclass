# -*- coding: UTF-8 -*-

import contextlib
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from include.histogramm.HistHelper import HistHelper
from ..name_alias import get_alias

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

alias = get_alias()


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray


class Hist(HistHelper):
    """
    Class that creates a histogram and, if necessary, corrects and draws or saves it according to the conditions.
    """
    
    def __init__(self, bins, hist_range, signal_mass_mc=None, save_dir=".", info=None):
        self.__cross_sec = {"2011": {"ZZ_4mu": 66.09, "ZZ_4el": 66.09, "ZZ_2el2mu": 152, "H_ZZ": 5.7},
                            "2012": {"ZZ_4mu": 76.91, "ZZ_4el": 76.91, "ZZ_2el2mu": 176.7, "H_ZZ": 6.5}}
        self.__k_factor = {"2011": {"ZZ_4mu": 1.386, "ZZ_4el": 1.386, "ZZ_2el2mu": 1.386, "H_ZZ": 1.0},
                           "2012": {"ZZ_4mu": 1.386, "ZZ_4el": 1.386, "ZZ_2el2mu": 1.386, "H_ZZ": 1.0}}
        self.__lumen = {"2011": {"A": 4.7499}, "2012": {"B": np.longdouble(4.429375295985512733),
                                                        "C": np.longdouble(7.152728016920716286)}}
        
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
        
        self.data = {}
        
        self.info = info if info is not None else [["2012"], ["B", "C"]]
    
    def create_dir(self, dir_="./histogramms"):
        """
        Creates directory.

        :param dir_: str
        """
        if not os.path.exists(dir_):
            os.mkdir(dir_)
            self.save_dir = dir_
    
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
        if isinstance(run, str):
            run = [run]
        tot_luminosity = np.sum(lum for run_, lum in self.__lumen[year].items() if run_ in run)
        
        fac_ = self.__k_factor[year][process] * self.__cross_sec[year][process] / self.__event_num_mc[year][process]
        
        return tot_luminosity * fac_
    
    def fill(self, name, array_of_interest, info=None, get_raw=False):
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
    
    @staticmethod
    def _get_cutted_list_from_dir(dir_, years_, runs_):
        if all(any(it in filename for it in alias["data types"].aliases_of("run")) for filename in os.listdir(dir_)):
            return [os.path.join(dir_, item) for item in os.listdir(dir_) if
                    any((f"{year}{run}" in item) or (year in item and run == "A-D") for year in years_ for run in runs_)]
        
        if all(any(it in filename for it in alias["data types"].aliases_of("mc")) for filename in os.listdir(dir_)):
            return [os.path.join(dir_, item) for item in os.listdir(dir_) if any(year in item for year in years_)]
    
    def _get_name_from_data_dict(self, name):
        """
        Returns a predefined name if it exists in one of the keys in the given dict.
        If not, the searched name is returned.
        
        :param name: str
        :return: str
        """
        
        return name if not [k for k in self.data if name in k] else [k for k in self.data if name in k][0]
    
    def _get_name_list(self, name_in_dict, file_list, glob_name_):
        """
        Gives a list of the length of the file_list with the name_in_dict if glob_name is None.
        
        :param name_in_dict: str
        :param file_list: list
                          1D list of str
        :param glob_name_: str or list
        :return: list
        """
        return [self._get_name_from_data_dict(name=name_in_dict) for _ in file_list] if glob_name_ is None else glob_name_
    
    def fill_hist_from_dir(self, column, directory=None, file_list=None, info=None, filter_by=None,
                           name_data=None, name_mc_sig=None, name_mc_bac=None, lepton_number=None,
                           **kwargs):
        """
        Fills the histogram according to the used files in directory or file_list_ and
        saves all intermediate steps if get_raw is passed.
        See self.fill(...).


        :param column: str
        :param directory: str
                     folder from which all files are read
        :param info: list
                     [["year"], ["run"]]
        :param lepton_number: int or list containing the desired leptons
        :param filter_by: list
                        ["column_name", (lower_value_limit, upper_value_limit)]
        :param name_data: list
                           1D list containing all histogram entry names of data len(file_list) (optional)
                           optional: in case you want create different empty bins
        :param name_mc_sig: list
                             1D list containing all histogram entry names of signal processes len(file_list) (optional)
                             optional: in case you want create different empty bins
        :param name_mc_bac: list
                             1D list containing all histogram entry names of background processes len(file_list) (optional)
                             optional: in case you want create different empty bins
        :param file_list: list
                           1D list containing all files that contribute to the histogram.
        :param kwargs: kwargs of fill
        """
        
        years_, runs_ = self.info if info is None else (info[0], info[1])
        
        file_list = file_list if file_list is not None else Hist._get_cutted_list_from_dir(directory, years_, runs_)
        
        with contextlib.suppress(KeyError):
            if "channel" in kwargs and kwargs["channel"] is not None:
                if isinstance(kwargs["channel"], str):
                    _collect_channels = {it for it in alias["channels"].aliases_of(kwargs["channel"])}
                    file_list = [file for file in file_list if any(it in file for it in list(_collect_channels))]
                if isinstance(kwargs["channel"], list):
                    _collect_channels = {it for channel in kwargs["channel"] for it in alias["channels"].aliases_of(channel)}
                    file_list = [file for file in file_list if any(it in file for it in list(_collect_channels))]
            kwargs.pop("channel")
        
        for item, data_, mc_sig_, mc_bac_ in zip(file_list,
                                                 self._get_name_list("data", file_list, name_data),
                                                 self._get_name_list("mc_sig", file_list, name_mc_sig),
                                                 self._get_name_list("mc_bac", file_list, name_mc_bac)):
            df_ = pd.read_csv(item, sep=";")
            if not df_.empty:
                col__ = [it for it in df_.columns if column in it or it in column] if isinstance(column, str) else column
                col__.append(filter_by[0]) if filter_by is not None else None
                array_to_fill = self.convert_column(df_[col__], filter_by=filter_by, lepton_number=lepton_number)
                if any(it in os.path.split(item)[1] for it in alias["data types"].aliases_of("run")):
                    self.fill(name=data_, array_of_interest=array_to_fill, **kwargs)
                if any(it in os.path.split(item)[1] for it in alias["data types"].aliases_of("mc")):
                    year = [y for y in years_ if y in item][0]
                    if any(it in os.path.split(item)[1] for it in alias["data types"].aliases_of("mc_sig")):
                        self.fill(name=mc_sig_, array_of_interest=array_to_fill,
                                  info=[year, runs_, "H_ZZ"], **kwargs)
                    if not any(it in os.path.split(item)[1] for it in alias["data types"].aliases_of("mc_sig")):
                        self.fill(name=mc_bac_, array_of_interest=array_to_fill,
                                  info=[year, runs_,
                                        f"ZZ_{'4mu' if '4mu' in item else ('4el' if '4el' in item else '2el2mu')}"],
                                  **kwargs)
    
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
        idx = 0
        
        if any(it in pass_name for it in alias["data types"].aliases_of("data")):
            if np.count_nonzero(self.data["data"]) > 0:
                
                kwargs_data_ = {"color": "black", "marker": "o", "fmt": "o"}
                kwargs_data_.update(kwlist[idx])
                
                pass_x, pass_y = np.array([]), np.array([])
                for i in range(len(self.x_range)):
                    if self.data["data"][i] != 0:
                        pass_x, pass_y = np.append(pass_x, self.x_range[i]), np.append(pass_y, self.data["data"][i])
                ax.errorbar(pass_x, pass_y, xerr=0, yerr=self.calc_errors_poisson_near_cont(pass_y), **kwargs_data_)
                idx += 1
        
        if any(it in pass_name for it in alias["data types"].aliases_of("mc_bac")):
            plt_xr = np.append([self.x_range[0] - self.bin_width], np.append(self.x_range, self.x_range[-1] + self.bin_width))
            ax.fill_between(plt_xr, append_zeros(self.data["mc_bac"]), step="mid", linewidth=0.0, **kwlist[idx])
            idx += 1
        
        if any(it in pass_name for it in alias["data types"].aliases_of("mc_sig")):
            temp_pileup = 0
            plt_xr = np.append([self.x_range[0] - self.bin_width], np.append(self.x_range, self.x_range[-1] + self.bin_width))
            if any(it in pass_name for it in alias["data types"].aliases_of("mc_bac")):
                temp_pileup += append_zeros(self.data["mc_bac"])
            ax.fill_between(plt_xr, append_zeros(self.data["mc_sig"]) + temp_pileup, temp_pileup,
                            step="mid", linewidth=0.0, **kwlist[idx])
        
        ax.legend()
        
        return fig, ax
    
    def save(self, name):
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
