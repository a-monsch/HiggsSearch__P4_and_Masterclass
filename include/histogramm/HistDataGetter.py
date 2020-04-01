import os

import numpy as np
import pandas as pd

from .Hist import Hist


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray


class HistDataGetter(object):
    """
    class to generate data for the structure for the fits and the subsequent
    statistical analysis (using the Hist class).
    """
    
    def __init__(self, bins=15, hist_range=(106, 151), info=None,
                 ru_dir="./data/ru_aftH", mc_dir="./data/mc_aftH"):
        self.bins = bins
        self.hist_range = hist_range
        self.ru_dir = ru_dir
        self.mc_dir = mc_dir
        
        self.info = [["2012"], ["A-D"]] if info is None else info
    
    def get_mc_raw(self, column_name="mass_4l", tag="Background".lower()):
        """
        Function to record the data of individual MC simulations separately by channel.
        The scaling factor, the unscaled histogram and the scaled histogram are calculated.

        :param column_name: str
                            "mass_4l" for the four lepton analysis. Might be changed.
        :param tag: str
                    "background" or "signal"
        :return: hist object
        """
        hist = Hist(bins=self.bins, hist_range=self.hist_range)
        raw_file_list = [os.path.join(self.mc_dir, it) for it in os.listdir(self.mc_dir)]
        
        if tag.lower() == "background":
            files_ = [it for it in raw_file_list if "_H_to_" not in it]
            names_bac_mc_ = [f"mc_track_ZZ_{'4mu' if '4mu' in it else ('4el' if '4el' in it else '2el2mu')}_{self.info[0][0]}" for it in files_]
            hist.fill_hist_from_dir(col_=column_name, dir_=None, info=self.info, name_bac_mc_=names_bac_mc_, file_list_=files_)
        
        if tag.lower() == "signal":
            files_ = [it for it in raw_file_list if "_H_to_" in it]
            names_sig_mc_ = [f"mc_track_H_ZZ_{'4mu' if '4mu' in it else ('4el' if '4el' in it else '2el2mu')}_{self.info[0][0]}" for it in files_]
            hist.fill_hist_from_dir(col_=column_name, dir_=None, info=self.info, name_sig_mc_=names_sig_mc_, file_list_=files_)
        
        return hist
    
    def get_mc_com(self):
        """
        Function to obtain a histogram of the combined channels, already scaled to the appropriate size.

        :return: hist object
        """
        hist = Hist(bins=self.bins, hist_range=self.hist_range)
        hist.set_empty_bin(["data", "mc_bac", "mc_sig"])
        
        hist.fill_hist_from_dir(col_="mass_4l", dir_=self.mc_dir, info=self.info)
        hist.fill_hist_from_dir(col_="mass_4l", dir_=self.ru_dir, info=self.info)
        return hist
    
    def get_data_raw(self, column_name="mass_4l"):
        """
        Fetches the individual measuring points of a specific column in a selected range self.hist_range.

        Note: This does not have to be stored in a hist object, but it has
              the advantage that additional sizes are stored with it.
              Can possibly be rewritten.

        :param column_name: str
                            "mass_4l" for the four lepton analysis. Might be changed.
        :return: hist obj
        """
        years, runs_ = self.info[0], self.info[1]
        hist = Hist(bins=self.bins, hist_range=self.hist_range)
        
        files_ = [os.path.join(self.ru_dir, n) for n in os.listdir(self.ru_dir) if
                  any(y in n for y in years if any(f"{y}{r}" in n or r == "A-D" for r in runs_))]
        
        to_pass_array = np.array([])
        for item in files_:
            df_ = pd.read_csv(item, sep=";")
            if not df_.empty:
                to_pass_array = np.append(to_pass_array, np.array(df_[column_name].values, dtype=float))
        
        filter_ = (to_pass_array <= self.hist_range[1]) & (self.hist_range[0] <= to_pass_array)
        to_pass_array = to_pass_array[filter_]
        hist.data = to_pass_array
        return hist
