import math

import numpy as np

from include.RandomHelper import ToSortHelper as TSH
from include.histogramm.Hist import Hist


class _WidgetHistLoader(object):
    
    def __init__(self, bins=37, hist_range=(70, 181),
                 mc_dir="./data/for_widgets/mc_aftH",
                 mc_other_dir="./data/for_widgets/other_mc/dXXX/mc_aftH", info=None):
        self.mc_other_dir = mc_other_dir
        self.mc_dir_ = mc_dir
        self.hist_range = hist_range
        self.bins_ = bins
        self.on_pseudo_data = True
        
        self.info = info if info is not None else [["2012"], ["B", "C"]]
        
        self._histograms = {}
        self.other_mc_sig_num_list = [115, 120, 122, 124, 128, 130, 135, 140, 145, 150]
        self.mc_sig_name_list = [115, 120, 122, 124, 125, 128, 130, 135, 140, 145, 150]
    
    def load_hists(self):
        h = Hist(bins=self.bins_, hist_range=self.hist_range)
        h.set_empty_bin(["data", "mc_bac"])
        h.fill_hist_from_dir(column="mass_4l", directory=self.mc_dir_,
                             info=self.info)
        
        self.y_plot_limits = (0, math.ceil(np.amax(h.data["mc_bac"])))
        self._histograms["mc_bac"] = h
        
        h = Hist(bins=self.bins_, hist_range=self.hist_range)
        h.set_empty_bin(["data", "mc_bac", "mc_sig"])
        h.fill_hist_from_dir(column="mass_4l", directory=self.mc_dir_,
                             info=self.info)
        
        self._histograms["mc_bac_sig_125"] = h
        
        for i, num in enumerate(self.other_mc_sig_num_list):
            file_list = TSH.mixed_file_list(
                TSH.get_other_mc_dir(num, dir_=self.mc_other_dir),
                main_dir_=self.mc_dir_)
            h = Hist(bins=self.bins_, hist_range=self.hist_range,
                     signal_mass_mc=None if self.on_pseudo_data else num)
            h.set_empty_bin(["data", "mc_bac", "mc_sig"])
            h.fill_hist_from_dir(column="mass_4l", directory=None,
                                 info=self.info, file_list=file_list)
            self._histograms[f"mc_bac_sig_{num}"] = h
    
    @property
    def histograms(self):
        self.load_hists()
        return self._histograms


def get_histograms(*args, **kwargs):
    LD = _WidgetHistLoader(*args, **kwargs)
    th = LD.histograms
    return th


if __name__ == "__main__":
    pass
