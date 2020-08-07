# -*- coding: UTF-8 -*-
from .Hist import Hist


class HistOf(object):
    """
    Class to draw histograms after the completed filter and reconstruction process.
    """
    
    _legend_dict = {"background": {"EN": "Background", "DE": "Untergrund"},
                    "signal": {"EN": "Signal", "DE": "Signal"},
                    "measurement": {"EN": "Measurement", "DE": "Messung"}}
    
    def __init__(self, mc_dir="./data/mc_aftH", ru_dir="./data/ru_aftH", info=None, language="EN"):
        self.ru_dir = ru_dir
        self.mc_dir = mc_dir
        self.info = info
        self.lang = language
    
    def variable(self, variable, bins, hist_range, filter_=None, info_=None):
        """
        Creates the histogram of the corresponding variable under the
        optional application of a filter. Using "data", "mc_sig" and "mc_bac".

        :param variable: str
        :param bins: int
        :param hist_range: tuple
                           (lower_value_limit, upper_value_limit)
        :param filter_: list
                        ["variable_name", (lower_value_limit, upper_value_limit)]
        :param info_: list
                      optional, see Hist.fill_hist
        """
        info_ = self.info if info_ is None else info_
        
        hist = Hist(bins=bins, hist_range=hist_range)
        hist.set_empty_bin(name=["data", "mc_bac", "mc_sig"])
        
        hist.fill_hist_from_dir(col_=variable, dir_=self.ru_dir, info=info_, filter_=filter_)
        hist.fill_hist_from_dir(col_=variable, dir_=self.mc_dir, info=info_, filter_=filter_)
        
        ax = hist.draw(pass_name=["data", "mc_bac", "mc_sig"],
                       alpha=[0.75, 0.75, 0.75], color=["black", "royalblue", "orangered"],
                       label=[HistOf._legend_dict["measurement"][self.lang], 
                              HistOf._legend_dict["background"][self.lang],
                              HistOf._legend_dict["signal"][self.lang]])
        
        return ax, hist
