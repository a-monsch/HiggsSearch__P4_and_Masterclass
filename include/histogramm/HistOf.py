# -*- coding: UTF-8 -*-
from .Hist import Hist
from ..name_alias import get_alias

alias = get_alias()


class HistOf(object):
    """
    Class to draw histograms after the completed filter and reconstruction process.
    """
    
    _legend_dict = {"background": {"EN": "Background", "DE": "Untergrund"},
                    "signal": {"EN": "Signal", "DE": "Signal"},
                    "measurement": {"EN": "Measurement", "DE": "Messung"}}
    
    def __init__(self, mc_dir="../data/mc_afterHreco", ru_dir="../data/ru_afterHreco", info=None, language="EN", channel=None):
        self.ru_dir = ru_dir
        self.mc_dir = mc_dir
        self.info = info if info is not None else [["2012"], ["B", "C"]]
        self.lang = language
        self.channel = channel
    
    def variable(self, variable, bins, hist_range, specific=None, filter_by=None, info=None, lepton_number=None, channel=None):
        """
        Creates the histogram of the corresponding variable under the
        optional application of a filter_by. Using "data", "mc_sig" and "mc_bac".

        :param variable: str
        :param bins: int
        :param lepton_number: int or list containing desired leptons
        :param channel: str or list of str
                        "4mu", "4el", "2el2mu"
        :param specific: str or list containing desired quantities ["data", "mc_sig", "mc_bac"]
        :param hist_range: tuple
                           (lower_value_limit, upper_value_limit)
        :param filter_by: list
                        ["variable_name", (lower_value_limit, upper_value_limit)]
        :param info: list
                      optional, see Hist.fill
        """
        info = self.info if info is None else info
        
        hist = Hist(bins=bins, hist_range=hist_range)
        hist.set_empty_bin(name=["data", "mc_bac", "mc_sig"])
        
        hist.fill_hist_from_dir(column=variable, directory=self.ru_dir, info=info, filter_by=filter_by,
                                lepton_number=lepton_number, channel=channel or self.channel)
        hist.fill_hist_from_dir(column=variable, directory=self.mc_dir, info=info, filter_by=filter_by,
                                lepton_number=lepton_number, channel=channel or self.channel)
        
        ax = hist.draw(**self._get_specific_parts(specific=specific))
        
        return ax, hist
    
    def _get_specific_parts(self, specific):
        
        if specific is None:
            return self._get_specific_parts(specific=["data", "mc_bac", "mc_sig"])
        
        if isinstance(specific, str):
            specific = [specific]
        
        _name, _alpha, _color, _label = [], [], [], []
        
        if any(it in specific for it in alias["data types"].aliases_of("data")):
            _name.append("data")
            _alpha.append(0.75)
            _color.append("black")
            _label.append(HistOf._legend_dict["measurement"][self.lang])
        
        if any(it in specific for it in alias["data types"].aliases_of("mc_bac")):
            _name.append("mc_bac")
            _alpha.append(0.75)
            _color.append("royalblue")
            _label.append(HistOf._legend_dict["background"][self.lang])
        
        if any(it in specific for it in alias["data types"].aliases_of("mc_sig")):
            _name.append("mc_sig")
            _alpha.append(0.75)
            _color.append("orangered")
            _label.append(HistOf._legend_dict["signal"][self.lang])
        
        return {"pass_name": _name, "alpha": _alpha, "color": _color, "label": _label}
