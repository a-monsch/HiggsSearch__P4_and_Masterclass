# -*- coding: UTF-8 -*-
from copy import copy

import numpy as np


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray


class ProcessingBasicAwk(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def event_based_filter(awk_array, filter_on):
        return awk_array[filter_on]
    
    @staticmethod
    def item_based_filter(awk_array, filter_on, look_for):
        
        _awk_array = copy(awk_array)
        
        for name in awk_array.fields:
            if look_for != "both":
                if "muon_" in name or "electron_" in name:
                    _awk_array[name] = awk_array[name][filter_on]
            if look_for == "both":
                if "muon_" in name:
                    _awk_array[name] = awk_array[name][filter_on[0]]
                if "electron_" in name:
                    _awk_array[name] = awk_array[name][filter_on[1]]
        return _awk_array
    
    @staticmethod
    def minimum_item_count_filter(awk_array, look_for, num=4):
        
        if look_for != "both":
            _cols = [item for item in awk_array.fields if "muon_" in item or "electron_" in item]
            _len = [len(awk_array[_cols[0]][i]) >= num for i in range(len(awk_array[_cols[0]]))]
            
            awk_array = ProcessingBasicAwk.event_based_filter(awk_array=awk_array, filter_on=_len)
            return awk_array
        
        if look_for == "both":
            _muon_cols = [item for item in awk_array.fields if "muon_" in item]
            _electron_cols = [item for item in awk_array.fields if "electron_" in item]
            _len_muon = np.array([len(item) >= int(num / 2) for item in awk_array[_muon_cols[0]]])
            _len_electron = np.array([len(item) >= int(num / 2) for item in awk_array[_electron_cols[0]]])
            _len = _len_muon & _len_electron
            
            awk_array = ProcessingBasicAwk.event_based_filter(awk_array=awk_array, filter_on=_len)
            return awk_array
