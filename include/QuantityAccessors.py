from typing import List

import numpy as np
import pandas as pd
from uproot3_methods.classes.TLorentzVector import TLorentzVector

from include.Particle import Lepton


# exemplary usage:
# (df is a DataFrame, df.leptons is a Column of lists of leptons, z1 TLorentzVector)
#
# df.leptons.quantity.pt -> pd.Series of all lepton pt
# df.leptons.quantity(flavour="m", lep_num=[0]) -> pd.Series of first muon in events
# df.z1.quantity.mass -> pd.Series of z1 mass of different events
#
# connection with present Series methods/attributes
#
# df.leptons.quantity.pt.hist(bins=10, range=(0, 100))
# df.z1.quantity.mass.max()
# and so on...


@pd.api.extensions.register_series_accessor("quantity")
class ParticleQuantitySeriesAccessor(object):
    _connected_obj = (Lepton, TLorentzVector)
    _convert_attributes = {"px": "x", "py": "y", "pz": "z",
                           "pseudorapidity": "eta", "m": "mass"}
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def __call__(self, flavour: str = None, lep_num: List[int] = None):
        _obj = self._obj
        if flavour:  # considering only leptons with a specific flavour
            _obj = pd.Series([np.array([lep for lep in ev if lep.flavour == flavour]) for ev in _obj])
        if lep_num:  # first ([0]), first and second ([0, 1]), first and third ([0, 2])...
            _list = []
            for it in _obj:
                try:
                    _list.append(np.array(it[lep_num]))
                except IndexError:  # if trying to get leptons that are not in a event...
                    _idx = []
                    for idx in lep_num:
                        try:
                            _ = it[idx]
                            _idx.append(idx)
                        except IndexError:
                            pass
                    _list.append(np.array(it[_idx]))  # ...getting only present leptons
            _obj = pd.Series(_list)
        return self.__class__(_obj)  # proceeding with considered leptons
    
    def __getattribute__(self, item):
        try:  # getting Attributes of Series as .hist(), .max(), .values ...
            return object.__getattribute__(self, item)
        except AttributeError:
            item = self._convert_attributes[item] if item in self._convert_attributes.keys() else item
            try:  # flatten if getting list of lists of leptons
                _flat_obj = np.concatenate(self._obj.values)
            except (ValueError, KeyError):
                _flat_obj = self._obj.values
            _attrs = []
            for _obj in _flat_obj:
                try:
                    _attr = getattr(_obj, item)
                except AttributeError:  # if having filler obj != _connected_obj in list
                    _attr = np.nan
                _attrs.append(_attr)
            return pd.Series(np.array(_attrs), name=item)
