from dataclasses import dataclass

import numpy as np
from uproot3_methods.classes.TLorentzVector import TLorentzVector


@dataclass
class Lepton(TLorentzVector):
    __slots__ = "lv", "charge", "flavour", "relpfiso", "dxy", "dz", "sip3d"
    _convert_attributes = {"px": "x", "py": "y", "pz": "z",
                           "pseudorapidity": "eta"}
    
    lv: TLorentzVector  # Lorentzvector containing (px, py, pz, E)
    charge: np.int32  # +1 or -1
    flavour: np.dtype('<U1')  # "m" for muon or "e" for electron
    relpfiso: np.float64  # relative Isolation of a Lepton
    dxy: np.float64  # impact parameter dxy
    dz: np.float64  # impact parameter dz
    sip3d: np.float64  # impact parameter sip3d
    
    def __getattribute__(self, item):
        try:  # accessing Lepton (or Class) specific Attributes
            return object.__getattribute__(self, item)
        except AttributeError:  # accessing Attributes of TLorentzVector
            item = self._convert_attributes[item] if item in self._convert_attributes.keys() else item
            return getattr(self.lv, item)
