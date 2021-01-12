# -*- coding: UTF-8 -*-

import gc

import awkward1 as awk
import numpy as np

from .CalcAndAllowerInit import FilterInit, CalcInit
from .ProcessingBasicAwk import ProcessingBasicAwk


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray

npawk = lambda x: np.array(awk.to_list(x))


class FilterStr(ProcessingBasicAwk):
    """
    Class that provides certain filters for data reduction.
    """
    calc_instance = CalcInit
    filter_instance = FilterInit
    
    def __init__(self, look_for):
        super().__init__()
        self.look_for = look_for
    
    @classmethod
    def set_instance(cls, **kwargs):
        """
        Sets the class instances used in the following.

        :param kwargs: class instances
        """
        if any("calc" in it for it in kwargs.keys()):
            cls.calc_instance = kwargs["calc_instance"]
        if any("filter" in it for it in kwargs.keys()):
            cls.filter_instance = kwargs["filter_instance"]
    
    # Filter No 1
    def lepton_detector_classification(self, awk_array):
        """
        Filter checks the classification of the leptons.
        
        workflow:
            - search for electron_type or muon_type in a event
            - retrieve a boolean array (FilterInit.lepton_type)
            - use the boolean array to remove undesired leptons from event
            - return reduced event if number_leptons >= 4 or discard event else
                (or in 2e2mu - channel : number_electrons >=2 and number_muons >=2)

        :return: pd.Series
        """
        
        _func = FilterStr.filter_instance.lepton_type
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            accept_array = _func(awk_array[f"{self.look_for}_type"], self.look_for)
            
            awk_array = self.item_based_filter(awk_array, accept_array, self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, self.look_for)
            
            del accept_array
            gc.collect()
            
            return awk_array
        
        if self.look_for == "both":
            
            accept_array_mu = _func(awk_array["muon_type"], "muon")
            accept_array_el = _func(awk_array["electron_type"], "electron")
            
            awk_array = self.item_based_filter(awk_array, [accept_array_mu, accept_array_el], self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, self.look_for)
            
            del accept_array_el, accept_array_mu
            gc.collect()
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 2
    def electric_charge(self, awk_array):
        """
        Filter that checks whether an electrically neutral charge
        combination can be formed from the leptons contained in the event.
        
        workflow:
            - search for electron_charge or muon_charge in a event
            - retrieve a user defined boolean (CalcStudent.combined_charge)
            - return unchanged event if (CalcStudent.combined_charge) returns True or discard else
        
        :return: pd.Series
        """
        
        _func = FilterStr.filter_instance.combined_charge
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            accept_charge = awk.from_iter([_func(it, combine_num=4) for it in awk_array[f"{self.look_for}_charge"]])
            
            awk_array = self.event_based_filter(awk_array, filter_on=accept_charge)
            
            del accept_charge
            gc.collect()
            
            return awk_array
        
        if self.look_for == "both":
            
            accept_charge_mu = awk.from_iter([_func(it, combine_num=2) for it in awk_array["muon_charge"]])
            accept_charge_el = awk.from_iter([_func(it, combine_num=2) for it in awk_array["electron_charge"]])
            
            accept_charge = awk.from_iter(list(np.array(accept_charge_el) & np.array(accept_charge_mu)))
            
            awk_array = self.event_based_filter(awk_array, filter_on=accept_charge)
            
            del accept_charge, accept_charge_mu, accept_charge_el
            gc.collect()
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 3
    def pt_min(self, awk_array):
        """
        Filter out leptons whose transverse momentum is smaller than the minimum allowed by filter.
        Also adds transverse momentum.
        
        workflow:
            - search for (px, py) or pt in a event
            - calculates pt if not done so with a user defined function (Calc_Start.pt)
            - adds pt to quantities if not present
            - retrieve a user defined boolean array (Filter_Start.pt_min)
            - use the boolean array to remove undesired leptons from event
            - return reduced event if number_leptons >= 4 or discard event else
                (or in 2e2mu - channel : number_electrons >=2 and number_muons >=2)
                
        :return: pd.Series
        """
        
        _calc_func = FilterStr.calc_instance.pt
        _filter_func = FilterStr.filter_instance.pt_min
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            
            if f"{self.look_for}_pt" not in awk_array.fields:
                awk_array[f"{self.look_for}_pt"] = _calc_func(awk_array[f"{self.look_for}_px"],
                                                              awk_array[f"{self.look_for}_py"])
            
            accept_array = _filter_func(awk_array[f"{self.look_for}_pt"], self.look_for)
            
            awk_array = self.item_based_filter(awk_array, accept_array, self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, self.look_for)
            
            del accept_array
            gc.collect()
            
            return awk_array
        
        if self.look_for == "both":
            
            if f"muon_pt" not in awk_array.fields:
                awk_array[f"muon_pt"] = _calc_func(awk_array[f"muon_px"],
                                                   awk_array[f"muon_py"])
            
            if f"electron_pt" not in awk_array.fields:
                awk_array[f"electron_pt"] = _calc_func(awk_array[f"electron_px"],
                                                       awk_array[f"electron_py"])
            
            accept_array_mu = _filter_func(awk_array[f"muon_pt"], "muon")
            accept_array_el = _filter_func(awk_array[f"electron_pt"], "electron")
            
            awk_array = self.item_based_filter(awk_array, [accept_array_mu, accept_array_el], self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, look_for=self.look_for)
            
            del accept_array_el, accept_array_mu
            gc.collect()
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 4
    def pseudorapidity(self, awk_array):
        """
        Filter leptons based on the calculated pseudorapidity. Also adds the pseudorapidity.
        
        workflow:
            - search for px, py, pz in a event
            - calculates pseudorapidity with a user defined function (CalcStudent.pseudorapidity)
            - adds pseudorapidity to quantities
            - retrieve a user defined boolean array (FilterStudent.pseudorapidity)
            - use the boolean array to remove undesired leptons from event
            - return reduced event if number_leptons >= 4 or discard event else
                (or in 2e2mu - channel : number_electrons >=2 and number_muons >=2)
        
        :return: pd.Series
        """
        
        _calc_func = FilterStr.calc_instance.pseudorapidity
        _filter_func = FilterStr.filter_instance.pseudorapidity
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            
            if f"{self.look_for}_pseudorapidity" not in awk_array.fields:
                awk_array[f"{self.look_for}_pseudorapidity"] = _calc_func(awk_array[f"{self.look_for}_px"],
                                                                          awk_array[f"{self.look_for}_py"],
                                                                          awk_array[f"{self.look_for}_pz"],
                                                                          energy=awk_array[f"{self.look_for}_energy"])
            
            accept_array = _filter_func(awk_array[f"{self.look_for}_pseudorapidity"], self.look_for)
            
            awk_array = self.item_based_filter(awk_array, accept_array, self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, self.look_for)
            
            del accept_array
            gc.collect()
            
            return awk_array
        
        if self.look_for == "both":
            
            if "muon_pseudorapidity" not in awk_array.fields:
                awk_array["muon_pseudorapidity"] = _calc_func(awk_array[f"muon_px"],
                                                              awk_array[f"muon_py"],
                                                              awk_array[f"muon_pz"],
                                                              energy=awk_array[f"muon_energy"])
            
            if "electron_pseudorapidity" not in awk_array.fields:
                awk_array["electron_pseudorapidity"] = _calc_func(awk_array[f"electron_px"],
                                                                  awk_array[f"electron_py"],
                                                                  awk_array[f"electron_pz"],
                                                                  energy=awk_array[f"electron_energy"])
            
            accept_array_mu = _filter_func(awk_array["muon_pseudorapidity"], "muon")
            
            accept_array_el = _filter_func(awk_array["electron_pseudorapidity"], "electron")
            
            awk_array = self.item_based_filter(awk_array, [accept_array_mu, accept_array_el], self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, look_for=self.look_for)
            
            del accept_array_el, accept_array_mu
            gc.collect()
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 5
    def misshit(self, awk_array):
        """
        Filters out electrons from an event that have an insufficient number of missing hits (>1).
        
        workflow:
            - search for electron_misshits in a event
            - retrieve a boolean array (FilterInit.misshits)
            - use the boolean array to remove undesired leptons from event
            - return reduced event if number_leptons >= 4 or discard event else
                (or in 2e2mu - channel : number_electrons >=2 and number_muons >=2)
        
        :return: pd.Series
        """
        
        _func = FilterStr.filter_instance.misshits
        
        if self.look_for == "electron":
            accept_array = _func(awk_array["electron_misshits"])
            
            awk_array = self.item_based_filter(awk_array, accept_array, self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, self.look_for)
            
            del accept_array
            gc.collect()
            
            return awk_array
        
        if self.look_for == "both":
            
            accept_array = _func(awk_array["electron_misshits"])
            
            awk_array = self.item_based_filter(awk_array, [True, accept_array], self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, self.look_for)
            
            del accept_array
            gc.collect()
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 6
    def relative_isolation(self, awk_array):
        """
        Filter leptons based on relative isolation.
        
        workflow:
            - search for relPFIso in a event
            - retrieve a user defined boolean array (FilterStudent.relative_isolation)
            - use the boolean array to remove undesired leptons from event
            - return reduced event if number_leptons >= 4 or discard event else
                (or in 2e2mu - channel : number_electrons >=2 and number_muons >=2)
        
        :return: pd.Series
        """
        
        _func = FilterStr.filter_instance.relative_isolation
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            accept_array = _func(awk_array[f"{self.look_for}_relPFIso"])
            
            awk_array = self.item_based_filter(awk_array, accept_array, self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, look_for=self.look_for)
            
            del accept_array
            gc.collect()
            
            return awk_array
        
        if self.look_for == "both":
            
            accept_array_mu = _func(awk_array[f"muon_relPFIso"])
            accept_array_el = _func(awk_array[f"electron_relPFIso"])
            
            awk_array = self.item_based_filter(awk_array, [accept_array_mu, accept_array_el], self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, look_for=self.look_for)
            
            del accept_array_el, accept_array_mu
            gc.collect()
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 7
    def impact_parameter(self, awk_array):
        """
        Filter leptons based on the impact parameter.
        
        workflow:
            - search for SIP3d, dxy and dz in a event
            - retrieve a user defined boolean array (FilterStudent.impact_parameter)
            - use the boolean array to remove undesired leptons from event
            - return reduced event if number_leptons >= 4 or discard event else
                (or in 2e2mu - channel : number_electrons >=2 and number_muons >=2)
        
        :return: pd.Series
        """
        
        _func = FilterStr.filter_instance.impact_parameter
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            accept_array = _func(awk_array[f"{self.look_for}_SIP3d"], awk_array[f"{self.look_for}_dxy"],
                                 awk_array[f"{self.look_for}_dz"])
            
            awk_array = self.item_based_filter(awk_array, accept_array, self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, self.look_for)
            
            del accept_array
            gc.collect()
            
            return awk_array
        
        if self.look_for == "both":
            accept_array_mu = _func(awk_array["muon_SIP3d"], awk_array["muon_dxy"], awk_array["muon_dz"])
            accept_array_el = _func(awk_array["electron_SIP3d"], awk_array["electron_dxy"], awk_array["electron_dz"])
            
            awk_array = self.item_based_filter(awk_array, [accept_array_mu, accept_array_el], self.look_for)
            awk_array = self.minimum_item_count_filter(awk_array, self.look_for)
            
            del accept_array_el, accept_array_mu
            gc.collect()
            
            return awk_array
    
    # Filter No 8
    def pt_exact(self, awk_array):
        """
        Filter events based on the exact required transverse momentum of each lepton in an event.
        
        workflow:
            - search for (px, py) or pt in a event
            - calculates pt if not done so with a user defined function (Calc_Start.pt)
            - adds pt to quantities if not present
            - retrieve a user defined boolean (FilterStudent.pt_exact)
            - return unchanged event if (FilterStudent.pt_exact) returns True or discard else
        
        
        :return: pd.Series
        """
        
        _calc_func = FilterStr.calc_instance.pt
        _filter_func = FilterStr.filter_instance.pt_exact
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            
            if f"{self.look_for}_pt" not in awk_array.fields:
                awk_array[f"{self.look_for}_pt"] = _calc_func(awk_array[f"{self.look_for}_px"],
                                                              awk_array[f"{self.look_for}_py"])
            
            accept_value = awk.from_iter([_filter_func(pt, self.look_for) for pt in awk_array[f"{self.look_for}_pt"]])
            
            awk_array = self.event_based_filter(awk_array, accept_value)
            
            del accept_value
            gc.collect()
            
            return awk_array
        
        if self.look_for == "both":
            if f"muon_pt" not in awk_array.fields:
                awk_array[f"muon_pt"] = _calc_func(awk_array[f"muon_px"],
                                                   awk_array[f"muon_py"])
            
            if f"electron_pt" not in awk_array.fields:
                awk_array[f"electron_pt"] = _calc_func(awk_array[f"electron_px"],
                                                       awk_array[f"electron_py"])
            
            accept_value = awk.from_iter([_filter_func(p_t=(npawk(pt_mu), npawk(pt_el)), lepton_type=self.look_for)
                                          for pt_mu, pt_el in zip(awk_array["muon_pt"], awk_array["electron_pt"])])
            
            awk_array = self.event_based_filter(awk_array, accept_value)
            
            del accept_value
            gc.collect()
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 9
    def two_lepton_mass(self, awk_array):
        """
        Quick Filter that checks if the minimum required two lepton invariant masses can be reconstructed.
        This filter do not make a complete reconstruction where all necessary variables and those further
        restrictions are taken into account. They only serve for a short estimate of the values whether a
        mass could be reconstructed at all in this range or not.
        
        workflow:
            - search for variables in a event that are required for mass calculation
            - calculates possible masses with a not user defined function (CalcInit.possible_invariant_masses)
            - retrieve a boolean (FilterInit.invariant_mass) if a two lepton mass can be reconstructed in [4, 70] GeV
            - return unchanged event if (FilterInit.invariant_mass) returns True or discard else
        
        :return: pd.Series
        """
        
        _calc_func = FilterStr.calc_instance.possible_invariant_masses
        _filter_func = FilterStr.filter_instance.invariant_mass
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            inv_2l_m = _calc_func(awk_array=awk_array, lepton_type=self.look_for, number_of_leptons=2)
            accept_value = [_filter_func(mass, number_of_leptons=2, lepton_type=self.look_for) for mass in inv_2l_m]
            
            awk_array = self.event_based_filter(awk_array, accept_value)
            
            del inv_2l_m, accept_value
            gc.collect()
            
            return awk_array
        
        if self.look_for == "both":
            
            inv_2l_m_mu = _calc_func(awk_array=awk_array, lepton_type="muon", number_of_leptons=2)
            inv_2l_m_el = _calc_func(awk_array=awk_array, lepton_type="electron", number_of_leptons=2)
            
            accept_value_mu = [_filter_func(mass[0], number_of_leptons=2, lepton_type=self.look_for) for mass in inv_2l_m_mu]
            accept_value_el = [_filter_func(mass, number_of_leptons=2, lepton_type=self.look_for) for mass in inv_2l_m_el]
            
            awk_array = self.event_based_filter(awk_array, awk.from_iter(accept_value_el) | awk.from_iter(accept_value_mu))
            
            del inv_2l_m_mu, inv_2l_m_el, accept_value_mu, accept_value_el
            gc.collect()
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 10
    def four_lepton_mass(self, awk_array):
        """
        Quick Filter that checks if the minimum required four lepton invariant masses can be reconstructed.
        This filter do not make a complete reconstruction where all necessary variables and those further
        restrictions are taken into account. They only serve for a short estimate of the values whether a
        mass could be reconstructed at all in this range or not.
        
        workflow:
            - search for variables in a event that are required for mass calculation
            - calculates possible masses with a not user defined function (CalcInit.possible_invariant_masses)
            - retrieve a boolean (FilterInit.invariant_mass) if a four lepton mass can be reconstructed in [4, 70] GeV
            - return unchanged event if (FilterInit.invariant_mass) returns True or discard else
        
        :return: pd.Series
        """
        
        _calc_func = FilterStr.calc_instance.possible_invariant_masses
        _filter_func = FilterStr.filter_instance.invariant_mass
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            inv_4l_m = _calc_func(awk_array=awk_array, lepton_type=self.look_for, number_of_leptons=4)
            accept_value = [_filter_func(mass, number_of_leptons=4, lepton_type=self.look_for) for mass in inv_4l_m]
            
            awk_array = self.event_based_filter(awk_array, accept_value)
            
            del inv_4l_m, accept_value
            gc.collect()
            
            return awk_array
        
        if self.look_for == "both":
            
            inv_4l_m = _calc_func(awk_array=awk_array, lepton_type=self.look_for, number_of_leptons=4)
            accept_value = [_filter_func(mass, number_of_leptons=4, lepton_type=self.look_for) for mass in inv_4l_m]
            
            awk_array = self.event_based_filter(awk_array, accept_value)
            
            del inv_4l_m, accept_value
            gc.collect()
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")


class AddVariable(ProcessingBasicAwk):
    """
    Class that adds certain variables to the data.
    """
    calc_instance = CalcInit
    filter_instance = FilterInit
    
    def __init__(self, look_for):
        super().__init__()
        self.look_for = look_for
    
    @classmethod
    def set_instance(cls, **kwargs):
        """
        Sets the class instances used in the following.

        :param kwargs: class instances
        """
        if any("calc" in it for it in kwargs.keys()):
            cls.calc_instance = kwargs["calc_instance"]
        if any("filter" in it for it in kwargs.keys()):
            cls.filter_instance = kwargs["filter_instance"]
    
    def pt(self, awk_array):
        """
        Adds transverse momentum to the pandas series.
        
        workflow:
            - search for px, py in a event
            - calculates pt with a user defined function (Calc_Start.pt)
            - adds pt to quantities
            - return event with added pt
        
        :return: pd.Series
        """
        
        _calc_func = AddVariable.calc_instance.pt
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            awk_array[f"{self.look_for}_pt"] = _calc_func(awk_array[f"{self.look_for}_px"],
                                                          awk_array[f"{self.look_for}_py"])
        
        if self.look_for == "both":
            
            awk_array[f"muon_pt"] = _calc_func(awk_array[f"muon_px"],
                                               awk_array[f"muon_py"])
            
            awk_array[f"electron_pt"] = _calc_func(awk_array[f"electron_px"],
                                                   awk_array[f"electron_py"])
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    def pseudorapidity(self, awk_array):
        """
        Adds pseudorapidity to the pandas series.
        
        workflow:
            - search for px, py, pz in a event
            - calculates pseudorapidity with a user defined function (CalcStudent.pseudorapidity)
            - adds pseudorapidity to quantities
            - return event with added pseudorapidity

        :return: pd.Series
        """
        
        _calc_func = AddVariable.calc_instance.pseudorapidity
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            awk_array[f"{self.look_for}_pseudorapidity"] = _calc_func(awk_array[f"{self.look_for}_px"],
                                                                      awk_array[f"{self.look_for}_py"],
                                                                      awk_array[f"{self.look_for}_pz"],
                                                                      energy=awk_array[f"{self.look_for}_energy"])
            
            return awk_array
        
        if self.look_for == "both":
            
            awk_array["muon_pseudorapidity"] = _calc_func(awk_array[f"muon_px"],
                                                          awk_array[f"muon_py"],
                                                          awk_array[f"muon_pz"],
                                                          energy=awk_array[f"muon_energy"])
            
            awk_array["electron_pseudorapidity"] = _calc_func(awk_array[f"electron_px"],
                                                              awk_array[f"electron_py"],
                                                              awk_array[f"electron_pz"],
                                                              energy=awk_array[f"electron_energy"])
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    def phi(self, awk_array):
        """
        Adds phi angle to the pandas series.
        
        workflow:
            - search for px, py in a event
            - calculates phi with a user defined function (CalcStudent.phi)
            - adds phi to quantities
            - return event with added phi
        
        :return: pd.Series
        """
        
        _calc_func = AddVariable.calc_instance.phi
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            awk_array[f"{self.look_for}_phi"] = _calc_func(awk_array[f"{self.look_for}_px"],
                                                           awk_array[f"{self.look_for}_py"])
            
            return awk_array
        
        if self.look_for == "both":
            
            awk_array[f"muon_phi"] = _calc_func(awk_array[f"muon_px"],
                                                awk_array[f"muon_py"])
            
            awk_array[f"electron_phi"] = _calc_func(awk_array[f"electron_px"],
                                                    awk_array[f"electron_py"])
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")


class Reconstruct(ProcessingBasicAwk):
    """
    Class that provides the reconstruction methods.
    """
    calc_instance = CalcInit
    filter_instance = FilterInit
    
    def __init__(self, look_for):
        super().__init__()
        self.look_for = look_for
    
    @classmethod
    def set_instance(cls, **kwargs):
        """
        Sets the class instances used in the following.

        :param kwargs: class instances
        """
        if any("calc" in it for it in kwargs.keys()):
            cls.calc_instance = kwargs["calc_instance"]
        if any("filter" in it for it in kwargs.keys()):
            cls.filter_instance = kwargs["filter_instance"]
    
    def zz(self, awk_array):
        """
        Reconstructs a Z boson pair and adds it and all the necessary quantities to the pandas series.
        
        workflow:
            - search for variables in a event that are required for mass calculation
                - calculated defined  quantities (if not present) are:
                               - pseudorapidity (CalcStudent.pseudorapidity),
                               - phi (CalcStudent.phi),
                               - pt  (Calc_Start.pt)
            - calculates zz and filters with (partially user defined):
                               - zz: (FilterStudent.zz),
                               - delta_r: FilterInit.delta_r
            - adds quantities to event: z1 and z2 mass, used lepton index for z reconstruction
                                        (tag "e" == electron or "m" == muon if 2e2mu channel)
            - return event with added quantities
        
        :return: pd.Series
        """
        
        _calc_func_pseudorapidity = Reconstruct.calc_instance.pseudorapidity
        _calc_func_pt = Reconstruct.calc_instance.pt
        _calc_func_phi = Reconstruct.calc_instance.phi
        _reconstruct_zz = Reconstruct.calc_instance.zz_and_index
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            
            if f"{self.look_for}_pt" not in awk_array.fields:
                awk_array[f"{self.look_for}_pt"] = _calc_func_pt(awk_array[f"{self.look_for}_px"],
                                                                 awk_array[f"{self.look_for}_py"])
            
            if f"{self.look_for}_pseudorapidity" not in awk_array.fields:
                awk_array[f"{self.look_for}_pseudorapidity"] = _calc_func_pseudorapidity(awk_array[f"{self.look_for}_px"],
                                                                                         awk_array[f"{self.look_for}_py"],
                                                                                         awk_array[f"{self.look_for}_pz"],
                                                                                         energy=awk_array[f"{self.look_for}_energy"])
            
            if f"{self.look_for}_phi" not in awk_array.fields:
                awk_array[f"{self.look_for}_phi"] = [_calc_func_phi(px, py) for px, py in zip(awk_array[f"{self.look_for}_px"],
                                                                                              awk_array[f"{self.look_for}_py"])]
            
            awk_array = Reconstruct.calc_instance.zz_and_index(awk_array=awk_array, lepton_type=self.look_for)
            
            return awk_array
        
        if self.look_for == "both":
            
            if f"muon_pt" not in awk_array.fields:
                awk_array[f"muon_pt"] = _calc_func_pt(awk_array[f"muon_px"],
                                                      awk_array[f"muon_py"])
            
            if f"muon_pseudorapidity" not in awk_array.fields:
                awk_array[f"muon_pseudorapidity"] = _calc_func_pseudorapidity(awk_array[f"muon_px"], awk_array[f"muon_py"],
                                                                              awk_array[f"muon_pz"], energy=awk_array[f"muon_energy"])
            
            if f"muon_phi" not in awk_array.fields:
                awk_array[f"muon_phi"] = [_calc_func_phi(px, py) for px, py in zip(awk_array[f"muon_px"],
                                                                                   awk_array[f"muon_py"])]
            
            if f"electron_pt" not in awk_array.fields:
                awk_array[f"electron_pt"] = _calc_func_pt(awk_array[f"electron_px"],
                                                          awk_array[f"electron_py"])
            
            if f"electron_pseudorapidity" not in awk_array.fields:
                awk_array[f"electron_pseudorapidity"] = _calc_func_pseudorapidity(awk_array[f"electron_px"], awk_array[f"electron_py"],
                                                                                  awk_array[f"electron_pz"], energy=awk_array[f"electron_energy"])
            
            if f"electron_phi" not in awk_array.fields:
                awk_array[f"electron_phi"] = [_calc_func_phi(px, py) for px, py in zip(awk_array[f"electron_px"],
                                                                                       awk_array[f"electron_py"])]
            
            awk_array = _reconstruct_zz(awk_array, lepton_type=self.look_for)
            
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    def four_lepton_mass_from_zz(self, awk_array):
        """
        Reconstructs the four lepton invariant masses from the already reconstructed Z boson pair.
        
        workflow:
            - zz reconstruction needed!
            - search for variables in a event that are required for mass calculation
            - calculates four lepton mass (CalcInit.mass_4l_out_zz) with help of a student
              provided function (CalcStudent.invariant_mass_square)
            - adds four lepton invariant mass to event (mass_4l)
            - return event with added quantities
        
        :return: pd.Series
        """
        
        _calc_mass = self.calc_instance.mass_4l_out_zz
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            awk_array = _calc_mass(awk_array, self.look_for)
            return awk_array
        
        if self.look_for == "both":
            
            awk_array = _calc_mass(awk_array, self.look_for)
            return awk_array
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
