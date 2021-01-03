# -*- coding: UTF-8 -*-

import numpy as np

from .CalcAndAllowerInit import FilterInit, CalcInit
from .ProcessingRow import ProcessingRow


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray


class FilterStr(ProcessingRow):
    """
    Class that provides certain filters for data reduction.
    """
    calc_instance = CalcInit
    filter_instance = FilterInit
    
    def __init__(self, row, name_list, look_for):
        super(FilterStr, self).__init__(row, name_list)
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
    def lepton_detector_classification(self):
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
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["type"],
                                          type_variables=[str])  # [lepton_type + "_type"], type_variables=[str])
            type_array = found_array[0]
            
            accept_array = FilterStr.filter_instance.lepton_type(type_array, self.look_for)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_type", "electron_type"], type_variables=[str, str])
            type_mu, type_el = found_array[0], found_array[1]
            
            accept_array_mu = FilterStr.filter_instance.lepton_type(type_mu, "muon")
            accept_array_el = FilterStr.filter_instance.lepton_type(type_el, "electron")
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 2
    def electric_charge(self):
        """
        Filter that checks whether an electrically neutral charge
        combination can be formed from the leptons contained in the event.
        
        workflow:
            - search for electron_charge or muon_charge in a event
            - retrieve a user defined boolean (CalcStudent.combined_charge)
            - return unchanged event if (CalcStudent.combined_charge) returns True or discard else
        
        :return: pd.Series
        """
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=[self.look_for + "_charge"], type_variables=[int])
            charge = found_array[0]
            
            accept_charge = FilterStr.filter_instance.combined_charge(charge, 4)
            self.eval_and_reduce(to_accept_bool=accept_charge)
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_charge", "electron_charge"],
                                          type_variables=[int, int])
            charge_mu, charge_el = found_array[0], found_array[1]
            
            accept_charge_mu = FilterStr.filter_instance.combined_charge(charge_mu, 2)
            accept_charge_el = FilterStr.filter_instance.combined_charge(charge_el, 2)
            
            self.eval_and_reduce(to_accept_bool=(accept_charge_mu and accept_charge_el))
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 3
    def pt_min(self):
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
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py"], type_variables=[float, float])
            px, py = found_array[0], found_array[1]
            
            pt = FilterStr.calc_instance.pt(px, py)
            self.add_raw_to_row(variable_name=f"{self.look_for}_pt", variable_array=pt)
            accept_array = FilterStr.filter_instance.pt_min(pt, self.look_for)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "electron_px", "electron_py"],
                                          type_variables=[float, float, float, float])
            px_mu, py_mu = found_array[0], found_array[1]
            px_el, py_el = found_array[2], found_array[3]
            
            pt_mu = FilterStr.calc_instance.pt(px_mu, py_mu)
            pt_el = FilterStr.calc_instance.pt(px_el, py_el)
            self.add_raw_to_row(variable_name="muon_pt", variable_array=pt_mu)
            self.add_raw_to_row(variable_name="electron_pt", variable_array=pt_el)
            accept_array_mu = FilterStr.filter_instance.pt_min(pt_mu, "muon")
            accept_array_el = FilterStr.filter_instance.pt_min(pt_el, "electron")
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 4
    def pseudorapidity(self):
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
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py", "pz", "energy"],
                                          type_variables=[float, float, float, float])
            px, py, pz, energy = found_array[0], found_array[1], found_array[2], found_array[3]
            
            pseudorapidity = FilterStr.calc_instance.pseudorapidity(px, py, pz, energy=energy)
            self.add_raw_to_row(variable_name=f"{self.look_for}_pseudorapidity", variable_array=pseudorapidity)
            accept_array = FilterStr.filter_instance.pseudorapidity(pseudorapidity, self.look_for)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "muon_pz",
                                                            "electron_px", "electron_py", "electron_pz",
                                                            "muon_energy", "electron_energy"],
                                          type_variables=[float, float, float, float, float, float, float, float])
            px_mu, py_mu, pz_mu = found_array[0], found_array[1], found_array[2]
            px_el, py_el, pz_el = found_array[3], found_array[4], found_array[5]
            energy_mu, energy_el = found_array[6], found_array[7]
            
            pseudorapidity_mu = FilterStr.calc_instance.pseudorapidity(px_mu, py_mu, pz_mu, energy=energy_mu)
            pseudorapidity_el = FilterStr.calc_instance.pseudorapidity(px_el, py_el, pz_el, energy=energy_el)
            self.add_raw_to_row(variable_name="muon_pseudorapidity", variable_array=pseudorapidity_mu)
            self.add_raw_to_row(variable_name="electron_pseudorapidity", variable_array=pseudorapidity_el)
            accept_array_mu = FilterStr.filter_instance.pseudorapidity(pseudorapidity_mu, "muon")
            accept_array_el = FilterStr.filter_instance.pseudorapidity(pseudorapidity_el, "electron")
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 5
    def misshit(self):
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
        if self.dataframe_head():
            return self.row
        
        if self.look_for == "electron":
            found_array = self.search_for(search_variables=["misshits"], type_variables=[int])
            misshits = found_array[0]
            
            accept_array = FilterStr.filter_instance.misshits(misshits)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["misshits"], type_variables=[int])
            misshits = found_array[0]
            
            accept_array_el = FilterStr.filter_instance.misshits(misshits)
            accept_array_mu = np.ones(len(accept_array_el), dtype=bool)  # dummy for separation mu/el in reduce_row
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 6
    def relative_isolation(self):
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
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["relPFIso"], type_variables=[float])
            rel_pf_iso = found_array[0]
            
            accept_array = FilterStr.filter_instance.relative_isolation(rel_pf_iso)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_relPFIso", "electron_relPFIso"],
                                          type_variables=[float, float])
            rel_pf_iso_mu, rel_pf_iso_el = found_array[0], found_array[1]
            
            accept_array_mu = FilterStr.filter_instance.relative_isolation(rel_pf_iso_mu)
            accept_array_el = FilterStr.filter_instance.relative_isolation(rel_pf_iso_el)
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 7
    def impact_parameter(self):
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
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["SIP3d", "dxy", "dz"],
                                          type_variables=[float, float, float])
            sip3d, dxy, dz = found_array[0], found_array[1], found_array[2]
            
            accept_array = FilterStr.filter_instance.impact_parameter(sip3d, dxy, dz)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_SIP3d", "muon_dxy", "muon_dz",
                                                            "electron_SIP3d", "electron_dxy", "electron_dz"],
                                          type_variables=[float, float, float, float, float, float])
            sip3d_mu, dxy_mu, dz_mu = found_array[0], found_array[1], found_array[2]
            sip3d_el, dxy_el, dz_el = found_array[3], found_array[4], found_array[5]
            
            accept_array_mu = FilterStr.filter_instance.impact_parameter(sip3d_mu, dxy_mu, dz_mu)
            accept_array_el = FilterStr.filter_instance.impact_parameter(sip3d_el, dxy_el, dz_el)
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        return None
    
    # Filter No 8
    def pt_exact(self):
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
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py"], type_variables=[float, float])
            px, py = found_array[0], found_array[1]
            
            pt = FilterStr.calc_instance.pt(px, py)
            self.add_raw_to_row(variable_name=f"{self.look_for}_pt", variable_array=pt)
            accept_value = FilterStr.filter_instance.pt_exact(pt, lepton_type=self.look_for)
            
            self.eval_and_reduce(to_accept_bool=accept_value)
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "electron_px", "electron_py"],
                                          type_variables=[float, float, float, float])
            px_mu, py_mu, px_el, py_el = found_array[0], found_array[1], found_array[2], found_array[3]
            
            pt_mu = FilterStr.calc_instance.pt(px_mu, px_mu)
            pt_el = FilterStr.calc_instance.pt(px_el, py_el)
            self.add_raw_to_row(variable_name="muon_pt", variable_array=pt_mu)
            self.add_raw_to_row(variable_name="electron_pt", variable_array=pt_el)
            accept_mu_and_el = FilterStr.filter_instance.pt_exact(p_t=(pt_mu, pt_el), lepton_type="both")
            self.eval_and_reduce(to_accept_bool=accept_mu_and_el)
            
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 9
    def two_lepton_mass(self):
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
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["energy", "px", "py", "pz", "charge"],
                                          type_variables=[float, float, float, float, float])
            energy, charge = found_array[0], found_array[4]
            px, py, pz = found_array[1], found_array[2], found_array[3]
            
            inv_m = FilterStr.calc_instance.possible_invariant_masses(px, py, pz, charge, energy=energy,
                                                                      number_of_leptons=2)
            accept_value = FilterStr.filter_instance.invariant_mass(inv_m, 2, lepton_type=self.look_for)
            
            self.eval_and_reduce(to_accept_bool=accept_value)
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_energy", "muon_charge",
                                                            "muon_px", "muon_py", "muon_pz",
                                                            "electron_energy", "electron_charge",
                                                            "electron_px", "electron_py", "electron_pz"],
                                          type_variables=[float, float, float, float, float,
                                                          float, float, float, float, float])
            energy_mu, charge_mu = found_array[0], found_array[1]
            energy_el, charge_el = found_array[5], found_array[6]
            px_mu, py_mu, pz_mu = found_array[2], found_array[3], found_array[4]
            px_el, py_el, pz_el = found_array[7], found_array[8], found_array[9]
            
            inv_m_mu = FilterStr.calc_instance.possible_invariant_masses(px_mu, py_mu, pz_mu, charge_mu,
                                                                         energy=energy_mu,
                                                                         number_of_leptons=2)
            inv_m_el = FilterStr.calc_instance.possible_invariant_masses(px_el, py_el, pz_el, charge_el,
                                                                         energy=energy_el,
                                                                         number_of_leptons=2)
            accept_value_mu = FilterStr.filter_instance.invariant_mass(inv_m_mu, 2, lepton_type=self.look_for)
            accept_value_el = FilterStr.filter_instance.invariant_mass(inv_m_el, 2, lepton_type=self.look_for)
            
            self.eval_and_reduce(to_accept_bool=(accept_value_mu and accept_value_el))
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 10
    def four_lepton_mass(self):
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
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["energy", "px", "py", "pz", "charge"],
                                          type_variables=[float, float, float, float, float])
            
            energy, charge = found_array[0], found_array[4]
            px, py, pz = found_array[1], found_array[2], found_array[3]
            
            inv_m = FilterStr.calc_instance.possible_invariant_masses(px, py, pz, charge, energy=energy,
                                                                      number_of_leptons=4)
            accept_value = FilterStr.filter_instance.invariant_mass(inv_m, 4, lepton_type=self.look_for)
            
            self.eval_and_reduce(to_accept_bool=accept_value)
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_energy", "muon_px", "muon_py", "muon_pz",
                                                            "electron_energy", "electron_px", "electron_py",
                                                            "electron_pz", "muon_charge", "electron_charge"],
                                          type_variables=[float, float, float, float, float,
                                                          float, float, float, float, float])
            
            energy_mu, energy_el = found_array[0], found_array[4]
            charge_mu, charge_el = found_array[8], found_array[9]
            px_mu, py_mu, pz_mu = found_array[1], found_array[2], found_array[3]
            px_el, py_el, pz_el = found_array[5], found_array[6], found_array[7]
            
            inv_m = FilterStr.calc_instance.possible_invariant_masses([px_mu, px_el], [py_mu, py_el], [pz_mu, pz_el],
                                                                      [charge_mu, charge_el],
                                                                      energy=[energy_mu, energy_el],
                                                                      number_of_leptons=4, look_for_both=True)
            accept_value = FilterStr.filter_instance.invariant_mass(inv_m, 4, lepton_type=self.look_for)
            
            self.eval_and_reduce(to_accept_bool=accept_value)
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")


class AddVariable(ProcessingRow):
    """
    Class that adds certain variables to the data.
    """
    calc_instance = CalcInit
    filter_instance = FilterInit
    
    def __init__(self, row, name_list, look_for):
        super(AddVariable, self).__init__(row, name_list)
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
    
    def pt(self):
        """
        Adds transverse momentum to the pandas series.
        
        workflow:
            - search for px, py in a event
            - calculates pt with a user defined function (Calc_Start.pt)
            - adds pt to quantities
            - return event with added pt
        
        :return: pd.Series
        """
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py"], type_variables=[float, float])
            px, py = found_array[0], found_array[1]
            
            pt = AddVariable.calc_instance.pt(px, py)
            self.add_raw_to_row(variable_name=f"{self.look_for}_pt", variable_array=pt)
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "electron_px", "electron_py"],
                                          type_variables=[float, float, float, float])
            px_mu, py_mu = found_array[0], found_array[1]
            px_el, py_el = found_array[2], found_array[3]
            
            pt_mu = AddVariable.calc_instance.pt(px_mu, py_mu)
            pt_el = AddVariable.calc_instance.pt(px_el, py_el)
            self.add_raw_to_row(variable_name="muon_pt", variable_array=pt_mu)
            self.add_raw_to_row(variable_name="electron_pt", variable_array=pt_el)
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    def pseudorapidity(self):
        """
        Adds pseudorapidity to the pandas series.
        
        workflow:
            - search for px, py, pz in a event
            - calculates pseudorapidity with a user defined function (CalcStudent.pseudorapidity)
            - adds pseudorapidity to quantities
            - return event with added pseudorapidity

        :return: pd.Series
        """
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py", "pz"],
                                          type_variables=[float, float, float])
            px, py, pz = found_array[0], found_array[1], found_array[2]
            
            pseudorapidity = AddVariable.calc_instance.pseudorapidity(px, py, pz)
            self.add_raw_to_row(variable_name=f"{self.look_for}_pseudorapidity", variable_array=pseudorapidity)
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "muon_pz",
                                                            "electron_px", "electron_py", "electron_pz"],
                                          type_variables=[float, float, float, float, float, float])
            px_mu, py_mu, pz_mu = found_array[0], found_array[1], found_array[2]
            px_el, py_el, pz_el = found_array[3], found_array[4], found_array[5]
            
            pseudorapidity_mu = AddVariable.calc_instance.pseudorapidity(px_mu, py_mu, pz_mu)
            pseudorapidity_el = AddVariable.calc_instance.pseudorapidity(px_el, py_el, pz_el)
            self.add_raw_to_row(variable_name="muon_pseudorapidity", variable_array=pseudorapidity_mu)
            self.add_raw_to_row(variable_name="electron_pseudorapidity", variable_array=pseudorapidity_el)
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    def phi(self):
        """
        Adds phi angle to the pandas series.
        
        workflow:
            - search for px, py in a event
            - calculates phi with a user defined function (CalcStudent.phi)
            - adds phi to quantities
            - return event with added phi
        
        :return: pd.Series
        """
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py"],
                                          type_variables=[float, float])
            px, py = found_array[0], found_array[1]
            
            phi = AddVariable.calc_instance.phi(px, py)
            self.add_raw_to_row(variable_name=f"{self.look_for}_phi", variable_array=phi)
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "electron_px", "electron_py"],
                                          type_variables=[float, float, float, float])
            px_mu, py_mu = found_array[0], found_array[1]
            px_el, py_el = found_array[2], found_array[3]
            
            phi_mu = AddVariable.calc_instance.phi(px_mu, py_mu)
            phi_el = AddVariable.calc_instance.phi(px_el, py_el)
            self.add_raw_to_row(variable_name="muon_phi", variable_array=phi_mu)
            self.add_raw_to_row(variable_name="electron_phi", variable_array=phi_el)
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")


class Reconstruct(ProcessingRow):
    """
    Class that provides the reconstruction methods.
    """
    calc_instance = CalcInit
    filter_instance = FilterInit
    
    def __init__(self, row, name_list, look_for):
        super(Reconstruct, self).__init__(row, name_list)
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
    
    def zz(self):
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
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            
            found_array = self.search_for(search_variables=["pseudorapidity", "px", "py", "pz", "energy",
                                                            "pt", "charge"],
                                          type_variables=[float, float, float, float, float, float, float])
            pseudorapidity, px, py, pz, energy = found_array[0], found_array[1], found_array[2], found_array[3], found_array[4]
            pt, charge = found_array[5], found_array[6]
            
            phi = Reconstruct.calc_instance.phi(px, py)
            z1, z2, index_z1, index_z2 = Reconstruct.calc_instance.zz_and_index(pseudorapidity, phi, pt, px, py, pz, charge,
                                                                                energy=energy,
                                                                                look_for=self.look_for)
            
            if z1 == 0:
                self.row["run"] = np.nan
                return self.row
            
            self.add_raw_to_row(variable_name=f"{self.look_for}_phi", variable_array=phi)
            self.add_raw_to_row(variable_name="z1_mass", variable_array=z1)
            self.add_raw_to_row(variable_name="z2_mass", variable_array=z2)
            self.add_raw_to_row(variable_name="z1_index", variable_array=index_z1)
            self.add_raw_to_row(variable_name="z2_index", variable_array=index_z2)
            
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_pseudorapidity", "muon_px", "muon_py", "muon_pz",
                                                            "muon_energy", "muon_pt", "muon_charge",
                                                            "electron_pseudorapidity", "electron_px", "electron_py", "electron_pz",
                                                            "electron_energy", "electron_pt", "electron_charge"],
                                          type_variables=[float, float, float, float, float, float, float,
                                                          float, float, float, float, float, float, float])
            pseudorapidity_mu, px_mu, py_mu, pz_mu = found_array[0], found_array[1], found_array[2], found_array[3]
            energy_mu, pt_mu, charge_mu = found_array[4], found_array[5], found_array[6]
            pseudorapidity_el, px_el, py_el, pz_el = found_array[7], found_array[8], found_array[9], found_array[10]
            energy_el, pt_el, charge_el = found_array[11], found_array[12], found_array[13]
            
            phi_mu = Reconstruct.calc_instance.phi(px_mu, py_mu)
            phi_el = Reconstruct.calc_instance.phi(px_el, py_el)
            z1, z2, index_z1, index_z2, z1_tag, z2_tag = Reconstruct.calc_instance.zz_and_index([pseudorapidity_mu, pseudorapidity_el],
                                                                                                [phi_mu, phi_el],
                                                                                                [pt_mu, pt_el],
                                                                                                [px_mu, px_el],
                                                                                                [py_mu, py_el],
                                                                                                [pz_mu, pz_el],
                                                                                                [charge_mu, charge_el],
                                                                                                look_for=self.look_for)
            
            if z1 == 0.0:
                self.row["run"] = np.nan
                return self.row
            
            self.add_raw_to_row(variable_name="muon_phi", variable_array=phi_mu)
            self.add_raw_to_row(variable_name="electron_phi", variable_array=phi_el)
            self.add_raw_to_row(variable_name="z1_mass", variable_array=z1)
            self.add_raw_to_row(variable_name="z2_mass", variable_array=z2)
            self.add_raw_to_row(variable_name="z1_index", variable_array=index_z1)
            self.add_raw_to_row(variable_name="z2_index", variable_array=index_z2)
            self.add_raw_to_row(variable_name="z1_tag", variable_array=z1_tag)
            self.add_raw_to_row(variable_name="z2_tag", variable_array=z2_tag)
            
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
    
    def four_lepton_mass_from_zz(self):
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
        if self.dataframe_head():
            return self.row
        
        if self.look_for != "both" and (self.look_for == "muon" or self.look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py", "pz", "z1_index", "z2_index", "energy"],
                                          type_variables=[float, float, float, int, int, float])
            px, py, pz = found_array[0], found_array[1], found_array[2]
            z1_index, z2_index = found_array[3], found_array[4]
            energy = found_array[5]
            
            mass_hi = self.calc_instance.mass_4l_out_zz(px, pz, py, [z1_index, z2_index], look_for=self.look_for, energy=energy)
            
            self.add_raw_to_row(variable_name="mass_4l", variable_array=mass_hi)
            
            if mass_hi == 0.0:
                self.row["run"] = np.nan
            
            return self.row
        
        if self.look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "muon_pz",
                                                            "z1_index", "z2_index", "z1_tag", "z2_tag",
                                                            "electron_px", "electron_py", "electron_pz",
                                                            "muon_energy", "electron_energy"],
                                          type_variables=[float, float, float, int, int, str, str, float, float, float, float, float])
            
            px_mu, py_mu, pz_mu = found_array[0], found_array[1], found_array[2]
            z1_index, z2_index, z1_tag, z2_tag = found_array[3], found_array[4], found_array[5], found_array[6]
            px_el, py_el, pz_el = found_array[7], found_array[8], found_array[9]
            energy_mu, energy_el = found_array[10], found_array[11]
            
            mass_hi = self.calc_instance.mass_4l_out_zz([px_mu, px_el], [py_mu, py_el], [pz_mu, pz_el],
                                                        [z1_index, z2_index],
                                                        energy=[energy_mu, energy_el],
                                                        tag=[str(z1_tag[0]), str(z2_tag[0])], look_for=self.look_for)
            
            self.add_raw_to_row(variable_name="mass_4l", variable_array=mass_hi)
            
            if mass_hi == 0.0:
                self.row["run"] = np.nan
            
            return self.row
        
        else:
            raise TypeError("'lepton_type' can only be: 'muon', 'electron' or 'both'")
