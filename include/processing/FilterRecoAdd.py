# -*- coding: UTF-8 -*-

import numpy as np
import swifter

from .CalcAndAllowerInit import AllowedInit, CalcInit
from .ProcessingRow import ProcessingRow


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray

sw_temp_ = swifter


class FilterStr(ProcessingRow):
    """
    Class that provides certain filters for data reduction.
    """
    calc_instance = CalcInit
    allowed_instance = AllowedInit
    
    def __init__(self, row, name_list):
        super(FilterStr, self).__init__(row, name_list)
    
    @classmethod
    def set_instance(cls, **kwargs):
        """
        Sets the class instances used in the following.

        :param kwargs: class instances
        """
        if any("calc" in it for it in kwargs.keys()): cls.calc_instance = kwargs["calc_instance"]
        if any("allowed" in it for it in kwargs.keys()): cls.allowed_instance = kwargs["allowed_instance"]
    
    # Filter No 1
    def check_type(self, look_for="muon"):
        """
        Filter checks the classification of the leptons.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["type"],
                                          type_variables=[str])  # [look_for + "_type"], type_variables=[str])
            type_array = found_array[0]
            
            accept_array = FilterStr.allowed_instance.lepton_type(type_array, look_for)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_type", "electron_type"], type_variables=[str, str])
            type_mu, type_el = found_array[0], found_array[1]
            
            accept_array_mu = FilterStr.allowed_instance.lepton_type(type_mu, "muon")
            accept_array_el = FilterStr.allowed_instance.lepton_type(type_el, "electron")
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 2
    def check_q(self, look_for="muon"):
        """
        Filter that checks whether an electrically neutral charge
        combination can be formed from the leptons contained in the event.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=[look_for + "_charge"], type_variables=[int])
            charge = found_array[0]
            
            accept_charge = FilterStr.calc_instance.combined_charge(charge, 4)
            self.eval_and_reduce(to_accept_bool=(accept_charge == 0))
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_charge", "electron_charge"],
                                          type_variables=[int, int])
            charge_mu, charge_el = found_array[0], found_array[1]
            
            accept_charge_mu = FilterStr.calc_instance.combined_charge(charge_mu, 2)
            accept_charge_el = FilterStr.calc_instance.combined_charge(charge_el, 2)
            
            self.eval_and_reduce(to_accept_bool=(accept_charge_mu == 0 and accept_charge_el == 0))
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 3
    def check_min_pt(self, look_for="muon"):
        """
        Filter of leptons removed whose transverse impulse is smaller than the minimum allowed.
        Also the transversal impulse is added.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column():
            return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py"], type_variables=[float, float])
            px, py = found_array[0], found_array[1]
            
            pt = FilterStr.calc_instance.pt(px, py)
            self.add_raw_to_row(variable_name="pt", variable_array=pt)
            accept_array = FilterStr.allowed_instance.min_pt(pt, look_for)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "electron_px", "electron_py"],
                                          type_variables=[float, float, float, float])
            px_mu, py_mu = found_array[0], found_array[1]
            px_el, py_el = found_array[2], found_array[3]
            
            pt_mu = FilterStr.calc_instance.pt(px_mu, py_mu)
            pt_el = FilterStr.calc_instance.pt(px_el, py_el)
            self.add_raw_to_row(variable_name="muon_pt", variable_array=pt_mu)
            self.add_raw_to_row(variable_name="electron_pt", variable_array=pt_el)
            accept_array_mu = FilterStr.allowed_instance.min_pt(pt_mu, "muon")
            accept_array_el = FilterStr.allowed_instance.min_pt(pt_el, "electron")
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 4
    def check_eta(self, look_for="muon"):
        """
        Filter of leptons removed based on the calculated pseudorapidity.
        Also the pseudorapidity is added.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column():
            return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py", "pz"],
                                          type_variables=[float, float, float])
            px, py, pz = found_array[0], found_array[1], found_array[2]
            
            eta = FilterStr.calc_instance.eta(px, py, pz)
            self.add_raw_to_row(variable_name="eta", variable_array=eta)
            accept_array = FilterStr.allowed_instance.eta(eta, look_for)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "muon_pz",
                                                            "electron_px", "electron_py", "electron_pz"],
                                          type_variables=[float, float, float, float, float, float])
            px_mu, py_mu, pz_mu = found_array[0], found_array[1], found_array[2]
            px_el, py_el, pz_el = found_array[3], found_array[4], found_array[5]
            
            eta_mu = FilterStr.calc_instance.eta(px_mu, py_mu, pz_mu)
            eta_el = FilterStr.calc_instance.eta(px_el, py_el, pz_el)
            self.add_raw_to_row(variable_name="muon_eta", variable_array=eta_mu)
            self.add_raw_to_row(variable_name="electron_eta", variable_array=eta_el)
            accept_array_mu = FilterStr.allowed_instance.eta(eta_mu, "muon")
            accept_array_el = FilterStr.allowed_instance.eta(eta_el, "electron")
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 5
    def check_misshit(self, look_for="electron"):
        """
        Filters out electrons from an event that have an insufficient number of missing hints.

        :param look_for: str
                         "electron" or "both"
        :return: pd.Series
        """
        if self.if_column():
            return self.row
        
        if look_for == "electron":
            found_array = self.search_for(search_variables=["misshits"], type_variables=[int])
            misshits = found_array[0]
            
            accept_array = FilterStr.allowed_instance.misshits(misshits)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["misshits"], type_variables=[int])
            misshits = found_array[0]
            
            accept_array_el = FilterStr.allowed_instance.misshits(misshits)
            accept_array_mu = np.ones(len(accept_array_el), dtype=bool)  # dummy for separation mu/el in __reduce_row
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 6
    def check_rel_iso(self, look_for="muon"):
        """
        Filter of leptons discarded based on relative isolation.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column():
            return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["relPFIso"], type_variables=[float])
            rel_pf_iso = found_array[0]
            
            accept_array = FilterStr.allowed_instance.rel_pf_iso(rel_pf_iso)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_relPFIso", "electron_relPFIso"],
                                          type_variables=[float, float])
            rel_pf_iso_mu, rel_pf_iso_el = found_array[0], found_array[1]
            
            accept_array_mu = FilterStr.allowed_instance.rel_pf_iso(rel_pf_iso_mu)
            accept_array_el = FilterStr.allowed_instance.rel_pf_iso(rel_pf_iso_el)
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 7
    def check_impact_param(self, look_for="muon"):
        """
        Filter of leptons discarded based on the impact parameter.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column():
            return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["SIP3d", "dxy", "dz"],
                                          type_variables=[float, float, float])
            sip3d, dxy, dz = found_array[0], found_array[1], found_array[2]
            
            accept_array = FilterStr.allowed_instance.impact_param(sip3d, dxy, dz)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_SIP3d", "muon_dxy", "muon_dz",
                                                            "electron_SIP3d", "electron_dxy", "electron_dz"],
                                          type_variables=[float, float, float, float, float, float])
            sip3d_mu, dxy_mu, dz_mu = found_array[0], found_array[1], found_array[2]
            sip3d_el, dxy_el, dz_el = found_array[3], found_array[4], found_array[5]
            
            accept_array_mu = FilterStr.allowed_instance.impact_param(sip3d_mu, dxy_mu, dz_mu)
            accept_array_el = FilterStr.allowed_instance.impact_param(sip3d_el, dxy_el, dz_el)
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        return None
    
    # Filter No 8
    def check_exact_pt(self, look_for="muon"):
        """
        Filters leptons based on the exact required transverse momentum of each lepton in an event.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py"], type_variables=[float, float])
            px, py = found_array[0], found_array[1]
            
            pt = FilterStr.calc_instance.pt(px, py)
            self.add_raw_to_row(variable_name="pt", variable_array=pt)
            accept_value = FilterStr.allowed_instance.pt(pt, look_for)
            
            self.eval_and_reduce(to_accept_bool=accept_value)
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "electron_px", "electron_py"],
                                          type_variables=[float, float, float, float])
            px_mu, py_mu, px_el, py_el = found_array[0], found_array[1], found_array[2], found_array[3]
            
            pt_mu = FilterStr.calc_instance.pt(px_mu, px_mu)
            pt_el = FilterStr.calc_instance.pt(px_el, py_el)
            self.add_raw_to_row(variable_name="muon_pt", variable_array=pt_mu)
            self.add_raw_to_row(variable_name="electron_pt", variable_array=pt_el)
            accept_mu = FilterStr.allowed_instance.pt(pt_mu, "muon", 2)
            accept_el = FilterStr.allowed_instance.pt(pt_el, "electron", 2)
            
            self.eval_and_reduce(to_accept_bool=(accept_el and accept_mu))
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 9
    def check_m_2l(self, look_for="muon"):
        """
        Filters leptons according to the minimum required two lepton invariant masses.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["energy", "px", "py", "pz", "charge"],
                                          type_variables=[float, float, float, float, float])
            energy, charge = found_array[0], found_array[4]
            px, py, pz = found_array[1], found_array[2], found_array[3]
            
            inv_m = FilterStr.calc_instance.possible_invariant_masses(px, py, pz, charge, energy=energy,
                                                                      number_of_leptons=2)
            accept_value = FilterStr.allowed_instance.invariant_mass(inv_m, 2, look_for=look_for)
            
            self.eval_and_reduce(to_accept_bool=accept_value)
            return self.row
        
        if look_for == "both":
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
            accept_value_mu = FilterStr.allowed_instance.invariant_mass(inv_m_mu, 2, look_for=look_for)
            accept_value_el = FilterStr.allowed_instance.invariant_mass(inv_m_el, 2, look_for=look_for)
            
            self.eval_and_reduce(to_accept_bool=(accept_value_mu and accept_value_el))
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 10
    def check_m_4l(self, look_for="muon"):
        """
        Filters leptons according to the minimum required four lepton invariant masses.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["energy", "px", "py", "pz", "charge"],
                                          type_variables=[float, float, float, float, float])
            
            energy, charge = found_array[0], found_array[4]
            px, py, pz = found_array[1], found_array[2], found_array[3]
            
            inv_m = FilterStr.calc_instance.possible_invariant_masses(px, py, pz, charge, energy=energy,
                                                                      number_of_leptons=4)
            accept_value = FilterStr.allowed_instance.invariant_mass(inv_m, 4, look_for=look_for)
            
            self.eval_and_reduce(to_accept_bool=accept_value)
            return self.row
        
        if look_for == "both":
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
            accept_value = FilterStr.allowed_instance.invariant_mass(inv_m, 4, look_for=look_for)
            
            self.eval_and_reduce(to_accept_bool=accept_value)
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")


class AddVariable(ProcessingRow):
    """
    Class that adds certain variables to the data.
    """
    calc_instance = CalcInit
    allowed_instance = AllowedInit
    
    def __init__(self, row, name_list):
        super(AddVariable, self).__init__(row, name_list)
    
    @classmethod
    def set_instance(cls, **kwargs):
        """
        Sets the class instances used in the following.

        :param kwargs: class instances
        """
        if any("calc" in it for it in kwargs.keys()): cls.calc_instance = kwargs["calc_instance"]
        if any("allowed" in it for it in kwargs.keys()): cls.allowed_instance = kwargs["allowed_instance"]
    
    def pt(self, look_for="muon"):
        """
        Adds transverse pulse to the pandas series.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py"], type_variables=[float, float])
            px, py = found_array[0], found_array[1]
            
            pt = AddVariable.calc_instance.pt(px, py)
            self.add_raw_to_row(variable_name="pt", variable_array=pt)
            return self.row
        
        if look_for == "both":
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
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
    
    def eta(self, look_for="muon"):
        """
        Adds pseudorapidity to the pandas series.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py", "pz"],
                                          type_variables=[float, float, float])
            px, py, pz = found_array[0], found_array[1], found_array[2]
            
            eta = AddVariable.calc_instance.eta(px, py, pz)
            self.add_raw_to_row(variable_name="eta", variable_array=eta)
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "muon_pz",
                                                            "electron_px", "electron_py", "electron_pz"],
                                          type_variables=[float, float, float, float, float, float])
            px_mu, py_mu, pz_mu = found_array[0], found_array[1], found_array[2]
            px_el, py_el, pz_el = found_array[3], found_array[4], found_array[5]
            
            eta_mu = AddVariable.calc_instance.eta(px_mu, py_mu, pz_mu)
            eta_el = AddVariable.calc_instance.eta(px_el, py_el, pz_el)
            self.add_raw_to_row(variable_name="muon_eta", variable_array=eta_mu)
            self.add_raw_to_row(variable_name="electron_eta", variable_array=eta_el)
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
    
    def phi(self, look_for="muon"):
        """
        Adds phi angle to the pandas series.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py"],
                                          type_variables=[float, float])
            px, py = found_array[0], found_array[1]
            
            phi = AddVariable.calc_instance.phi(px, py)
            self.add_raw_to_row(variable_name="phi", variable_array=phi)
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "electron_px", "electron_py"],
                                          type_variables=[float, float, float, float])
            px_mu, py_mu = found_array[0], found_array[1]
            px_el, py_el = found_array[2], found_array[3]
            
            phi_mu = AddVariable.calc_instance.eta(px_mu, py_mu)
            phi_el = AddVariable.calc_instance.eta(px_el, py_el)
            self.add_raw_to_row(variable_name="muon_phi", variable_array=phi_mu)
            self.add_raw_to_row(variable_name="electron_phi", variable_array=phi_el)
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")


class Reconstruct(ProcessingRow):
    """
    Class that provides the reconstruction methods.
    """
    calc_instance = CalcInit
    allowed_instance = AllowedInit
    
    def __init__(self, row, name_list):
        super(Reconstruct, self).__init__(row, name_list)
    
    @classmethod
    def set_instance(cls, **kwargs):
        """
        Sets the class instances used in the following.

        :param kwargs: class instances
        """
        if any("calc" in it for it in kwargs.keys()): cls.calc_instance = kwargs["calc_instance"]
        if any("allowed" in it for it in kwargs.keys()): cls.allowed_instance = kwargs["allowed_instance"]
    
    def zz(self, look_for="muon"):
        """
        Reconstructs a Z boson pair and adds it and all the necessary sizes to the pandas series.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            
            found_array = self.search_for(search_variables=["eta", "px", "py", "pz", "energy",
                                                            "pt", "charge"],
                                          type_variables=[float, float, float, float, float, float, float])
            eta, px, py, pz, energy = found_array[0], found_array[1], found_array[2], found_array[3], found_array[4]
            pt, charge = found_array[5], found_array[6]
            
            phi = Reconstruct.calc_instance.phi(px, py)
            z1, z2, index_z1, index_z2 = Reconstruct.calc_instance.zz_and_index(eta, phi, pt, px, py, pz, charge,
                                                                                energy=energy,
                                                                                look_for=look_for)
            
            if z1 == 0:
                self.row["run"] = np.nan
                return self.row
            
            self.add_raw_to_row(variable_name="phi", variable_array=phi)
            self.add_raw_to_row(variable_name="z1_mass", variable_array=z1)
            self.add_raw_to_row(variable_name="z2_mass", variable_array=z2)
            self.add_raw_to_row(variable_name="z1_index", variable_array=index_z1)
            self.add_raw_to_row(variable_name="z2_index", variable_array=index_z2)
            
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_eta", "muon_px", "muon_py", "muon_pz",
                                                            "muon_energy", "muon_pt", "muon_charge",
                                                            "electron_eta", "electron_px", "electron_py", "electron_pz",
                                                            "electron_energy", "electron_pt", "electron_charge"],
                                          type_variables=[float, float, float, float, float, float, float,
                                                          float, float, float, float, float, float, float])
            eta_mu, px_mu, py_mu, pz_mu = found_array[0], found_array[1], found_array[2], found_array[3]
            energy_mu, pt_mu, charge_mu = found_array[4], found_array[5], found_array[6]
            eta_el, px_el, py_el, pz_el = found_array[7], found_array[8], found_array[9], found_array[10]
            energy_el, pt_el, charge_el = found_array[11], found_array[12], found_array[13]
            
            phi_mu = Reconstruct.calc_instance.phi(px_mu, py_mu)
            phi_el = Reconstruct.calc_instance.phi(px_el, py_el)
            z1, z2, index_z1, index_z2, z1_tag, z2_tag = Reconstruct.calc_instance.zz_and_index([eta_mu, eta_el],
                                                                                                [phi_mu, phi_el],
                                                                                                [pt_mu, pt_el],
                                                                                                [px_mu, px_el],
                                                                                                [py_mu, py_el],
                                                                                                [pz_mu, pz_el],
                                                                                                [charge_mu, charge_el],
                                                                                                look_for=look_for)
            
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
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
    
    def mass_4l_out_zz(self, look_for="muon"):
        """
        Reconstructs the four lepton invariant masses from the already reconstructed Z boson pair.

        :param look_for: str
                         "muon"; "electron" or "both"
        :return: pd.Series
        """
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py", "pz", "z1_index", "z2_index"],
                                          type_variables=[float, float, float, int, int])
            px, py, pz = found_array[0], found_array[1], found_array[2]
            z1_index, z2_index = found_array[3], found_array[4]
            
            mass_hi = self.calc_instance.mass_4l_out_zz(px, pz, py, [z1_index, z2_index], look_for=look_for)
            
            self.add_raw_to_row(variable_name="mass_4l", variable_array=mass_hi)
            
            if mass_hi == 0.0:
                self.row["run"] = np.nan
            
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "muon_pz",
                                                            "z1_index", "z2_index", "z1_tag", "z2_tag",
                                                            "electron_px", "electron_py", "electron_pz"],
                                          type_variables=[float, float, float, int, int, str, str, float, float, float])
            
            px_mu, py_mu, pz_mu = found_array[0], found_array[1], found_array[2]
            z1_index, z2_index, z1_tag, z2_tag = found_array[3], found_array[4], found_array[5], found_array[6]
            px_el, py_el, pz_el = found_array[7], found_array[8], found_array[9]
            
            mass_hi = self.calc_instance.mass_4l_out_zz([px_mu, px_el], [py_mu, py_el], [pz_mu, pz_el],
                                                        [z1_index, z2_index],
                                                        tag=[str(z1_tag[0]), str(z2_tag[0])], look_for=look_for)
            
            self.add_raw_to_row(variable_name="mass_4l", variable_array=mass_hi)
            
            if mass_hi == 0.0:
                self.row["run"] = np.nan
            
            return self.row
        
        else:
            raise TypeError("'look_for' can only be: 'muon', 'electron' or 'both'")
