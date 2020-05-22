# -*- coding: UTF-8 -*-
from itertools import combinations, product

import numpy as np
import swifter


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray

sw_temp_ = swifter


class AllowedInit(object):
    """
    Class that introduces certain cuts and thus restricts the leptons in the events.
    """
    
    a_calc_instance = None
    a_allowed_instance = None
    
    def __init__(self, *, calc_obj=None, allowed_obj=None):
        AllowedInit.a_allowed_instance = AllowedInit if allowed_obj is None else allowed_obj
        AllowedInit.a_calc_instance = CalcInit if calc_obj is None else calc_obj
        
    
    # Used allowed parameter in class
    va = {'rel_pf_iso': 0.35, 'misshit': 1.0,
          'pt_1': 20.0, 'pt_2': 10.0, 'pt_min_mu': 5.0, 'pt_min_el': 7.0,
          'max_eta_mu': 2.4, 'max_eta_el': 2.5,
          'type_mu': 'G', 'type_el': 'T',
          'min_2l_m': 4.0, 'min_4l_m': 70.0, 'max_4l_m': 180.0,
          'sip3d': 4.0, 'dxy': 0.5, 'dz': 1.0,
          'z1_min': 40.0, 'z1_max': 120.0, 'z2_min': 12.0, 'z2_max': 120.0, 'delta_r': 0.02}
    
    @staticmethod
    def delta_r(delta_r):
        """
        Checks if delta_r is smaller than the allowed value.

        :param delta_r: ndarray
                        1D array containing data with `float` type.
        :return: ndarray
                 1D array containing data with `bool` type.
        """
    
    @staticmethod
    def rel_pf_iso(rel_pf_iso):
        """
        Checks if rel_pf_iso is smaller than the allowed value.

        :param rel_pf_iso: ndarray
                           1D array containing data with `float` type.
        :return: ndarray
                 1D array containing data with `bool` type.
        """
    
    @staticmethod
    def misshits(misshits):
        """
        Checks if the minimum number of misshits was kept.

        :param misshits:
        :return:
        """
    
    @staticmethod
    def pt(p_t, look_for, coll_size=4):
        """
        Checks if the exact pedingun regarding pt is observed.
        (>20 GeV: >= 1; >10 GeV: >= 2; >Minimum pt: >= 4).

        :param p_t: ndarray
                    1D array containing data with `float` type.
        :param look_for: str
                         "muon"; "electron" or "both"
        :param coll_size: int
                          4 if look_for is not "both", 2 else
        :return: ndarray
                 1D array containing data with `bool` type.
        """
    
    @staticmethod
    def min_pt(p_t, look_for):
        """
        Checks whether the minimum transverse impulse of the individual leptons is maintained.
        A case distinction between electrons and muons must be made.

        :param p_t: ndarray
                    1D array containing data with `float` type.
        :param look_for: str
                         "muon"; "electron" or "both"
        :return: ndarray
                 1D array containing data with `bool` type.
        """
    
    @staticmethod
    def eta(eta, look_for):
        """
        Checks if the pseudorapidity of leptons is valid.

        :param eta: ndarray
                    1D array containing data with "float" type.
        :param look_for: str
                         "muon"; "electron" or "both"
        :return: ndarray
                 1D array containing data with "bool" type.
        """
    
    @staticmethod
    def lepton_type(typ, look_for):
        """
        Checks for the permitted classification of leptons.

        :param typ: ndarray
                    1D array containing data with "float" type.
        :param look_for: str
                         "muon"; "electron" or "both"
        :return: ndarray
                 1D array containing data with "bool" type.
        """
    
    @staticmethod
    def impact_param(sip3d, dxy, dz):
        """
        Checks if the impact parameters of the collision are valid and sorts out
        events that do not have a clear and equal collision point.

        :param sip3d: ndarray
                      1D array containing data with "float" type.
        :param dxy: ndarray
                    1D array containing data with "float" type.
        :param dz: ndarray
                   1D array containing data with "float" type.
        :return: ndarray
                 1D array containing data with "bool" type.
        """
    
    @staticmethod
    def zz(z1, z2):
        """
        Checks if the Z1 candidate and the Z2 candidate is within the allowed range.

        :param z1: float
        :param z2: float
        :return: bool
        """
    
    @staticmethod
    def invariant_mass(mass_list, number_of_leptons, look_for="muon"):
        """
        Checks if a mass (or a combination of masses) meets the condition satisfying that
        0 < min_mass <= mass <= max_mass.

        :param mass_list: ndarray
                          1D array containing data with "float" type.
        :param number_of_leptons: int
                                  4 if look_for is not "both", 2 else
        :param look_for: str
                         "muon"; "electron" or "both"
        :return: bool
        """
        if number_of_leptons == 4:
            accept_array = (mass_list > AllowedInit.a_allowed_instance.va["min_4l_m"]) & (mass_list < AllowedInit.a_allowed_instance.va["max_4l_m"])
            return True if np.sum(accept_array) >= 1 else False
        if number_of_leptons == 2:
            accept_array = (mass_list > AllowedInit.a_allowed_instance.va["min_2l_m"])
            if look_for != "both":
                return True if np.sum(accept_array) >= 2 else False
            if look_for == "both":
                return True if np.sum(accept_array) >= 1 else False


class CalcInit(object):
    """
    Class for the calculation of certain sizes that are used for
    the cuts or are essential for the reconstruction.
    """
    
    c_calc_instance = None
    c_allowed_instance = None
    
    def __init__(self, *, calc_obj=None, allowed_obj=None):
        CalcInit.c_allowed_instance = AllowedInit if allowed_obj is None else allowed_obj
        CalcInit.c_calc_instance = CalcInit if calc_obj is None else calc_obj
    
    @staticmethod
    def combined_charge(charge, combine_num):
        """
        Tests whether an electrically neutral charge combination is possible.

        :param charge: ndarray
                       1D array containing data with "int" type.
        :param combine_num: int
                            4 if look_for is not "both", 2 else
        :return: bool
        """
    
    @staticmethod
    def eta(px=None, py=None, pz=None, energy=None):
        """
        Calculates the pseudorapidity.
        Optional with or without energy.

        :param px: ndarray
                   1D array containing data with "float" type.
        :param py: ndarray
                    1D array containing data with "float" type.
        :param pz: ndarray
                   1D array containing data with "float" type.
        :param energy: ndarray
                       1D array containing data with "float" type.
        :return: ndarray
                 1D array containing data with "float" type.
        """
    
    @staticmethod
    def pt(px, py):
        """
        Calculates the transverse impulse.

        :param px: ndarray
                   1D array containing data with "float" type.
        :param py: ndarray
                   1D array containing data with "float" type.
        :return: ndarray
                 1D array containing data with "float" type.
        """
    
    @staticmethod
    def invariant_mass_square(px, py, pz, energy=None, eta=None, phi=None):
        """
        Calculates the square of the invariant mass.
        Optional with or without energy.
        Optionally with or without eta and phi.

        :param phi: ndarray
                    1D array containing data with "float" type.
        :param eta: ndarray
                    1D array containing data with "float" type.
        :param px: ndarray
                   1D array containing data with "float" type.
        :param py: ndarray
                   1D array containing data with "float" type.
        :param pz: ndarray
                   1D array containing data with "float" type.
        :param energy: ndarray
                       1D array containing data with "float" type.
        :return: ndarray
                 1D array containing data with "float" type.
        """
        return 1.0
    
    @staticmethod
    def phi(px, py):
        """
        Calculation of the angle phi.

        :param px: ndarray
                   1D array containing data with "float" type.
        :param py:  ndarray
                    1D array containing data with "float" type.
        :return: ndarray
                 1D array containing data with "float" type.
        """
    
    @staticmethod
    def delta_phi(phi1, phi2):
        """
        Calculation of the difference between two phi angles.

        :param phi1: ndarray
                     1D array containing data with "float" type.
        :param phi2: ndarray
                     1D array containing data with "float" type.
        :return: ndarray
                 1D array containing data with "float" type.
        """
    
    @staticmethod
    def delta_r(eta, phi):
        """
        Calculation of delta_r.

        :param eta: ndarray
                    1D array containing data with "float" type.
        :param phi: ndarray
                    1D array containing data with "float" type.
        :return: ndarray
                 1D array containing data with "float" type.
        """
    
    @staticmethod
    def mass_4l_out_zz(px, py, pz, index, energy=None, tag=None, look_for="muon"):
        """
        Calculation of the four lepton invariant masses from two given Z bosons.

        :param px: ndarray
                   1D or 2D array containing data with "float" type, depends on look_for
        :param py: ndarray
                   1D or 2D array containing data with "float" type, depends on look_for
        :param pz: ndarray
                   1D or 2D array containing data with "float" type, depends on look_for
        :param index: ndarray
                      1D or 2D array containing data with "int" type, depends on look_for
        :param energy: ndarray
                       1D or 2D array containing data with "float" type, depends on look_for
        :param tag: list
                    list of strings containing "m" and "e"
        :param look_for: str
                         "muon"; "electron" or "both"
        :return: float
        """
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            
            tot_idx = np.concatenate([index[0], index[1]])
            
            if energy is None:
                return np.sqrt(CalcInit.c_calc_instance.invariant_mass_square(px[tot_idx], py[tot_idx], pz[tot_idx]))
            if energy is not None:
                return np.sqrt(CalcInit.c_calc_instance.invariant_mass_square(px[tot_idx], py[tot_idx], pz[tot_idx], energy[tot_idx]))
        
        if look_for == "both":
            
            px_mu, px_el, py_mu, py_el, pz_mu, pz_el = px[0], px[1], py[0], py[1], pz[0], pz[1]
            
            idx_mu, idx_el = (index[0], index[1]) if (tag[0] == "e" and tag[1] == "m") else (index[1], index[0])
            
            if tag[0] == tag[1]: raise ValueError("z1_tag == z2_tag in 2mu2e channel!")
            
            if energy is None:
                return np.sqrt(CalcInit.c_calc_instance.invariant_mass_square(np.concatenate([px_mu[idx_mu], px_el[idx_el]]),
                                                                                     np.concatenate([py_mu[idx_mu], py_el[idx_el]]),
                                                                                     np.concatenate([pz_mu[idx_mu], pz_el[idx_el]])))
            if energy is not None:
                energy_mu, energy_el = energy[0], energy[1]
                return np.sqrt(CalcInit.c_calc_instance.invariant_mass_square(np.concatenate((px_mu[idx_mu], px_el[idx_el])),
                                                                                     np.concatenate((py_mu[idx_mu], py_el[idx_el])),
                                                                                     np.concatenate((pz_mu[idx_mu], pz_el[idx_el])),
                                                                                     np.concatenate((energy_mu[idx_mu], energy_el[idx_el]))))
    
    @staticmethod
    def zz_and_index(eta, phi, pt, px, py, pz, charge, energy=None, look_for="muon"):
        """
        Calculation of the Z boson pair used for the reconstruction
        of the four lepton invariant mass spectrum.

        :param eta: ndarray
                    1D or 2D array containing data with "float" type, depends on look_for
        :param phi: ndarray
                    1D or 2D array containing data with "float" type, depends on look_for
        :param pt: ndarray
                    1D or 2D array containing data with "float" type, depends on look_for
        :param px: ndarray
                    1D or 2D array containing data with "float" type, depends on look_for
        :param py: ndarray
                    1D or 2D array containing data with "float" type, depends on look_for
        :param pz: ndarray
                    1D or 2D array containing data with "float" type, depends on look_for
        :param charge: ndarray
                       1D array containing data with "int" type.
        :param energy: ndarray
                       1D or 2D array containing data with "float" type, depends on look_for
        :param look_for: str
                         "muon"; "electron" or "both"
        :return: tuple
                 (float, float, [int, int], [int int]) or (float, float, [int, int], [int int], str, str)
                 tuple containing Z1 mass, Z2 mass, corresponding lepton index and z1 and z2 tags if look_for = "both"
        """
        z_mass = 91.1876
        
        def create_z_list(eta_, phi_, pt_, px_, py_, pz_, q_, e_=None, tag=None, z_="z1", idx_=None):
            """


            :param eta_: ndarray
                         1D array containing data with "float" type.
            :param phi_: ndarray
                         1D array containing data with "float" type.
            :param pt_: ndarray
                        1D array containing data with "float" type.
            :param px_: ndarray
                        1D array containing data with "float" type.
            :param py_: ndarray
                        1D array containing data with "float" type.
            :param pz_: ndarray
                        1D array containing data with "float" type.
            :param q_: ndarray
                       1D array containing data with "int" type.
            :param e_: ndarray
                        1D array containing data with "float" type.
            :param tag: str
                        "m" or "e"
            :param z_: str
                       "z1" or "z2". Describes which Z is to be reconstructed
            :param idx_: ndarray
                         1D array containing data with "int" type. Corresponds to used index of leptons in event.
            :return: list
                     list of lists containing reconstructed Z boson mass, corresponding index tag and optional sum of pt.
            """
            m_list = []
            
            idx_ = np.linspace(0, len(pt_) - 1, len(pt_)) if idx_ is None else idx_
            
            for i, j in combinations(idx_, 2):
                i, j = int(i), int(j)
                if (q_[i] + q_[j]) == 0:
                    if CalcInit.c_allowed_instance.delta_r(CalcInit.c_calc_instance.delta_r(eta_[[i, j]], phi_[[i, j]])):
                        try:
                            inv_z_m = CalcInit.c_calc_instance.invariant_mass_square(px_[[i, j]], py_[[i, j]], pz_[[i, j]], e_[[i, j]])
                        except TypeError:
                            inv_z_m = CalcInit.c_calc_instance.invariant_mass_square(px_[[i, j]], py_[[i, j]], pz_[[i, j]])
                        
                        if inv_z_m > 4.0:
                            if z_ == "z1":
                                m_list.append([np.sqrt(inv_z_m), np.array([i, j]), tag])
                            if z_ == "z2":
                                m_list.append([np.sqrt(inv_z_m), np.array([i, j]), tag, np.sum(pt_[[i, j]])])
            return m_list
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            
            try:
                inv_z1_list = create_z_list(z_="z1", eta_=eta, phi_=phi, pt_=pt, px_=px, py_=py, pz_=pz, q_=charge,
                                            e_=energy)
                inv_z1 = min(inv_z1_list, key=lambda x: np.abs(x[0] - z_mass))
                
                index = np.delete(np.linspace(0, len(pt) - 1, len(pt)), inv_z1[1])
                inv_z2_list = create_z_list(z_="z2", idx_=index, eta_=eta, phi_=phi, pt_=pt, px_=px, py_=py, pz_=pz,
                                            q_=charge, e_=energy)
                inv_z2 = max(inv_z2_list, key=lambda x: x[3])
                
                if inv_z2[0] > inv_z1[0]: inv_z1, inv_z2 = inv_z2, inv_z1
                
                valid_pt = CalcInit.c_allowed_instance.pt(pt[np.concatenate([inv_z1[1], inv_z2[1]])], look_for=look_for)
                valid_zz = CalcInit.c_allowed_instance.zz(inv_z1[0], inv_z2[0])
                
                if valid_pt and valid_zz:
                    return [inv_z1[0]], [inv_z2[0]], inv_z1[1], inv_z2[1]
                else:
                    raise ValueError
            
            except ValueError:
                return 0, 0, 0, 0
        
        if look_for == "both":
            try:
                try:
                    inv_z1_mu = create_z_list(z_="z1", tag="m", eta_=eta[0], phi_=phi[0], pt_=pt[0], px_=px[0],
                                              py_=py[0], pz_=pz[0], q_=charge[0], e_=energy[0])
                    inv_z1_el = create_z_list(z_="z1", tag="e", eta_=eta[1], phi_=phi[1], pt_=pt[1], px_=px[1],
                                              py_=py[1], pz_=pz[1], q_=charge[1], e_=energy[1])
                except TypeError:
                    inv_z1_mu = create_z_list(z_="z1", tag="m", eta_=eta[0], phi_=phi[0], pt_=pt[0], px_=px[0],
                                              py_=py[0], pz_=pz[0], q_=charge[0])
                    inv_z1_el = create_z_list(z_="z1", tag="e", eta_=eta[1], phi_=phi[1], pt_=pt[1], px_=px[1],
                                              py_=py[1], pz_=pz[1], q_=charge[1])
                
                inv_z1_list = inv_z1_mu + inv_z1_el
                inv_z1 = min(inv_z1_list, key=lambda x: np.abs(x[0] - z_mass))
                
                inv_z2_list = []
                if inv_z1[2] == "m":
                    try:
                        inv_z2_list = create_z_list(z_="z2", tag="e", eta_=eta[1], phi_=phi[1], pt_=pt[1], px_=px[1],
                                                    py_=py[1], pz_=pz[1], q_=charge[1], e_=energy[1])
                    except TypeError:
                        inv_z2_list = create_z_list(z_="z2", tag="e", eta_=eta[1], phi_=phi[1], pt_=pt[1], px_=px[1],
                                                    py_=py[1], pz_=pz[1], q_=charge[1])
                if inv_z1[2] == "e":
                    try:
                        inv_z2_list = create_z_list(z_="z2", tag="m", eta_=eta[0], phi_=phi[0], pt_=pt[0], px_=px[0],
                                                    py_=py[0], pz_=pz[0], q_=charge[0], e_=energy[0])
                    except TypeError:
                        inv_z2_list = create_z_list(z_="z2", tag="m", eta_=eta[0], phi_=phi[0], pt_=pt[0], px_=px[0],
                                                    py_=py[0], pz_=pz[0], q_=charge[0])
                inv_z2 = max(inv_z2_list, key=lambda x: x[3])
                
                if inv_z2[0] > inv_z1[0]: inv_z1, inv_z2 = inv_z2, inv_z1
                
                mu_idx, el_idx = 0, 1
                if inv_z2[2] == "m":
                    mu_idx, el_idx = el_idx, mu_idx
                
                used_pt = np.concatenate([pt[mu_idx][inv_z1[1]], pt[el_idx][inv_z2[1]]])
                valid_pt = (np.sum(used_pt > 20) >= 1) and (np.sum(used_pt > 10) >= 2)
                valid_zz = CalcInit.c_allowed_instance.zz(inv_z1[0], inv_z2[0])
                
                if valid_pt and valid_zz:
                    return inv_z1[0], inv_z2[0], inv_z1[1], inv_z2[1], inv_z1[2], inv_z2[2]
                else:
                    raise ValueError
            
            except ValueError:
                return 0, 0, 0, 0, "nan", "nan"
    
    @staticmethod
    def possible_invariant_masses(px, py, pz, charge, energy=None, number_of_leptons=2, look_for_both=False):
        """
        Calculation of all possible invariant masses for a given number of leptons.

        :param px: ndarray
                   1D array containing data with "float" type.
        :param py: ndarray
                   1D array containing data with "float" type.
        :param pz: ndarray
                   1D array containing data with "float" type.
        :param charge: ndarray
                       1D array containing data with "int" type.
        :param energy: ndarray
                   1D array containing data with "float" type.
        :param number_of_leptons: int
                                  4 if look_for is not "both", 2 else
        :param look_for_both: bool
        :return: ndarray
                 1D array containing data with "float" type.
        """
        combined_mass = []
        if number_of_leptons == 2:
            index = np.linspace(0, len(px) - 1, len(px))
            for p in combinations(index, 2):
                p = np.array(p, dtype=int)
                inv_square_mass = 0
                if (charge[p[0]] + charge[p[1]]) == 0:
                    if energy is None:
                        inv_square_mass = CalcInit.c_calc_instance.invariant_mass_square(px[p], py[p], pz[p])
                    if energy is not None:
                        inv_square_mass = CalcInit.c_calc_instance.invariant_mass_square(px[p], py[p], pz[p], energy[p])
                    if inv_square_mass >= 0:
                        combined_mass.append(np.sqrt(inv_square_mass))
                    else:
                        combined_mass.append(0.0)
                else:
                    combined_mass.append(0.0)
            
            return np.array(combined_mass)
        
        if number_of_leptons == 4:
            if not look_for_both:
                index = np.linspace(0, len(px) - 1, len(px))
                for p in combinations(index, 4):
                    p = np.array(p, dtype=int)
                    inv_square_mass = 0
                    if (charge[p[0]] + charge[p[1]] + charge[p[2]] + charge[p[3]]) == 0:
                        if energy is not None:
                            inv_square_mass = CalcInit.c_calc_instance.invariant_mass_square(np.array(px)[p], np.array(py)[p],
                                                                             np.array(pz)[p], np.array(energy)[p])
                        if energy is None:
                            inv_square_mass = CalcInit.c_calc_instance.invariant_mass_square(np.array(px)[p], np.array(py)[p],
                                                                             np.array(pz)[p])
                        if inv_square_mass >= 0:
                            combined_mass.append(np.sqrt(inv_square_mass))
                        else:
                            combined_mass.append(0.0)
                    else:
                        combined_mass.append(0.0)
                
                return np.array(combined_mass)
            
            if look_for_both:
                px_mu, py_mu, pz_mu, px_el, py_el, pz_el = px[0], py[0], pz[0], px[1], py[1], pz[1]
                charge_mu, charge_el = charge[0], charge[1]
                
                index_mu = np.linspace(0, len(px_mu) - 1, len(px_mu))
                index_el = np.linspace(0, len(px_el) - 1, len(px_el))
                
                mu_comb, el_comb = [], []
                for p1 in combinations(index_mu, 2):
                    mu_comb.append(np.array(p1, dtype=int))
                
                for p2 in combinations(index_el, 2):
                    el_comb.append(np.array(p2, dtype=int))
                
                two_comb_list = [np.array(mu_comb), np.array(el_comb)]
                
                for two_comb in product(*two_comb_list):
                    tc = np.array(two_comb)
                    if ((charge_mu[tc[0][0]] + charge_mu[tc[0][1]]) == 0) and (
                            (charge_el[tc[1][0]] + charge_el[tc[1][1]]) == 0):
                        inv_square_mass = 0
                        if energy is not None:
                            energy_mu, energy_el = energy[0], energy[1]
                            inv_square_mass = CalcInit.c_calc_instance.invariant_mass_square(np.concatenate((px_mu[tc[0]], px_el[tc[1]])),
                                                                                    np.concatenate((py_mu[tc[0]], py_el[tc[1]])),
                                                                                    np.concatenate((pz_mu[tc[0]], pz_el[tc[1]])),
                                                                                    energy=np.concatenate((energy_mu[tc[0]], energy_el[tc[1]])))
                        if energy is None:
                            inv_square_mass = CalcInit.c_calc_instance.invariant_mass_square(np.concatenate((px_mu[tc[0]], px_el[tc[1]])),
                                                                                    np.concatenate((py_mu[tc[0]], py_el[tc[1]])),
                                                                                    np.concatenate((pz_mu[tc[0]], pz_el[tc[1]])))
                        
                        if inv_square_mass >= 0:
                            combined_mass.append(np.sqrt(inv_square_mass))
                        else:
                            combined_mass.append(0.0)
                    else:
                        combined_mass.append(0.0)
                
                return np.array(combined_mass)
