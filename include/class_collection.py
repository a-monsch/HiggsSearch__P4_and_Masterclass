# -*- coding: UTF-8 -*-
import ast
import itertools
import math
import multiprocessing as mp
import os
import sys
import warnings
from collections import namedtuple
from functools import partial
from itertools import product, combinations

import kafe2 as K2
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as sci
import scipy.special as scsp
import scipy.stats as scst
import yaml
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pandas.util.testing import assert_frame_equal

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


# Increase the default precession
# -----------------------------------------------------


def array32(*args, **kwargs):
    if 'dtype' not in kwargs:
        kwargs['dtype'] = "longdouble"  # 'float64'
    _oldarray(*args, **kwargs)


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


array32 = partial(np.array, dtype=np.float32)
_oldarray = np.array
np.array = _oldarray


# -----------------------------------------------------

# TODO: optimize paths via os.path. ... in specific classes!

class Data(object):
    """
    Class to convert the saved "string-only" dataframe to a "array_only" dataframe.
    Might be useful for a Filter that takes an array like dataframe and returns a
    array like dataframe. (For more see Legacy.py)
    """
    
    def __init__(self, class_input):
        if type(class_input).__name__ == "str":
            print("Load " + class_input)
            self.data = pd.read_csv(class_input, sep=";")
            print("Loading successful completed")
        if type(class_input).__name__ != "str":
            self.data = class_input
        self.names = list(self.data)
    
    @staticmethod
    def __convert_row_wise_to_array(row, names_list):
        
        """
        Function to convert a given row of strings with commas in it to a numpy array or a single value

        :param row: actual row (from pandas apply)
        :param names_list: list of column names of the dataset
        :return: new row of numpy arrays
        """
        
        pair = list(zip(names_list, row))
        if pair[0][0] == pair[0][1]:
            return row
        for i in range(len(pair)):
            if ("event" in pair[i][0]) or ("run" in pair[i][0]):
                row[i] = int(pair[i][1])
                continue
            if "type" in pair[i][0]:
                row[i] = np.array(pair[i][1].split(","), dtype="<U1")
                continue
            if ("charge" in pair[i][0]) or ("misshits" in pair[i][0]):
                row[i] = np.array(pair[i][1].split(","), dtype=int)
                continue
            else:
                row[i] = np.array(pair[i][1].split(","), dtype=float)
        
        return row
    
    @staticmethod
    def __convert_row_wise_to_string(row, names_list):
        
        """
        Function to convert a given  row of numpy arrays to strings with comma in it.

        :param row: actual row (from pandas apply)
        :param names_list: list of column names of the dataset
        :return: new row of of strings
        """
        
        pair = list(zip(names_list, row))
        if pair[0][0] == pair[0][1]:
            return row
        for i in range(len(pair)):
            if ("event" in pair[i][0]) or ("run" in pair[i][0]):
                row[i] = str(pair[i][1])
                continue
            else:
                temp_var_array = ""
                for j in range(len(pair[i][1])):
                    temp_var_array += str(pair[j][1]) + ","
                row[i] = temp_var_array[:-1]
                continue
        return row
    
    def get_array_like(self):
        
        """
        Actual callable function to convert a dataframe of strings to dataframe of numpy arrays.

        :return: dataframe contained with numpy arrays only
        """
        
        print("Begin conversion")
        self.data = self.data.apply(lambda x: self.__convert_row_wise_to_array(x, self.names), axis=1)
        print("Conversion complete")
        return self.data
    
    def get_string_like(self):
        
        """
        Actual callable function to convert a dataframe of numpy arrays to dataframe of strings.

        :return: dataframe contained strings only
        """
        
        print("Begin conversion")
        self.data = self.data.apply(lambda x: self.__convert_row_wise_to_string(x, self.names), axis=1)
        print("Conversion complete")
        return self.data


class Allowed(object):
    """
    Class that calculates the Allowed conditions on specific variables
    """
    
    # Used allowed parameter in class
    va = {"rel_pf_iso": 0.35, "misshit": 1.0,
          "pt_1": 20.0, "pt_2": 10.0, "pt_min_mu": 5.0, "pt_min_el": 7.0,
          "max_eta_mu": 2.4, "max_eta_el": 2.5,
          "type_mu": "G", "type_el": "T",
          "min_2l_m": 4.0, "min_4l_m": 70.0, "max_4l_m": 180.0,
          "sip3d": 4.0, "dxy": 0.5, "dz": 1.0,
          "z1_min": 40.0, "z1_max": 120.0, "z2_min": 12.0, "z2_max": 120.0,
          "delta_r": 0.02}
    
    def __init__(self):
        self.init = True
    
    @staticmethod
    def delta_r(delta_r):
        
        """
        Function to check if delta_r value is smaller than a certain value

        :param delta_r: float
        :return: True/False
        """
        
        return delta_r > Allowed.va["delta_r"]
    
    @staticmethod
    def rel_pf_iso(rel_pf_iso):
        
        """
        Function to return a boolean numpy array that is True when the corresponding input variable satisfy
        the following condition(s):

        - relative particle flow isolation is smaller than a certain value.

        :param rel_pf_iso: numpy array of relative particle flow isolation
        :return: boolean numpy array.

        True == Value will be keep in the following
        False == Value will be removed in the following
        """
        
        return rel_pf_iso < Allowed.va["rel_pf_iso"]
    
    @staticmethod
    def misshits(misshits):
        
        """
        Function to return a boolean numpy array that is True when the corresponding input variable satisfy
        the following condition(s):

        - misshits are below a certain value.

        :param misshits: numpy array of misshits
        :return: boolean numpy array.

        True == Value will be keep in the following
        False == Value will be removed in the following
        """
        
        return misshits <= Allowed.va["misshit"]
    
    @staticmethod
    def pt(p_t, look_for, coll_size=4):
        
        """
        Function to return a boolean numpy array that is True when the corresponding input variable satisfy
        the following condition(s):

        - transverse impulses of particles satisfy the exact condition.

        :param p_t: numpy array of transverse impulse
        :param look_for: "muon" of "electron"
        :param coll_size: number of viewed particles; "muon" and "electron" -> 2, else -> 4
        :return: boolean numpy array.

        True == Value will be keep in the following
        False == Value will be removed in the following
        """
        if coll_size == 4:
            if look_for == "muon":
                accept_value = (np.sum((p_t > Allowed.va["pt_1"])) >= 1) & (np.sum((p_t > Allowed.va["pt_2"])) >= 2) & (
                        np.sum((p_t > Allowed.va["pt_min_mu"])) >= 4)
                return accept_value
            if look_for == "electron":
                accept_value = (np.sum((p_t > Allowed.va["pt_1"])) >= 1) & (np.sum((p_t > Allowed.va["pt_2"])) >= 2) & (
                        np.sum((p_t > Allowed.va["pt_min_el"])) >= 4)
                return accept_value
        if coll_size == 2:
            if look_for == "muon":
                accept_value = ((np.sum((p_t > Allowed.va["pt_1"])) >= 1) & (
                        np.sum((p_t > Allowed.va["pt_2"])) >= 2)) | (np.sum((p_t > Allowed.va["pt_min_mu"])) >= 2)
                return accept_value
            if look_for == "electron":
                accept_value = ((np.sum((p_t > Allowed.va["pt_1"])) >= 1) & (
                        np.sum((p_t > Allowed.va["pt_2"])) >= 2)) | (np.sum((p_t > Allowed.va["pt_min_el"])) >= 2)
                return accept_value
    
    @staticmethod
    def min_pt(p_t, look_for):
        
        """
        Function to return a boolean numpy array that is True when the corresponding input variable satisfy
        the following condition(s):

        - transverse impulses of particles satisfy the minimum condition.

        :param p_t: numpy array of transverse impulse
        :param look_for: "muon" or "electron"
        :return: boolean numpy array.

        True == Value will be keep in the following
        False == Value will be removed in the following
        """
        
        if look_for == "muon":
            accept_array = (p_t > Allowed.va["pt_min_mu"])
            return accept_array
        if look_for == "electron":
            accept_array = (p_t > Allowed.va["pt_min_el"])
            return accept_array
    
    @staticmethod
    def eta(eta, look_for):
        
        """
        Function to return a boolean numpy array that is True when the corresponding input variable satisfy
        the following condition(s):

        - eta of particles satisfy the given condition.

        :param eta: numpy array of transverse impulse
        :param look_for: "muon" or "electron"
        :return: boolean numpy array.

        True == Value will be keep in the following
        False == Value will be removed in the following
        """
        
        if look_for == "muon":
            accept_array = (np.abs(eta) < Allowed.va["max_eta_mu"])
            return accept_array
        if look_for == "electron":
            accept_array = (np.abs(eta) < 1.479) | ((np.abs(eta) > 1.653) & (np.abs(eta) < Allowed.va["max_eta_el"]))
            # accept_array = (abs(eta) < Allowed.va["max_eta_el"])
            return accept_array
    
    @staticmethod
    def type(typ, look_for):
        
        """
        Function to return a boolean numpy array that is True when the corresponding input variable satisfy
        the following condition(s):

        - the particle is labeled as a specific type

        :param typ: array of strings of particle type
        :param look_for: "muon" or "electron"
        :return: boolean numpy array.

        True == Value will be keep in the following
        False == Value will be removed in the following
        """
        
        if look_for == "muon":
            accept_array = (typ == Allowed.va["type_mu"])
            return np.array(accept_array)
        if look_for == "electron":
            accept_array = (typ == Allowed.va["type_el"])
            return np.array(accept_array)
    
    @staticmethod
    def invariant_mass(mass_list, number_of_leptons, look_for="muon"):
        
        """
        Function to check if the mass (or combination of masses) satisfy the conditions of:
        mass_min, mass_max and mass > 0.

        :param mass_list: numpy array of possible masses
        :param number_of_leptons: number of leptons to calculate the invariant mass (4 or 2)
        :param look_for: "muon", "electron" or "both", only important for number_of_leptons == 2
        :return: True or False
        """
        
        if number_of_leptons == 4:
            accept_array = (mass_list > Allowed.va["min_4l_m"]) & (mass_list < Allowed.va["max_4l_m"])
            if np.sum(accept_array) >= 1:
                return True
            else:
                return False
        if number_of_leptons == 2:
            accept_array = (mass_list > Allowed.va["min_2l_m"])
            if look_for != "both":
                if np.sum(accept_array) >= 2:
                    return True
            if look_for == "both":
                if np.sum(accept_array) >= 1:
                    return True
            else:
                return False
    
    @staticmethod
    def impact_param(sip3d, dxy, dz):
        
        """
        Function to return a boolean numpy array that is True when the corresponding input variable satisfy
        the following condition(s):

        - the particle satisfy impact parameter

        :param sip3d: Impact parameter
        :param dxy: error in xy direction
        :param dz: error in z direction
        :return:  boolean array

        True == Value will be keep in the following
        False == Value will be removed in the following
        """
        
        accept_array = (sip3d < Allowed.va["sip3d"]) & (dxy < Allowed.va["dxy"]) & (dz < Allowed.va["dz"])
        return accept_array
    
    @staticmethod
    def zz(z1, z2):
        
        """
        Function to check the conditions on Z1 and Z2

        :param z1: single float!
        :param z2: single float!
        :return: True/False


        This function is implemented in Calc.zz_and_index. Therefore this function is not taking arrays as
        input. If this function moves to a place where it has to handle array like input it should not be a problem.
        """
        
        return (Allowed.va["z2_min"] < z2 < Allowed.va["z2_max"]) & (Allowed.va["z1_min"] < z1 < Allowed.va["z1_max"])


class Calc(object):
    """
    Class that provides the functions that calculates the needed variables in FilterStr or Reconstruction
    """
    
    def __init__(self):
        self.init = True
    
    @staticmethod
    def combined_charge(charge, combine_num):
        
        """
        Function to check if charge = 0 is possible out of given array with length >= 4 or length >=2.

        :param charge: numpy array of ints corresponding to to the particle charge
        :param combine_num: number of particles that should check the combined charge.
               "muon" or "electron" -> 4; "muon" and "electron" -> 2
        :return: 0 if it is possible to get a combined charge of zero, 1 else.
        """
        
        accept_charge = 1
        for p in combinations(charge, combine_num):
            accept_charge = np.sum(p)
            if accept_charge == 0:
                break
        return accept_charge
    
    @staticmethod
    def eta(px=None, py=None, pz=None, energy=None):
        """
        Function to calculate eta.

        :param px: numpy array of impulses in x direction
        :param py: numpy array of impulses in y direction
        :param pz: numpy array of impulses in z direction
        :param energy: numpy array of energies for possible other calculation
        :return: numpy array of etas
        """
        
        if type(energy).__name__ == "NoneType":
            p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
            return 0.5 * np.log((p + pz) / (p - pz))
        if type(energy).__name__ != "NoneType":
            return 0.5 * np.log((energy + pz) / (energy - pz))
    
    @staticmethod
    def pt(px, py):
        
        """
        Function to calculate the transverse impulses.

        :param px: numpy array of impulses in x direction
        :param py: numpy array of impulses in y direction
        :return: numpy array of transverse impulses
        """
        
        return np.sqrt(px ** 2 + py ** 2)
    
    @staticmethod
    def invariant_mass_square(px, py, pz, energy=None):
        
        """
        Function to calculate the square invariant mass.

        :param px: numpy array of impulses in x direction
        :param py: numpy array of impulses in y direction
        :param pz: numpy array of impulses in z direction
        :param energy: optional numpy array of corresponding energies (for an alternative calculation)
        :return: one square invariant mass of corresponding particles; float
        """
        
        if type(energy).__name__ == "NoneType":
            w = np.sum(np.sqrt(np.array(px) ** 2 + np.array(py) ** 2 + np.array(pz) ** 2))
            inv_m_square = w ** 2 - np.sum(px) ** 2 - np.sum(py) ** 2 - np.sum(pz) ** 2
            return inv_m_square
        else:
            e_tot = np.sum(energy)
            px_tot, py_tot, pz_tot = np.sum(px), np.sum(py), np.sum(pz)
            inv_m_square = e_tot ** 2 - (px_tot ** 2 + py_tot ** 2 + pz_tot ** 2)
            return inv_m_square
    
    @staticmethod
    def possible_invariant_masses(px, py, pz, charge, energy=None, number_of_leptons=2, look_for_both=False):
        
        """
        Function to calculate all possible invariant masses on a given lepton number.

        :param px: numpy array of impulses in x direction
        :param py: numpy array of impulses in y direction
        :param pz: numpy array of impulses in z direction
        :param charge: numpy array of corresponding charges
        :param energy: optional numpy array of corresponding energies (for an alternative calculation)
        :param number_of_leptons: how many leptons contribute to the invariant mass?
        :param look_for_both: affect only number_of_leptons=4, is used in the 2e2mu channel
        :return: an array of possible invariant masses, contains zeros if a calculation is not possible
        """
        
        combined_mass = []
        if number_of_leptons == 2:
            index = np.linspace(0, len(px) - 1, len(px))
            for p in combinations(index, 2):
                p = np.array(p, dtype=int)
                inv_square_mass = 0
                if (charge[p[0]] + charge[p[1]]) == 0:
                    if type(energy).__name__ == "NoneType":
                        inv_square_mass = Calc.invariant_mass_square(px[p], py[p], pz[p])
                    if type(energy).__name__ != "NoneType":
                        inv_square_mass = Calc.invariant_mass_square(px[p], py[p], pz[p], energy[p])
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
                        if type(energy).__name__ != "NoneType":
                            inv_square_mass = Calc.invariant_mass_square(np.array(px)[p], np.array(py)[p],
                                                                         np.array(pz)[p], np.array(energy)[p])
                        if type(energy).__name__ == "NoneType":
                            inv_square_mass = Calc.invariant_mass_square(np.array(px)[p], np.array(py)[p],
                                                                         np.array(pz)[p])
                        if inv_square_mass >= 0:
                            combined_mass.append(np.sqrt(inv_square_mass))
                        else:
                            combined_mass.append(0.0)
                    else:
                        combined_mass.append(0.0)
                
                return np.array(combined_mass)
            
            if look_for_both:
                px_mu, py_mu, pz_mu = px[0], py[0], pz[0]
                px_el, py_el, pz_el = px[1], py[1], pz[1]
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
                        if type(energy).__name__ != "NoneType":
                            energy_mu, energy_el = energy[0], energy[1]
                            inv_square_mass = Calc.invariant_mass_square(np.concatenate((px_mu[tc[0]], px_el[tc[1]])),
                                                                         np.concatenate((py_mu[tc[0]], py_el[tc[1]])),
                                                                         np.concatenate((pz_mu[tc[0]], pz_el[tc[1]])),
                                                                         energy=np.concatenate(
                                                                             (energy_mu[tc[0]], energy_el[tc[1]])))
                        if type(energy).__name__ == "NoneType":
                            inv_square_mass = Calc.invariant_mass_square(np.concatenate((px_mu[tc[0]], px_el[tc[1]])),
                                                                         np.concatenate((py_mu[tc[0]], py_el[tc[1]])),
                                                                         np.concatenate((pz_mu[tc[0]], pz_el[tc[1]])))
                        
                        if inv_square_mass >= 0:
                            combined_mass.append(np.sqrt(inv_square_mass))
                        else:
                            combined_mass.append(0.0)
                    else:
                        combined_mass.append(0.0)
                
                return np.array(combined_mass)
    
    @staticmethod
    def phi(px, py):
        
        """
        Function to calculate phi

        :param px: numpy array of the impulse in x direction
        :param py: numpy array of the impulse in x direction
        :return: numpy array of corrssponding phi

        It is possible to recreate np.arctan2 by yourslef... np.arctan2 is cleaner!

        """
        
        return np.arctan2(np.array(py), np.array(px))
    
    @staticmethod
    def delta_r(eta, phi):
        
        """
        Function to calculate the Isolation between two particles.

        :param eta: numpy array of etas
        :param phi: numpy array of phis
        :return: True or False
        """
        
        delta_eta = eta[0] - eta[1]
        delta_phi = np.arctan2(np.sin(phi[0] - phi[1]), np.cos(phi[0] - phi[1]))
        
        pass_delta_r = np.sqrt(delta_eta ** 2 + delta_phi ** 2)
        
        return pass_delta_r
    
    @staticmethod
    def zz_and_index(eta, phi, pt, px, py, pz, charge, energy=None, look_for="muon"):
        
        z_mass = 91.1876
        
        def create_z_list(eta_, phi_, pt_, px_, py_, pz_, q_, e_=None, tag=None, z_="z1", idx_=None):
            m_list = []
            if type(idx_).__name__ == "NoneType":
                idx_ = np.linspace(0, len(pt_) - 1, len(pt_))
            
            for i, j in combinations(idx_, 2):
                i, j = int(i), int(j)
                if (q_[i] + q_[j]) == 0:
                    if Allowed.delta_r(Calc.delta_r(eta_[[i, j]], phi_[[i, j]])):
                        try:
                            inv_z_m = Calc.invariant_mass_square(px_[[i, j]], py_[[i, j]], pz_[[i, j]], e_[[i, j]])
                        except TypeError:
                            inv_z_m = Calc.invariant_mass_square(px_[[i, j]], py_[[i, j]], pz_[[i, j]])
                        
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
                
                valid_pt = Allowed.pt(pt[np.concatenate([inv_z1[1], inv_z2[1]])], look_for=look_for)
                valid_zz = Allowed.zz(inv_z1[0], inv_z2[0])
                
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
                valid_zz = Allowed.zz(inv_z1[0], inv_z2[0])
                
                if valid_pt and valid_zz:
                    return inv_z1[0], inv_z2[0], inv_z1[1], inv_z2[1], inv_z1[2], inv_z2[2]
                else:
                    raise ValueError
            
            except ValueError:
                return 0, 0, 0, 0, "nan", "nan"
    
    @staticmethod
    def delta_phi(phi1, phi2):
        
        """
        Function to calculate delta_phi out of two given phi's

        :param phi1: float
        :param phi2: float
        :return: float
        """
        
        del_phi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
        return del_phi
    
    @staticmethod
    def mass_4l_out_zz(px, py, pz, index, energy=None, tag=None, look_for="muon"):
        
        """
        Function to calculate the four lepton invariant mass out of two given Z bosons

        :param px: numpy array or list of numpy arrays of impulse in x direction
        :param py: numpy array or list of numpy arrays of impulse in y direction
        :param pz: numpy array or list of numpy arrays of impulse in z direction
        :param index: list containing two lists that corresponds to the particle that were used to reconstruct ZZ
        :param energy: numpy array or list of numpy arrays of energy
        :param tag: list containing the tag that corresponds to Z1 and one to Z2 (only used if look_for='both')
        :param look_for: 'muon', 'electron', or 'both'
        :return: four lepton invariant mass (float)
        """
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            
            tot_idx = np.concatenate([index[0], index[1]])
            
            mass_4l_pass = 0
            if type(energy).__name__ == "NoneType":
                mass_4l_pass = Calc.invariant_mass_square(px[tot_idx], py[tot_idx], pz[tot_idx])
            if type(energy).__name__ != "NoneType":
                mass_4l_pass = Calc.invariant_mass_square(px[tot_idx], py[tot_idx], pz[tot_idx], energy[tot_idx])
            mass_4l_pass = np.sqrt(mass_4l_pass)
            
            return mass_4l_pass
        
        if look_for == "both":
            
            px_mu, px_el, py_mu, py_el, pz_mu, pz_el = px[0], px[1], py[0], py[1], pz[0], pz[1]
            
            idx_mu, idx_el = index[0], index[1]
            if tag[0] == "e" and tag[1] == "m":
                idx_mu, idx_el = index[1], index[0]
            if tag[0] == tag[1]:
                sys.exit("Error: z1_tag == z2_tag")
            
            if type(energy).__name__ == "NoneType":
                mass_4l_pass = Calc.invariant_mass_square(np.concatenate([px_mu[idx_mu], px_el[idx_el]]),
                                                          np.concatenate([py_mu[idx_mu], py_el[idx_el]]),
                                                          np.concatenate([pz_mu[idx_mu], pz_el[idx_el]]))
            else:
                energy_mu, energy_el = energy[0], energy[1]
                mass_4l_pass = Calc.invariant_mass_square(np.concatenate((px_mu[idx_mu], px_el[idx_el])),
                                                          np.concatenate((py_mu[idx_mu], py_el[idx_el])),
                                                          np.concatenate((pz_mu[idx_mu], pz_el[idx_el])),
                                                          np.concatenate((energy_mu[idx_mu], energy_el[idx_el])))
            mass_4l_pass = np.sqrt(mass_4l_pass)
            
            return mass_4l_pass


class ProcessingRow(object):
    
    def __init__(self, row, name_list):
        self.row = row
        self.names = name_list
        self.pairs = list(zip(name_list, row))
        self.ignore = ["run", "event", "luminosity"]
    
    @staticmethod
    def __to_str(variable):
        try:
            return ",".join([str(item) for item in variable]).replace(" ", "")
        except TypeError:
            return str(variable)
    
    def __to_str_from_str(self, variable, accept):
        try:
            return self.__to_str(np.array(ast.literal_eval("[{}]".format(variable)))[accept])
        except IndexError or ValueError:
            return self.__to_str(np.array(variable.split(","))[accept])
    
    def if_column(self):
        if self.pairs[0][0] == self.pairs[0][1]:
            return True
        return False
    
    def add_raw_to_row(self, variable_name, variable_array):
        if variable_name not in self.names:
            self.row[variable_name] = self.__to_str(variable_array)
            self.names = list(self.row.to_frame().T)
    
    def search_for(self, search_variables, type_variables):
        found_array = []
        for i, var in enumerate(search_variables):
            for pair in self.pairs:
                if var in pair[0]:
                    found_array.append(np.array(pair[1].split(","), dtype=type_variables[i]))
        return found_array
    
    def reduce_row(self, accept_array):
        if len(accept_array) == 2:
            mu_accept, el_accept = accept_array[0], accept_array[1]
            mu_names = [item for item in self.names if "muon" in item]
            el_names = [item for item in self.names if "electron" in item or "misshits" in item]
            if np.sum(mu_accept) != len(mu_accept):
                self.row[mu_names] = self.row[mu_names].apply(lambda x: self.__to_str_from_str(x, mu_accept))
            if np.sum(el_accept) != len(el_accept):
                self.row[el_names] = self.row[el_names].apply(lambda x: self.__to_str_from_str(x, el_accept))
        if len(accept_array) == 1:
            accept = accept_array[0]
            names = [it for it in self.names if it not in self.ignore]
            if np.sum(accept) != len(accept):
                self.row[names] = self.row[names].apply(lambda x: self.__to_str_from_str(x, accept))
    
    def eval_on_length(self, accept_array):
        if len(accept_array) == 1:
            if np.sum(accept_array[0]) < 4:
                self.row["run"] = np.nan
                return True
            if np.sum(accept_array[0]) == len(accept_array[0]):
                return True
        if len(accept_array) == 2:
            if np.sum(accept_array[0]) < 2 or np.sum(accept_array[1]) < 2:
                self.row["run"] = np.nan
                return True
            if np.sum(accept_array[0]) == len(accept_array[0]) and np.sum(accept_array[1]) == len(accept_array[1]):
                return True
        return False
    
    def eval_and_reduce(self, to_accept_list=None, to_accept_bool=None):
        if type(to_accept_bool).__name__ != "NoneType":
            if to_accept_bool:
                pass
            else:
                self.row["run"] = np.nan
        if type(to_accept_list).__name__ != "NoneType":
            if not self.eval_on_length(to_accept_list):
                self.reduce_row(to_accept_list)


class FilterStr(ProcessingRow):
    
    def __init__(self, row, name_list, calc_instance=Calc, allowed_instance=Allowed):
        super(FilterStr, self).__init__(row, name_list)
        self.calc_instance = calc_instance
        self.allowed_instance = allowed_instance
    
    # Filter No 1
    def check_type(self, look_for="muon"):
        
        """
        Delete all non "G" tagged muons and all non "T" tagged electrons and their
        corresponding other values in event.

        :param look_for: "muon", "electron", or "both"
        :return: reduced row or row with np.nan value in it
        """
        
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["type"],
                                          type_variables=[str])  # [look_for + "_type"], type_variables=[str])
            type_array = found_array[0]
            
            accept_array = self.allowed_instance.type(type_array, look_for)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_type", "electron_type"], type_variables=[str, str])
            type_mu, type_el = found_array[0], found_array[1]
            
            accept_array_mu = self.allowed_instance.type(type_mu, "muon")
            accept_array_el = self.allowed_instance.type(type_el, "electron")
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 2
    def check_q(self, look_for="muon"):
        
        """
        Function to check combined charge.
        If look_for != "both" then it will check all possible combinations out of 4 particles.
        If look_for == "both" then it will check all possible combinations of electrons and muons separately
        out of 2 particles.
        If 0 is one possible outcome the function wont touch the row.

        :param look_for: "muon", "electron" or "both"
        :return: reduced row or row with np.nan value in it
        """
        
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=[look_for + "_charge"], type_variables=[int])
            charge = found_array[0]
            
            accept_charge = self.calc_instance.combined_charge(charge, 4)
            self.eval_and_reduce(to_accept_bool=(accept_charge == 0))
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_charge", "electron_charge"],
                                          type_variables=[int, int])
            charge_mu, charge_el = found_array[0], found_array[1]
            
            accept_charge_mu = self.calc_instance.combined_charge(charge_mu, 2)
            accept_charge_el = self.calc_instance.combined_charge(charge_el, 2)
            
            self.eval_and_reduce(to_accept_bool=(accept_charge_mu == 0 and accept_charge_el == 0))
            return self.row
        
        else:
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 3
    def check_min_pt(self, look_for="muon"):
        
        """
        Function to remove all particles in a run with a pt lesser than minimal reasonable.

        :param look_for: "muon", "electron" or "both"
        :return: unchanged row with np.nan value in it
        """
        
        if self.if_column():
            return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py"], type_variables=[float, float])
            px, py = found_array[0], found_array[1]
            
            pt = self.calc_instance.pt(px, py)
            self.add_raw_to_row(variable_name="pt", variable_array=pt)
            accept_array = self.allowed_instance.min_pt(pt, look_for)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "electron_px", "electron_py"],
                                          type_variables=[float, float, float, float])
            px_mu, py_mu = found_array[0], found_array[1]
            px_el, py_el = found_array[2], found_array[3]
            
            pt_mu = self.calc_instance.pt(px_mu, py_mu)
            pt_el = self.calc_instance.pt(px_el, py_el)
            self.add_raw_to_row(variable_name="muon_pt", variable_array=pt_mu)
            self.add_raw_to_row(variable_name="electron_pt", variable_array=pt_el)
            accept_array_mu = self.allowed_instance.min_pt(pt_mu, "muon")
            accept_array_el = self.allowed_instance.min_pt(pt_el, "electron")
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 4
    def check_eta(self, look_for="muon"):
        
        """
        Function to check if particle is in allowed eta region. Will remove those particles if not.

        :param look_for: "muon", "electron" or "both"
        :return: reduced row or row with np.nan value in it
        """
        
        if self.if_column():
            return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py", "pz"],
                                          type_variables=[float, float, float])
            px, py, pz = found_array[0], found_array[1], found_array[2]
            
            eta = self.calc_instance.eta(px, py, pz)
            self.add_raw_to_row(variable_name="eta", variable_array=eta)
            accept_array = self.allowed_instance.eta(eta, look_for)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "muon_pz",
                                                            "electron_px", "electron_py", "electron_pz"],
                                          type_variables=[float, float, float, float, float, float])
            px_mu, py_mu, pz_mu = found_array[0], found_array[1], found_array[2]
            px_el, py_el, pz_el = found_array[3], found_array[4], found_array[5]
            
            eta_mu = self.calc_instance.eta(px_mu, py_mu, pz_mu)
            eta_el = self.calc_instance.eta(px_el, py_el, pz_el)
            self.add_raw_to_row(variable_name="muon_eta", variable_array=eta_mu)
            self.add_raw_to_row(variable_name="electron_eta", variable_array=eta_el)
            accept_array_mu = self.allowed_instance.eta(eta_mu, "muon")
            accept_array_el = self.allowed_instance.eta(eta_el, "electron")
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 5
    def check_misshit(self, look_for="electron"):
        
        """
        Function to remove electrons with misshits count greater than a specific value.

        :param look_for: "electron" or "both"
        :return: reduced row or row with np.nan value in it
        """
        
        if self.if_column():
            return self.row
        
        if look_for == "electron":
            found_array = self.search_for(search_variables=["misshits"], type_variables=[int])
            misshits = found_array[0]
            
            accept_array = self.allowed_instance.misshits(misshits)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["misshits"], type_variables=[int])
            misshits = found_array[0]
            
            accept_array_el = self.allowed_instance.misshits(misshits)
            accept_array_mu = np.ones(len(accept_array_el), dtype=bool)  # dummy for separation mu/el in __reduce_row
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            sys.exit("Error: look_for can only be: 'electron' or 'both'")
    
    # Filter No 6
    def check_rel_iso(self, look_for="muon"):
        
        """
        Function to Check if relIso of a Particle satisfy the given requirement.

        :param look_for: "electron", "muon" or "both"
        :return: reduced row with np.nan value in it
        """
        
        if self.if_column():
            return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["relPFIso"], type_variables=[float])
            rel_pf_iso = found_array[0]
            
            accept_array = self.allowed_instance.rel_pf_iso(rel_pf_iso)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_relPFIso", "electron_relPFIso"],
                                          type_variables=[float, float])
            rel_pf_iso_mu, rel_pf_iso_el = found_array[0], found_array[1]
            
            accept_array_mu = self.allowed_instance.rel_pf_iso(rel_pf_iso_mu)
            accept_array_el = self.allowed_instance.rel_pf_iso(rel_pf_iso_el)
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        else:
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 7
    def check_impact_param(self, look_for="muon"):
        
        """
        Function than checks in impact parameter and filter out particle that do not satisfy the specific condition.

        :param look_for: "muon" or "electron" or "both"
        :return: reduced row or row with np.nan value in it
        """
        
        if self.if_column():
            return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["SIP3d", "dxy", "dz"],
                                          type_variables=[float, float, float])
            sip3d, dxy, dz = found_array[0], found_array[1], found_array[2]
            
            accept_array = self.allowed_instance.impact_param(sip3d, dxy, dz)
            
            self.eval_and_reduce(to_accept_list=[accept_array])
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_SIP3d", "muon_dxy", "muon_dz",
                                                            "electron_SIP3d", "electron_dxy", "electron_dz"],
                                          type_variables=[float, float, float, float, float, float])
            sip3d_mu, dxy_mu, dz_mu = found_array[0], found_array[1], found_array[2]
            sip3d_el, dxy_el, dz_el = found_array[3], found_array[4], found_array[5]
            
            accept_array_mu = self.allowed_instance.impact_param(sip3d_mu, dxy_mu, dz_mu)
            accept_array_el = self.allowed_instance.impact_param(sip3d_el, dxy_el, dz_el)
            
            self.eval_and_reduce(to_accept_list=[accept_array_mu, accept_array_el])
            return self.row
        
        return None
    
    # Filter No 8
    def check_exact_pt(self, look_for="muon"):
        
        """
        Function to check if the final restriction on pt is passed by enough amount of particle for ZZ combination.

        :param look_for: "electron", "muon" or "both"
        :return: unchanged row or row with np.nan value in it
        """
        
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py"], type_variables=[float, float])
            px, py = found_array[0], found_array[1]
            
            pt = self.calc_instance.pt(px, py)
            self.add_raw_to_row(variable_name="pt", variable_array=pt)
            accept_value = self.allowed_instance.pt(pt, look_for)
            
            self.eval_and_reduce(to_accept_bool=accept_value)
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "electron_px", "electron_py"],
                                          type_variables=[float, float, float, float])
            px_mu, py_mu, px_el, py_el = found_array[0], found_array[1], found_array[2], found_array[3]
            
            pt_mu = self.calc_instance.pt(px_mu, px_mu)
            pt_el = self.calc_instance.pt(px_el, py_el)
            self.add_raw_to_row(variable_name="muon_pt", variable_array=pt_mu)
            self.add_raw_to_row(variable_name="electron_pt", variable_array=pt_el)
            accept_mu = self.allowed_instance.pt(pt_mu, "muon", 2)
            accept_el = self.allowed_instance.pt(pt_el, "electron", 2)
            
            self.eval_and_reduce(to_accept_bool=(accept_el and accept_mu))
            return self.row
        
        else:
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 9
    def check_m_2l(self, look_for="muon"):
        
        """
        Function to filter out, by looking at the invariant two lepton mass.

        :param look_for: "muon", "electron" or both
        :return: unchanged row or row with np.nan value in it
        """
        
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["energy", "px", "py", "pz", "charge"],
                                          type_variables=[float, float, float, float, float])
            energy, charge = found_array[0], found_array[4]
            px, py, pz = found_array[1], found_array[2], found_array[3]
            
            inv_m = self.calc_instance.possible_invariant_masses(px, py, pz, charge, energy=energy, number_of_leptons=2)
            accept_value = self.allowed_instance.invariant_mass(inv_m, 2, look_for=look_for)
            
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
            
            inv_m_mu = self.calc_instance.possible_invariant_masses(px_mu, py_mu, pz_mu, charge_mu, energy=energy_mu,
                                                                    number_of_leptons=2)
            inv_m_el = self.calc_instance.possible_invariant_masses(px_el, py_el, pz_el, charge_el, energy=energy_el,
                                                                    number_of_leptons=2)
            accept_value_mu = self.allowed_instance.invariant_mass(inv_m_mu, 2, look_for=look_for)
            accept_value_el = self.allowed_instance.invariant_mass(inv_m_el, 2, look_for=look_for)
            
            self.eval_and_reduce(to_accept_bool=(accept_value_mu and accept_value_el))
            return self.row
        
        else:
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")
    
    # Filter No 10
    def check_m_4l(self, look_for="muon"):
        
        """
        Function to check if the invariant mass of a combination of 4 leptons is valid.

        :param look_for: "muon" or "electron" or "both"
        :return: unchanged row or row with np.nan value in it
        """
        
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["energy", "px", "py", "pz", "charge"],
                                          type_variables=[float, float, float, float, float])
            
            energy, charge = found_array[0], found_array[4]
            px, py, pz = found_array[1], found_array[2], found_array[3]
            
            inv_m = self.calc_instance.possible_invariant_masses(px, py, pz, charge, energy=energy, number_of_leptons=4)
            accept_value = self.allowed_instance.invariant_mass(inv_m, 4, look_for=look_for)
            
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
            
            inv_m = self.calc_instance.possible_invariant_masses([px_mu, px_el], [py_mu, py_el], [pz_mu, pz_el],
                                                                 [charge_mu, charge_el], energy=[energy_mu, energy_el],
                                                                 number_of_leptons=4, look_for_both=True)
            accept_value = self.allowed_instance.invariant_mass(inv_m, 4, look_for=look_for)
            
            self.eval_and_reduce(to_accept_bool=accept_value)
            return self.row
        
        else:
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")


class AddVariable(ProcessingRow):
    
    def __init__(self, row, name_list, calc_instance=Calc, allowed_instance=Allowed):
        super(AddVariable, self).__init__(row, name_list)
        self.calc_instance = calc_instance
        self.allowed_instance = allowed_instance
    
    def pt(self, look_for="muon"):
        
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py"], type_variables=[float, float])
            px, py = found_array[0], found_array[1]
            
            pt = self.calc_instance.pt(px, py)
            self.add_raw_to_row(variable_name="pt", variable_array=pt)
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "electron_px", "electron_py"],
                                          type_variables=[float, float, float, float])
            px_mu, py_mu = found_array[0], found_array[1]
            px_el, py_el = found_array[2], found_array[3]
            
            pt_mu = self.calc_instance.pt(px_mu, py_mu)
            pt_el = self.calc_instance.pt(px_el, py_el)
            self.add_raw_to_row(variable_name="muon_pt", variable_array=pt_mu)
            self.add_raw_to_row(variable_name="electron_pt", variable_array=pt_el)
            return self.row
        
        else:
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")
    
    def eta(self, look_for="muon"):
        
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            found_array = self.search_for(search_variables=["px", "py", "pz"],
                                          type_variables=[float, float, float])
            px, py, pz = found_array[0], found_array[1], found_array[2]
            
            eta = self.calc_instance.eta(px, py, pz)
            self.add_raw_to_row(variable_name="eta", variable_array=eta)
            return self.row
        
        if look_for == "both":
            found_array = self.search_for(search_variables=["muon_px", "muon_py", "muon_pz",
                                                            "electron_px", "electron_py", "electron_pz"],
                                          type_variables=[float, float, float, float, float, float])
            px_mu, py_mu, pz_mu = found_array[0], found_array[1], found_array[2]
            px_el, py_el, pz_el = found_array[3], found_array[4], found_array[5]
            
            eta_mu = self.calc_instance.eta(px_mu, py_mu, pz_mu)
            eta_el = self.calc_instance.eta(px_el, py_el, pz_el)
            self.add_raw_to_row(variable_name="muon_eta", variable_array=eta_mu)
            self.add_raw_to_row(variable_name="electron_eta", variable_array=eta_el)
            return self.row
        
        else:
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")


class Reconstruct(ProcessingRow):
    """
    Filter used to Reconstruct:
        - 4l->ZZ
        - ZZ->H
    row-wise, might look different for a column-wise reconstruction
    """
    
    def __init__(self, row, name_list, calc_instance=Calc, allowed_instance=Allowed):
        super(Reconstruct, self).__init__(row, name_list)
        self.calc_instance = calc_instance
        self.allowed_instance = allowed_instance
    
    def zz(self, look_for="muon"):
        
        """
        Function to reconstruct Z1 and Z2 from a given dataset

        :param look_for: "muon", "electron" or both
        :return: a) Row with a NaN if it was not possible to recreate Z1 and Z2
                 b) Row with phi, Z1, Z2, Z1_index and Z2_index if look_for == "muon" or "electron"
                 c) Row with muon_phi, electron_phi, Z1, Z2, Z1_index, Z2_index, Z1_tag and Z2_ tag if look_for="both"
        """
        
        if self.if_column(): return self.row
        
        if look_for != "both" and (look_for == "muon" or look_for == "electron"):
            
            found_array = self.search_for(search_variables=["eta", "px", "py", "pz", "energy",
                                                            "pt", "charge"],
                                          type_variables=[float, float, float, float, float, float, float])
            eta, px, py, pz, energy = found_array[0], found_array[1], found_array[2], found_array[3], found_array[4]
            pt, charge = found_array[5], found_array[6]
            
            phi = self.calc_instance.phi(px, py)
            z1, z2, index_z1, index_z2 = self.calc_instance.zz_and_index(eta, phi, pt, px, py, pz, charge,
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
            
            phi_mu = self.calc_instance.phi(px_mu, py_mu)
            phi_el = self.calc_instance.phi(px_el, py_el)
            z1, z2, index_z1, index_z2, z1_tag, z2_tag = self.calc_instance.zz_and_index([eta_mu, eta_el],
                                                                                         [phi_mu, phi_el],
                                                                                         [pt_mu, pt_el], [px_mu, px_el],
                                                                                         [py_mu, py_el], [pz_mu, pz_el],
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
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")
    
    def mass_4l_out_zz(self, look_for="muon"):
        
        """
        Function to reconstruct the four lepton invariant mass out of given two Z bosons

        :param look_for: 'muon', 'electron' or 'both'
        :return: a row that contains 'mass_4l'
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
            sys.exit("Error: look_for can only be: 'muon', 'electron' or 'both'")


class ProcessDuplicates(object):
    """
    Class to Search for duplicates and handle them

    Useful case: minimum after ZZ reconstruction
    """
    
    def __init__(self, input_1, input_2):
        if type(input_1).__name__ == "str" and type(input_2).__name__ == "str":
            self.data_1 = pd.read_csv(input_1, sep=";")
            self.data_2 = pd.read_csv(input_2, sep=";")
            self.name_1 = input_1
            self.name_2 = input_2
        else:
            self.data_1 = input_1
            self.data_2 = input_2
            self.name_1 = None
            self.name_2 = None
    
    def remove_by_best_z1(self):
        
        """
        Function to handle the duplicates as follows:
            1) take useful columns from two dataset and create two sub_datasets
            2) calculate the difference between Z1 and Z0 and add this column to 1)
            3) label those two sub_datasets individually 'frame_id'
            4) search for duplicates, save those pairs and remove the best fitting
            5) collect remaining rows from 4) in two separate datasets
            6) create keys from 5) and remove rows from initial datasets that have those keys in them
        :return: update the datasets (class intern)
        """
        
        z_mass = 91.1876
        
        d1 = self.data_1.filter(["run", "event", "z1_mass", "z2_mass"], axis=1)
        d1["frame_id"] = 1
        d1["z1_diff"] = np.abs(d1["z1_mass"] - z_mass)
        
        d2 = self.data_2.filter(["run", "event", "z1_mass", "z2_mass"], axis=1)
        d2["frame_id"] = 2
        d2["z1_diff"] = np.abs(d2["z1_mass"] - z_mass)
        
        dg = pd.concat([d1, d2])
        
        try:
            dg_dup = [g for _, g in dg.groupby(["run", "event"]) if len(g) > 1]
            red_d = pd.concat(frame.sort_values(["z1_diff"]).iloc[:2] for frame in dg_dup)
            
            red_d1 = red_d[red_d["frame_id"] == 1].drop(["frame_id", "z1_diff"], axis=1)
            red_d2 = red_d[red_d["frame_id"] == 2].drop(["frame_id", "z1_diff"], axis=1)
            
            print(str(red_d1.shape[0]) + " duplicates found!")
            
            keys1 = list(red_d1.columns.values)
            keys2 = list(red_d2.columns.values)
            
            index_all_1 = self.data_1.set_index(keys1).index
            index_all_2 = self.data_2.set_index(keys2).index
            
            index_drop_1 = red_d1.set_index(keys1).index
            index_drop_2 = red_d2.set_index(keys2).index
            
            self.data_2 = self.data_2[~index_all_2.isin(index_drop_2)]
            self.data_1 = self.data_1[~index_all_1.isin(index_drop_1)]
        
        except ValueError:
            print("No duplicates found!")
    
    def test_equality(self):
        
        """
        Function to test if two datasets are equal or not.
        Useful to check if you wanr to make sure that two different filter sequence dont produce
        two different outputs

        :return: True == are equal; False == are not equal
        """
        
        # might be changed to float accidentally, therefore...
        self.data_1 = self.data_1.astype({"run": "int64", "event": "int64"})
        self.data_2 = self.data_2.astype({"run": "int64", "event": "int64"})
        
        try:
            assert_frame_equal(self.data_1, self.data_2)
            return True
        except Exception:
            return False
    
    def get_frames(self):
        
        """
        Function to return given  dataframes

        :return: pandas.DataFrame, pandas.DataFrame
        """
        
        return self.data_1, self.data_2
    
    def save_to_csv(self, name1=None, name2=None):
        
        """
        Function to save given dataframes as csv

        :param name1: filename for the first dataframe (not needed if dataframe is loaded from file)
        :param name2: filename for the second dataframe (not needed if dataframe is loaded from file)
        """
        
        check_self_name_exist = type(self.name_1).__name__ == "str" and type(self.data_2).__name__ == "str"
        check_given_name = type(name1).__name__ == "str" and type(name2).__name__ == "str"
        
        if check_given_name:
            self.data_1.to_csv(name1, sep=";", index=False)
            self.data_2.to_csv(name2, sep=";", index=False)
        
        if check_self_name_exist and not check_given_name:
            self.data_1.to_csv(self.name_1, sep=";", index=False)
            self.data_2.to_csv(self.name_2, sep=";", index=False)
        
        if not check_given_name and not (check_self_name_exist and not check_given_name):
            self.data_1.to_csv("dataset_1_checked_for_duplicates.csv", sep=";", index=False)
            self.data_2.to_csv("dataset_2_checked_for_duplicates.csv", sep=";", index=False)


class Apply(object):
    """
    Class to apply filter or a reconstruction to a dataset.
    """
    
    def __init__(self, file_name, particle_type, verbose=False, use_n_rows=None,
                 filter_instance=FilterStr,
                 add_variable_instance=AddVariable,
                 reconstruction_instance=Reconstruct,
                 calc_instance=Calc,
                 allowed_instance=Allowed,
                 multi_cpu=True):
        if type(file_name).__name__ == "str":
            print("Load " + file_name)
            self.data = pd.read_csv(file_name, sep=";", nrows=use_n_rows)
            print("Loading successful completed")
            self.name = file_name
        if type(file_name).__name__ != "str":
            self.data = file_name
        self.particle_type = particle_type
        self.add_variable_instance = add_variable_instance
        self.reconstruct_instance = reconstruction_instance
        self.filter_instance = filter_instance
        self.calc_instance = calc_instance
        self.allowed_instance = allowed_instance
        self.verbose = verbose
        self.multi_cpu = multi_cpu
        self.calculated_dump = {}
    
    def set_calc_instance(self, **kwargs):
        if any("calc" in it for it in kwargs.keys()): self.calc_instance = kwargs["calc_instance"]
        if any("allowed" in it for it in kwargs.keys()): self.allowed_instance = kwargs["allowed_instance"]
        if any("filter" in it for it in kwargs.keys()): self.filter_instance = kwargs["filter_instance"]
        if any("reconstruct" in it for it in kwargs.keys()): self.reconstruct_instance = kwargs["reconstruct_instance"]
    
    def get_partial(self, arg_tuple=False, **kwargs):
        
        used_class, used_name, data_frame, intern_verbose = None, None, None, True
        if arg_tuple:
            used_class, used_name, data_frame, intern_verbose = arg_tuple
        if kwargs:
            used_class, used_name, data_frame = kwargs["used_class"], kwargs["used_name"], kwargs["data_frame"]
        
        if intern_verbose:
            print("Do {}: {}; shape: {}".format(used_class.__name__, used_name, data_frame.shape))
        if not data_frame.empty:
            if self.verbose:
                data_frame = data_frame.swifter.apply(
                    lambda x: getattr(used_class(x,
                                                 list(data_frame),
                                                 calc_instance=self.calc_instance,
                                                 allowed_instance=self.allowed_instance),
                                      used_name)(look_for=self.particle_type), axis=1)
            if not self.verbose:
                data_frame = data_frame.apply(
                    lambda x: getattr(used_class(x,
                                                 list(data_frame),
                                                 calc_instance=self.calc_instance,
                                                 allowed_instance=self.allowed_instance),
                                      used_name)(look_for=self.particle_type), axis=1)
            data_frame = data_frame.dropna()
            if intern_verbose:
                print("Done {}: {}; shape: {}".format(used_class.__name__, used_name, data_frame.shape))
            return data_frame
    
    def __multiprocessing(self, **kwargs):
        
        n_cpu = mp.cpu_count()
        if not kwargs["data_frame"].empty:
            data_frames = np.array_split(kwargs["data_frame"], n_cpu)
            pass_args = [(kwargs["used_class"], kwargs["used_name"], frame_, False) for frame_ in data_frames]
            pool = mp.Pool(processes=n_cpu)
            print("Do {}: {}; shape: {}".format(kwargs["used_class"].__name__, kwargs["used_name"],
                                                kwargs["data_frame"].shape))
            results = pool.map(self.get_partial, pass_args)
            pool.close()
            pool.join()
            collected_frame = pd.concat([item for item in results])
            print("Done {}: {}; shape: {}".format(kwargs["used_class"].__name__, kwargs["used_name"],
                                                  collected_frame.shape))
            return collected_frame
        else:
            return kwargs["data_frame"]
    
    def __do_quicksave(self, name):
        if type(name).__name__ == "str":
            if not self.data.empty:
                to_place_dir, _ = os.path.split(name)
                if not os.path.isdir(to_place_dir):
                    os.makedirs(to_place_dir, exist_ok=True)
                self.data.to_csv(name, index=False, sep=";")
    
    @staticmethod
    def help(name_list="--"):
        
        """
        Function to print all given functionalities by Apply class

        :param name_list: 'reconstruction' or 'filter'; '--' will show both
        """
        
        if name_list != "reconstruction":
            print("Possible Filter:\n" +
                  "- 'check_type'\n- 'check_q'\n- 'check_min_pt'\n- 'check_eta'\n- 'check_misshit'\n" +
                  "- 'check_rel_iso'\n- 'check_impact_param'\n- 'check_exact_pt'\n- 'check_m_2l'\n- 'check_m_4l'\n" +
                  "\n")
        
        if name_list != "filter":
            print("Possible Reconstructions:" +
                  "- 'zz'\n" +
                  "- 'mass_4l_out_zz'\n" +
                  "\n")
        
        if name_list != "add_variables":
            print("Adding possible variables: \n" +
                  "- 'pt'\n" +
                  "- 'eta'\n")
    
    def add_variable(self, variable_name, quicksave=None):
        
        if not self.multi_cpu:
            self.data = self.get_partial(used_class=self.add_variable_instance, used_name=variable_name,
                                         data_frame=self.data)
        if self.multi_cpu:
            self.data = self.__multiprocessing(used_class=self.add_variable_instance, used_name=variable_name,
                                               data_frame=self.data)
        
        self.__do_quicksave(name=quicksave)
    
    def filter(self, filter_name, quicksave=None):
        
        if not self.multi_cpu:
            self.data = self.get_partial(used_class=self.filter_instance, used_name=filter_name, data_frame=self.data)
        
        if self.multi_cpu:
            self.data = self.__multiprocessing(used_class=self.filter_instance, used_name=filter_name,
                                               data_frame=self.data)
        
        self.__do_quicksave(quicksave)
    
    def reconstruct(self, reco_name, quicksave=None):
        
        if not self.multi_cpu:
            self.data = self.get_partial(used_class=self.reconstruct_instance, used_name=reco_name,
                                         data_frame=self.data)
        if self.multi_cpu:
            self.data = self.__multiprocessing(used_class=self.reconstruct_instance, used_name=reco_name,
                                               data_frame=self.data)
        
        self.__do_quicksave(quicksave)
    
    @property
    def get(self):
        return self.data
    
    def draw_hist(self, bins, hist_range, variable,
                  used_filter=None, alpha=None, label=None, color=None):
        
        hist = Hist(bins=bins, hist_range=hist_range)
        if type(variable).__name__ == "str": variable = [variable]
        if ("2el2mu" in self.name or "DiMuon" in self.name) and len(variable) == 1: variable = ["muon_" + variable[0],
                                                                                                "electron_" + variable[
                                                                                                    0]]
        if type(used_filter).__name__ != "NoneType": variable.append(used_filter[0])
        array_to_fill = hist.convert_column(column=self.data.filter(variable), used_filter=used_filter)
        hist.fill_hist(name="raw_undefined", array_of_interest=array_to_fill)
        hist.draw_hist(alpha=[alpha], label=[label], color=[color])
        self.calculated_dump[self.name] = hist
        return hist
    
    def draw_2d_hist(self, bins, hist_range, variable, used_filter=None):
        
        atf_2d = []
        for var, one_hist_range in zip(variable, hist_range):
            hist = Hist(bins=bins, hist_range=one_hist_range)
            if ("2el2mu" in self.name or "DiMuon" in self.name) and ("electron_" not in var and "muon_" not in var):
                var = ["muon_" + var, "electron_" + var]
            var = [var]
            if type(used_filter).__name__ != "NoneType": var = [var, used_filter[0]]
            array_to_fill = hist.convert_column(column=self.data.filter(var), used_filter=used_filter)
            atf_2d.append(array_to_fill)
        h, x_edg, y_edg = np.histogram2d(atf_2d[0][0:len(atf_2d[1])], atf_2d[1][0:len(atf_2d[0])], bins=bins)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(h, aspect='auto', extent=[hist_range[1][0], hist_range[1][1], hist_range[0][0], hist_range[0][1]])
        plt.tight_layout()
        plt.show()


class Hist(object):
    """
    Class for creating histograms
    """
    
    def __init__(self, **kwargs):
        self.save_dir = "./histograms"
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.kwargs = dict(kwargs)
        
        self.__lumen = {"2011": {"A": 4.7499}, "2012": {"A": np.longdouble(0.889391999043448687),
                                                        "B": np.longdouble(4.429375295985512733),
                                                        "C": np.longdouble(7.152728016920716286),
                                                        "D": np.longdouble(7.318301466596649170),
                                                        "A-D": np.longdouble(19.789796778546325684)}}
        self.__event_num_mc = {"2011": {"ZZ_4mu": 1447136, "ZZ_4el": 1493308, "ZZ_2el2mu": 1479879, "H_ZZ": 299683},
                               "2012": {"ZZ_4mu": 1499064, "ZZ_4el": 1499093, "ZZ_2el2mu": 1497445, "H_ZZ": 299973}}
        
        self.kwargs["signal_mass_mc"] = 125
        
        if "signal_mass_mc" in list(self.kwargs.keys()):
            if self.kwargs["signal_mass_mc"] == 115:
                print("{} total event number changed from {} to {} for a signal MC at {} GeV".format("H_ZZ", 299973,
                                                                                                     299971, 115))
                self.__event_num_mc["2012"]["H_ZZ"] = 299971
            if self.kwargs["signal_mass_mc"] == 145:
                print("{} total event number changed from {} to {} for a signal MC at {} GeV".format("H_ZZ", 299971,
                                                                                                     287375, 145))
                self.__event_num_mc["2012"]["H_ZZ"] = 287375
            if self.kwargs["signal_mass_mc"] == 135:
                print("{} total event number changed from {} to {} for a signal MC at {} GeV".format("H_ZZ", 299971,
                                                                                                     299971, 145))
                self.__event_num_mc["2012"]["H_ZZ"] = 299971
            if self.kwargs["signal_mass_mc"] == 140:
                print("{} total event number changed from {} to {} for a signal MC at {} GeV".format("H_ZZ", 299971,
                                                                                                     299971, 145))
                self.__event_num_mc["2012"]["H_ZZ"] = 299971
            if self.kwargs["signal_mass_mc"] == 124:
                print("{} total event number changed from {} to {} for a signal MC at {} GeV".format("H_ZZ", 299971,
                                                                                                     276573, 145))
                self.__event_num_mc["2012"]["H_ZZ"] = 276573
            if self.kwargs["signal_mass_mc"] == 120:
                print("{} total event number changed from {} to {} for a signal MC at {} GeV".format("H_ZZ", 299971,
                                                                                                     295473, 145))
                self.__event_num_mc["2012"]["H_ZZ"] = 295473
            if self.kwargs["signal_mass_mc"] == 128:
                print("{} total event number changed from {} to {} for a signal MC at {} GeV".format("H_ZZ", 299971,
                                                                                                     267274, 145))
                self.__event_num_mc["2012"]["H_ZZ"] = 267274
            if self.kwargs["signal_mass_mc"] == 122:
                print("{} total event number changed from {} to {} for a signal MC at {} GeV".format("H_ZZ", 299971,
                                                                                                     299970, 145))
                self.__event_num_mc["2012"]["H_ZZ"] = 299970
            if self.kwargs["signal_mass_mc"] == 150:
                print("{} total event number changed from {} to {} for a signal MC at {} GeV".format("H_ZZ", 299971,
                                                                                                     299973, 145))
                self.__event_num_mc["2012"]["H_ZZ"] = 299973
            if self.kwargs["signal_mass_mc"] == 130:
                print("{} total event number changed from {} to {} for a signal MC at {} GeV".format("H_ZZ", 299971,
                                                                                                     299976, 145))
                self.__event_num_mc["2012"]["H_ZZ"] = 299976
        
        self.__cross_sec = {"2011": {"ZZ_4mu": 66.09, "ZZ_4el": 66.09, "ZZ_2el2mu": 152, "H_ZZ": 5.7},
                            "2012": {"ZZ_4mu": 76.91, "ZZ_4el": 76.91, "ZZ_2el2mu": 176.7, "H_ZZ": 6.5}}
        self.__k_factor = {"2011": {"ZZ_4mu": 1.386, "ZZ_4el": 1.386, "ZZ_2el2mu": 1.386, "H_ZZ": 1.0},
                           "2012": {"ZZ_4mu": 1.386, "ZZ_4el": 1.386, "ZZ_2el2mu": 1.386, "H_ZZ": 1.0}}
        self.data = {}
        
        create_by_bins_and_hist_range = "hist_range" in self.kwargs.keys() and "bins" in self.kwargs.keys()
        given_pandas_dataframe = "file" in self.kwargs.keys()
        
        if create_by_bins_and_hist_range:
            self.range_interval = self.kwargs["hist_range"]
            self.bins = self.kwargs["bins"]
            
            self.range = np.linspace(self.kwargs["hist_range"][0], self.kwargs["hist_range"][1],
                                     self.kwargs["bins"] + 1)
            self.x_range = self.range[:-1] + np.abs(self.range[0] - self.range[1]) / 2.
            self.bin_width = np.abs(self.x_range[0] - self.x_range[1])
            if given_pandas_dataframe:
                loaded_data = pd.read_csv(self.kwargs["file"])
                for name in list(loaded_data):
                    if name == "x_range":
                        temp_x_range = loaded_data["x_range"].values
                        if not all(temp_x_range) == all(self.x_range):
                            sys.exit("Error: Dimension from file != dimension from bins")
                        continue
                    self.data[name] = loaded_data[name].values
        
        if given_pandas_dataframe and not create_by_bins_and_hist_range:
            loaded_data = pd.read_csv(self.kwargs["file"])
            for name in list(loaded_data):
                if name == "x_range":
                    self.x_range = loaded_data["x_range"].values
                    continue
                self.data[name] = loaded_data[name].values
            
            self.bins = loaded_data.shape[0]
            temp_width = np.abs(self.x_range[0] - self.x_range[1])
            self.range_interval = (self.x_range[0] - temp_width / 2, self.x_range[-1] + temp_width / 2.)
            self.range = np.linspace(self.range_interval[0], self.range_interval[1], self.bins + 1)
            self.bin_width = np.abs(self.x_range[0] - self.x_range[1])
    
    def set_empty_bins(self, name):
        
        """
        Function to create specific empty bins

        :param name: 'data', 'mc_sig' or 'mc_bac'
        """
        
        self.data[name] = np.zeros(len(self.x_range))
    
    def __correct_fac(self, dim, year, run, process):
        
        """
        Function to calculate the factor to correct the montecarlo data

        :param dim: dimension (int) (schould correspond to the size of the initial array
        :param year: specific year
        :param run: specific run(s) (list of string arrays if more than one)
        :param process: viewed process
        :return: array with dimension dim containing the correction factor in every item
        """
        
        total_lumen = 0.0
        if type(run).__name__ == "str":
            run = [run]
        
        for ru in run:
            total_lumen += self.__lumen[year][ru]
        
        return np.ones(dim) * total_lumen * self.__k_factor[year][process] * self.__cross_sec[year][process] / \
               self.__event_num_mc[year][process]
    
    @staticmethod
    def convert_column(column, used_filter=None):
        
        """
        convert a given column to a numpy list of floats

        :param column: list of strings corresponding to specific columns
        :param used_filter: if filter is used: ['column_name', [a, b]]
                                'column_name': should be 'z1_mass', 'z2_mass' or 'mass_4l'
                                a: lower border; will remove everything below this value
                                b: upper border; will remove everything above this value
        :return: numpy array
        """
        
        if type(used_filter).__name__ != "NoneType":
            filter_array = (column[used_filter[0]].values > used_filter[1][0]) & (
                    column[used_filter[0]].values < used_filter[1][1])
            column = column.drop([used_filter[0]], axis=1)
            pass_array = column.values
            pass_array = pass_array[filter_array]
        
        else:
            pass_array = column.values
        
        try:
            pass_array = np.concatenate(pass_array)
        
        except ValueError:
            pass  # there is no need for concatenate because it is 4mu or 4e
        try:
            pass_array = np.concatenate([np.array(item.split(","), dtype=float) for item in pass_array])
        
        except AttributeError:
            pass  # there is no need for split because it only contains one item
        except ValueError:
            pass  # there is no need for split because it only contains one item
        
        return pass_array
    
    @staticmethod
    def __calc_errors_sqrt(array_of_interest):
        return np.sqrt(np.array(array_of_interest))
    
    @staticmethod
    def __calc_errors_poisson(array_of_interest, interval=0.68):
        
        """
        Poisson approach

        :param array_of_interest: given numpy array containing data
        :param interval: coverage interval
        :return: following list of numpy arrays [lower_error, upper_error]
        """
        
        lower_cap = (1 - interval) / 2.
        higher_cap = (1 - interval) / 2. + interval
        
        def get_lower_higher_value(x):
            return scst.poisson.ppf(lower_cap, x), scst.poisson.ppf(higher_cap, x)
        
        lower_limit = np.array([get_lower_higher_value(value)[0] for value in array_of_interest])
        higher_limit = np.array([get_lower_higher_value(value)[1] for value in array_of_interest])
        return [lower_limit, higher_limit]
    
    @staticmethod
    def __calc_errors_poisson_with_gamma(array_of_interest, interval=0.68):
        
        """
        Poisson approach approximated a s gamma

        :param array_of_interest: given numpy array containing data
        :param interval: coverage interval
        :return: following list of numpy arrays [lower_error, upper_error]
        """
        
        lower_cap = (1 - interval) / 2.
        higher_cap = (1 - interval) / 2. + interval
        
        def get_lower_higher_value(x):
            return scst.gamma.ppf(lower_cap, x), scst.gamma.ppf(higher_cap, x)
        
        lower_limit = np.array([get_lower_higher_value(value)[0] for value in array_of_interest])
        higher_limit = np.array([get_lower_higher_value(value)[1] for value in array_of_interest])
        return [lower_limit, higher_limit]
    
    @staticmethod
    def __calc_errors_alternative_near_simplified(array_of_interest):
        
        """
        Alternative calculation of asymetric errorbars - simplified self.__calc_errors_poisson version

        :param array_of_interest: given numpy array containing data
        :return: following list of numpy arrays [lower_error, upper_error]
        """
        
        def get_lower_higher_value(x):
            return -0.5 + np.sqrt(x + 0.025), 0.5 + np.sqrt(x + 0.25)
        
        lower_limit = np.array([get_lower_higher_value(value)[0] for value in array_of_interest])
        higher_limit = np.array([get_lower_higher_value(value)[1] for value in array_of_interest])
        return [lower_limit, higher_limit]
    
    @staticmethod
    def __calc_errors_poisson_near_cont(array_of_interest):
        """
        Function to calculate errors on "continuous" poisson in an brute-force-way.
        self.__calc_errors_alternative_near_simplified is the simplified version

        :param array_of_interest: numpy array of datapoints
        :return: list of [lower_error, upper_error]
        """
        
        # some constants
        used_lower = {0: 0.0, 1: 0.42364121373791264, 2: 0.7557519173057685, 3: 1.3496165388462822,
                      4: 2.114204734911637, 5: 2.964488162720907, 6: 3.8554518172724244, 7: 4.768089363121041,
                      8: 5.691397132377459, 9: 6.621373791263754, 10: 7.556018672890964, 11: 8.493664554851618,
                      12: 9.427309103034345, 13: 10.367289096365456, 14: 11.305935311770591, 15: 12.254251417139045,
                      16: 13.200566855618538, 17: 14.14754918306102, 18: 15.099199733244415, 19: 16.04818272757586,
                      20: 17.005835278426144, 21: 17.958152717572524, 22: 18.910470156718908, 23: 19.869456485495167,
                      24: 20.82710903634545, 25: 21.79076358786262, 26: 22.743081027009005, 27: 23.709069689896634
                      }
        used_upper = {0: 0.8396132044014671, 1: 2.5648549516505503, 2: 3.903465944452633, 3: 5.244664026665221,
                      4: 6.52384128042681, 5: 7.74448539398113, 6: 8.926443628311018, 7: 10.13103509229021,
                      8: 11.331467532309372, 9: 12.505835278426144, 10: 13.66171032706473, 11: 14.800793145198382,
                      12: 15.931584823984581, 13: 17.077208622982575, 14: 18.229214603941177, 15: 19.362740025881315,
                      16: 20.488829609869956, 17: 21.606727701711947, 18: 22.723381958864202, 19: 23.83094681572135,
                      20: 24.937042735538103, 21: 26.04188309162396, 22: 27.154889212951222, 23: 28.267731912181613,
                      24: 29.38055770055487, 25: 30.47282427475825, 26: 31.57456070860244, 27: 32.66483687876201
                      }
        
        def upper_lower_l(mu, intervall=0.68):
            @np.vectorize
            def __pp_f(x):
                return scst.poisson.pmf(math.floor(x), mu=mu)
            
            @np.vectorize
            def __pp_c(x):
                return scst.poisson.pmf(np.ceil(x), mu=mu)
            
            scan_num = 7000
            
            upper_limit = intervall + (1 - intervall) / 2.
            lower_limit = intervall / 2.
            
            pass_x_upper = mu
            pass_x_lower = 0.0
            if mu not in used_upper.keys():
                
                start_block = 0
                a = 0
                while True:
                    if a + __pp_f(start_block) <= upper_limit:
                        a += __pp_f(start_block)
                        start_block += 1
                        continue
                    break
                
                scan_x_upper = np.linspace(start_block, mu + np.sqrt(mu) + 1, scan_num)
                width = scan_x_upper[1] - scan_x_upper[0]
                for i in range(len(scan_x_upper)):
                    if a + __pp_f(scan_x_upper[i]) * width <= upper_limit:
                        a += __pp_f(scan_x_upper[i]) * width
                        pass_x_upper = scan_x_upper[i]
                    if a + __pp_f(scan_x_upper[i]) * width > upper_limit:
                        break
                
                used_upper[mu] = pass_x_upper
            
            if mu in used_upper.keys():
                pass_x_upper = used_upper[mu]
            
            if mu not in used_lower.keys():
                
                start_block = 0
                a = 0
                while True:
                    if a + __pp_c(start_block) <= lower_limit:
                        a += __pp_c(start_block)
                        start_block += 1
                        continue
                    break
                scan_x_lower = np.linspace(start_block - 1, mu, scan_num)
                width = scan_x_lower[1] - scan_x_lower[0]
                for i in range(len(scan_x_lower)):
                    if a + __pp_c(scan_x_lower[i]) * width <= lower_limit:
                        a += __pp_c(scan_x_lower[i]) * width
                        pass_x_lower = scan_x_lower[i] - 0.5
                    
                    if a + __pp_c(scan_x_lower[i]) * width > lower_limit:
                        break
                
                if mu == 0:
                    pass_x_lower = 0.0
                used_lower[mu] = pass_x_lower
            
            if mu in used_lower.keys():
                pass_x_lower = used_lower[mu]
            
            return [pass_x_upper - mu, mu - pass_x_lower]
        
        upper_limit_pass = np.array([upper_lower_l(num)[0] for num in array_of_interest])
        lower_limit_pass = np.array([upper_lower_l(num)[1] for num in array_of_interest])
        return [lower_limit_pass, upper_limit_pass]
    
    def fill_hist(self, name, array_of_interest, info=None, get_raw=False):
        
        """
        Function to fill a specific sub_histogramm manually

        :param name: 'data', 'mc_sig' or 'mc_bac'
        :param array_of_interest: numpy array that will be filled
        :param info: if 'mc_bac' or 'mc_sig': following list: [year, runs(s), process] (runs can be a list of strings)
        :param: save raw histogram data for recreating
        :param get_raw: True/False; saves raw data that is needed to recreate the histogram
        :return:
        """
        
        if name not in self.data.keys():
            self.set_empty_bins(name=name)
        
        if "data" in name:
            pile_array, _ = np.histogram(array_of_interest, bins=self.bins, density=False, range=self.range_interval)
            if not get_raw:
                self.data[name] += pile_array
            if get_raw:
                self.data[name] = {"raw_data": array_of_interest}
        
        if "mc" in name:
            year, run, process = info[0], info[1], info[2]
            pile_array, _ = np.histogram(array_of_interest, bins=self.bins, range=self.range_interval, density=False,
                                         weights=self.__correct_fac(len(array_of_interest), year, run, process))
            self.data[name] += pile_array
        
        if "mc_track" in name:
            year, run, process = info[0], info[1], info[2]
            pile_array, _ = np.histogram(array_of_interest, bins=self.bins, density=False, range=self.range_interval)
            if not get_raw:
                self.data[name] = {"raw_hist": pile_array, "corr_fac": self.__correct_fac(2, year, run, process)[0]}
            if get_raw:
                self.data[name] = {"raw_hist": pile_array, "corr_fac": self.__correct_fac(2, year, run, process)[0],
                                   "raw_data": array_of_interest}
        
        if "raw_undefined" in name:
            pile_array, _ = np.histogram(array_of_interest, bins=self.bins, density=False, range=self.range_interval)
            self.data[name] += pile_array
    
    def fill_hist_from_dir(self, columns, file_path, info=None, used_filter=None):
        
        """
        Function to fill the histogram from a directory.

        :param columns: string that contains a wanted column
        :param file_path: directory path
        :param info: contains a list of lists as following: [year(s), run(s)]
        :param used_filter: possible filter as following list: ['column_name', [a, b]]
                                'column_name': should be 'z1_mass', 'z2_mass' or 'mass_4l'
                                a: lower border; will remove everything below this value
                                b: upper border; will remove everything above this value
        """
        
        file_path = file_path
        file_list = [file_path + item for item in os.listdir(file_path)]
        
        if type(info).__name__ == "NoneType":
            years, runs = ["2012"], ["A-D"]
        else:
            years, runs = info[0], info[1]
        
        for item in file_list:
            data_frame = pd.read_csv(item, sep=";")
            if data_frame.empty:
                continue
            
            if "DiElectronDiMuon" in item or "2el2mu" in item:
                if not any(n in item for n in ["z1_mass", "z2_mass", "mass_4l", "event", "run"]):
                    col = [columns]
                else:
                    col = ["muon_" + columns, "electron_" + columns]
            else:
                col = [columns]
            
            if type(used_filter).__name__ != "NoneType":
                col.append(used_filter[0])
            
            array_to_fill = self.convert_column(data_frame.filter(col), used_filter=used_filter)
            if "CMS_Run20" in item:
                if ("2012" in item and "2012" in years) or ("2011" in item and "2011" in years):
                    pileup_array, _ = np.histogram(array_to_fill, bins=self.bins, density=False,
                                                   range=self.range_interval)
                    self.data["data"] += pileup_array
            if "MC_20" in item:
                for i in range(len(years)):
                    if years[i] in item:
                        if "_H_to" in item:
                            pileup_array, _ = np.histogram(array_to_fill, bins=self.bins, range=self.range_interval,
                                                           density=False,
                                                           weights=self.__correct_fac(len(array_to_fill), item[18:22],
                                                                                      runs[i], "H_ZZ"))
                            self.data["mc_sig"] += pileup_array
                        if "_H_to" not in item:
                            pileup_array, _ = np.histogram(array_to_fill, bins=self.bins, range=self.range_interval,
                                                           density=False,
                                                           weights=self.__correct_fac(len(array_to_fill), item[18:22],
                                                                                      runs[i],
                                                                                      "ZZ_" + item.split("_")[-3]))
                            self.data["mc_bac"] += pileup_array
    
    def draw_hist(self, stacked=None, **kwargs):
        
        """
        Function to draw the histogram created so far

        :param stacked: list of following order ['data', 'name_1', 'name_2'] if sub_histograms should be stacked
        :param kwargs: 'color', 'alpha', 'label' als list of same length as stacked or number of created empty bins
        """
        
        plot_tag = "none"
        pass_name = []
        if type(stacked).__name__ == "NoneType":
            pass_name = list(self.data.keys())
            plot_tag = "not_stacked"
        if type(stacked).__name__ != "NoneType":
            pass_name = stacked
            plot_tag = "stacked"
        
        color = kwargs.get("color", ["black", "blue", "red"])
        alpha = kwargs.get("alpha", [None for name in pass_name])
        label = kwargs.get("label", [None for name in pass_name])
        
        if "raw_undefined" in self.data.keys():
            plt.fill_between(self.x_range, self.data["raw_undefined"], step="mid")
        
        if "data" in pass_name[0]:
            pass_x, pass_y = np.array([]), np.array([])
            for i in range(len(self.x_range)):
                if self.data[pass_name[0]][i] != 0:
                    pass_x = np.append(pass_x, self.x_range[i])
                    pass_y = np.append(pass_y, self.data[pass_name[0]][i])
            plt.errorbar(pass_x, pass_y,
                         xerr=0, yerr=self.__calc_errors_poisson_near_cont(pass_y),
                         fmt="o", color="black", alpha=alpha[0], label=label[0])
        
        if "data" not in pass_name[0] and "raw_undefined" not in self.data.keys():
            sys.exit("Error: wrong form! -> 'data' first")
        
        if plot_tag == "not_stacked":
            for i in range(1, len(pass_name)):
                if "data" not in pass_name[i]:
                    plt.fill_between(self.x_range, self.data[pass_name[i]], step="mid", color=color[i],
                                     alpha=alpha[i], label=label[i], linewidth=0.0)
        
        if plot_tag == "stacked":
            plot_x_range = np.append([self.x_range[0] - self.bin_width],
                                     np.append(self.x_range, self.x_range[-1] + self.bin_width))
            
            plt.fill_between(plot_x_range, np.append([0], np.append(self.data[pass_name[1]], 0)),
                             step="mid", color=color[1], alpha=alpha[1], label=label[1], linewidth=0.0)
            temp_pileup = 0
            for i in range(2, len(pass_name)):
                temp_pileup += np.append([0], np.append(self.data[pass_name[i - 1]], 0))
                plt.fill_between(plot_x_range, np.append([0], np.append(self.data[pass_name[i]], 0)) + temp_pileup,
                                 temp_pileup, step="mid", color=color[2], alpha=alpha[2], label=label[2], linewidth=0.0)
    
    def save_to_csv(self, filename=None):
        
        """
        Function to save the histogram data (might be updated by also saving 'color', 'alpha' and 'label'

        :param filename: string like file name
        """
        
        data_init = pd.DataFrame(self.x_range, columns=["x_range"])
        for name_init in list(self.data.keys()):
            data_init[name_init] = self.data[name_init]
        if type(filename).__name__ == "NoneType":
            data_init.to_csv(os.path.join(self.save_dir, "hist_entry_data.csv"), index=False)
        if type(filename).__name__ == "str":
            data_init.to_csv(os.path.join(self.save_dir, filename), index=False)
    
    def get_data(self):
        
        """
        Function to create a pandas.DataFrame from the created histogram.

        :return: pandas.DataFrame
        """
        
        used_data_init = pd.DataFrame(self.x_range, columns=["x_range"])
        for name_init in list(self.data.keys()):
            used_data_init[name_init] = self.data[name_init]
        return used_data_init
    
    def show_hist(self, save_fig=None):
        
        """
        Function to show/save the histogram

        :param save_fig: filename if histogram should be saved
        :return: plot of histogram/file of the histogram
        """
        
        if type(save_fig).__name__ != "NoneType":
            plt.savefig(os.path.join(self.save_dir, save_fig))
        else:
            plt.show()


class Poly(object):
    x_mean = 0
    x_ms = "{x}-\\bar{{m}}_{{4\ell}}"
    
    own_dict = {"legendre_grade_0": {"f_name": "f_1 \\ ", "f_expr": "{0}P_=(" + x_ms + ")"},
                "legendre_grade_1": {"f_name": "f_1 \\ ", "f_expr": "\\sum_{{n=0}}^{{1}}a_{{n}}P_n(" + x_ms + ")"},
                "legendre_grade_2": {"f_name": "f_2 \\ ", "f_expr": "\\sum_{{n=0}}^{{2}}a_{{n}}P_n(" + x_ms + ")"},
                "legendre_grade_3": {"f_name": "f_3 \\ ", "f_expr": "\\sum_{{n=0}}^{{3}}a_{{n}}P_n(" + x_ms + ")"},
                "legendre_grade_4": {"f_name": "f_4 \\ ", "f_expr": "\\sum_{{n=0}}^{{4}}a_{{n}}P_n(" + x_ms + ")"},
                "legendre_grade_5": {"f_name": "f_5 \\ ", "f_expr": "\\sum_{{n=0}}^{{5}}a_{{n}}P_n(" + x_ms + ")"},
                "legendre_grade_6": {"f_name": "f_6 \\ ", "f_expr": "\\sum_{{n=0}}^{{6}}a_{{n}}P_n(" + x_ms + ")"}}
    
    def __init__(self):
        self.init = True
    
    @staticmethod
    def set_x_mean(value):
        Poly.x_mean = value
        Poly.x_ms = "{x}-\\bar{{m}}_{{4\ell}}"
    
    @staticmethod
    def legendre_custom(x, *args, **kwargs):
        x = x - Poly.x_mean
        
        if kwargs and args:
            sys.exit("Error: Wow you´re doing something really nasty. Reconsider!")
        if not kwargs and not args:
            sys.exit("Error: What are you expecting out of this?")
        
        if args:
            return np.polynomial.legendre.legval(x, [*args])
        if kwargs:
            return np.polynomial.legendre.legval(x, np.array(list(kwargs.values())))
    
    @staticmethod
    def legendre_grade_0(x, a):
        x = x - Poly.x_mean
        return np.polynomial.legendre.legval(x, [a])
    
    @staticmethod
    def legendre_grade_1(x, a, b):
        x = x - Poly.x_mean
        return np.polynomial.legendre.legval(x, [a, b])
    
    @staticmethod
    def legendre_grade_2(x, a=50. / 2500 + 0.05 / 2500 / 3, b=0, c=-0.05 / 2500 * 2 / 3):
        x = x - Poly.x_mean
        return np.polynomial.legendre.legval(x, [a, b, c])
    
    @staticmethod
    def legendre_grade_3(x, a, b, c, d):
        x = x - Poly.x_mean
        return np.polynomial.legendre.legval(x, [a, b, c, d])
    
    @staticmethod
    def legendre_grade_4(x, a, b, c, d, e):
        x = x - Poly.x_mean
        return np.polynomial.legendre.legval(x, [a, b, c, d, e])
    
    @staticmethod
    def legendre_grade_5(x, a, b, c, d, e, f):
        x = x - Poly.x_mean
        return np.polynomial.legendre.legval(x, [a, b, c, d, e, f])
    
    @staticmethod
    def legendre_grade_6(x, a, b, c, d, e, f, g):
        x = x - Poly.x_mean
        return np.polynomial.legendre.legval(x, [a, b, c, d, e, f, g])


class SignalFunc(object):
    own_dict = {"gauss": {"f_name": "g \\ ",
                          "f_expr": "\\frac{{1}}{{  \\sqrt{{2 \\pi}} {0}  }} e^{{ \\frac{{ ({x} - {1})^2  }}{{ 2 {0}^2  }}  }}"},
                "cauchy": {"f_name": "c \\ ",
                           "f_expr": "\\frac{{1}}{{\\pi}} \\cdot \\frac{{{0}}}{{{0}^2 + ({x} - {1})^2}}"},
                "single_side_crystal_ball": {"f_name": "SSCB \\ ",
                                             "f_expr": "\\rm{{single}}\\_\\rm{{sided}}\\_\\rm{{crystal}}\\_\\rm{{ball}}"},
                "DSCB": {"f_name": "DSCB \\ ",
                         "f_expr": "\\rm{{double}}\\_\\rm{{sided}}\\_\\rm{{crystal}}\\_\\rm{{ball}}"},
                "voigt": {"f_name": "v \\ ", "f_expr": "(g*c)({x})"}}
    
    def __init__(self):
        self.init = True
    
    x_mean = 0.0
    
    @staticmethod
    def set_x_mean(value):
        SignalFunc.x_mean = value
    
    @staticmethod
    def gauss(x, sigma=2.027, mu=124.807):
        x = x - SignalFunc.x_mean
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)
    
    @staticmethod
    def cauchy(x, s=0.824, t=124.484):
        x = x - SignalFunc.x_mean
        return (1. / np.pi) * (s / (s ** 2 + (x - t) ** 2))
    
    @staticmethod
    def single_side_crystal_ball(x, scale=1.2, loc=125.0, beta=1.1, m=1.2):
        x = x - SignalFunc.x_mean
        return scst.crystalball.pdf(x, beta=beta, m=m, loc=loc, scale=scale)
    
    @staticmethod
    @np.vectorize
    def DSCB(x, sigma=1.0, mu=125.0, alpha_l=0.5, alpha_r=0.5, n_l=1.1, n_r=1.1):
        t = (x - mu) / sigma
        lf = ((alpha_l / n_l) * ((n_l / alpha_l) - alpha_l - t)) ** (-n_l)
        rf = ((alpha_r / n_r) * ((n_r / alpha_r) - alpha_r + t)) ** (-n_r)
        
        n1 = np.sqrt(np.pi / 2.) * (math.erf(alpha_r / np.sqrt(2.)) + math.erf(alpha_l / np.sqrt(2.)))
        n2 = np.exp(-0.5 * alpha_l ** 2) * ((alpha_l / n_l) ** (-n_l)) * ((n_l / alpha_l) ** (-n_l + 1)) * (
                1. / (n_l - 1))
        n3 = np.exp(-0.5 * alpha_r ** 2) * ((alpha_r / n_r) ** (-n_r)) * ((n_r / alpha_r) ** (-n_r + 1)) * (
                1. / (n_r - 1))
        
        norm = 1. / (n1 + n2 + n3)
        norm = norm / sigma
        
        if -alpha_l <= t <= alpha_r:
            return norm * np.exp(-0.5 * t ** 2)
        if t < -alpha_l:
            return norm * np.exp(-0.5 * alpha_l ** 2) * lf
        if t > alpha_r:
            return norm * np.exp(-0.5 * alpha_r ** 2) * rf
    
    @staticmethod
    @np.vectorize
    def voigt(x, sigma=1.0, mu=125.0, gamma=1.0, ):
        z = ((x - mu) + 1j * gamma) / (np.sqrt(2) * sigma)
        f1_ = np.real(scsp.wofz(z).real)
        f2_ = sigma * np.sqrt(2 * np.pi)
        return f1_ / f2_


class GetRaw(object):
    
    def __init__(self, bins=14, hist_range=(109, 151), info=None, path_ru=None, path_mc=None):
        self.bins = bins
        self.hist_range = hist_range
        self.info = info
        self.path_mc = path_mc
        self.path_ru = path_ru
        if type(path_mc).__name__ == "NoneType":
            self.path_mc = "./data/mc_aftH/"
        if type(path_ru).__name__ == "NoneType":
            self.path_ru = "./data/ru_aftH/"
    
    def get_mc_raw(self, wanted_column="mass_4l", tag="Background"):
        hist = Hist(bins=self.bins, hist_range=self.hist_range)
        file_list = [os.path.join(self.path_mc, item) for item in os.listdir(self.path_mc)]
        
        years, run = "2012", "A-D"
        if type(self.info).__name__ != "NoneType":
            years, run = self.info[0], self.info[1]
        
        for item in file_list:
            data_frame = pd.read_csv(item, sep=";")
            if not data_frame.empty:
                wanted_array = np.array(data_frame[wanted_column].values, dtype=float)
                for year in years:
                    if year in item:
                        # noinspection PyTypeChecker
                        channel_ = item.split("_ZZ_to_4L_to_")[-1].split("_")[0]
                        if "_H_to_ZZ_" not in item and tag == "background":
                            hist_part_name_ = "mc_track_ZZ_{}_{}".format(channel_, year)
                            hist.fill_hist(name=hist_part_name_, array_of_interest=wanted_array,
                                           info=[year, run, "ZZ_{}".format(channel_)], get_raw=True)
                        if "_H_to" in item and tag == "signal":
                            hist_part_name_ = "mc_track_H_ZZ_{}_{}".format(channel_, year)
                            hist.fill_hist(name=hist_part_name_, array_of_interest=wanted_array,
                                           info=[year, run, "H_ZZ"], get_raw=True)
        return hist
    
    def get_mc_com(self):
        hist = Hist(bins=self.bins, hist_range=self.hist_range)
        hist.set_empty_bins("data")
        hist.set_empty_bins("mc_bac")
        hist.set_empty_bins("mc_sig")
        
        hist.fill_hist_from_dir(columns="mass_4l", file_path=self.path_mc, info=self.info)
        hist.fill_hist_from_dir(columns="mass_4l", file_path=self.path_ru, info=self.info)
        return hist
    
    def get_data_raw(self, wanted_column="mass_4l", year="2012"):
        hist = Hist(bins=self.bins, hist_range=self.hist_range)
        file_list = [os.path.join(self.path_ru, item) for item in os.listdir(self.path_ru)]
        for item in file_list:
            data_frame = pd.read_csv(item, sep=";")
            if not data_frame.empty:
                wanted_array = np.array(data_frame[wanted_column].values, dtype=float)
                hist.fill_hist(name="data_{}".format(item), array_of_interest=wanted_array, get_raw=True)
        
        to_pass_data = []
        for key in hist.data.keys():
            for v in hist.data[key]["raw_data"]:
                if (self.hist_range[0] <= v <= self.hist_range[1]): to_pass_data.append(v)
        hist.data = np.array(to_pass_data)
        return hist


class MCFitChi2NotSep(object):
    
    def __init__(self, bins=15, hist_range=(106, 151), tag="background", to_chi2_one=False,
                 error_type_model="sqrt_addition_absolute", info=None, path_mc=None, path_ru=None, verbose=True):
        self.verbose = verbose
        self.error_type_model = error_type_model
        self.save_dir = "./mc_fits"
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.to_chi2_one = to_chi2_one
        self.bins = bins
        self.hist_range = hist_range
        self.tag = tag
        self.info = info
        if self.tag == "background":
            self.data_raw = GetRaw(bins=self.bins, hist_range=self.hist_range, info=self.info,
                                   path_mc=path_mc).get_mc_raw(tag=self.tag)
            
            print(self.data_raw)
            input()
        
        if self.tag == "signal":
            self.data_raw = GetRaw(bins=self.bins, hist_range=self.hist_range, info=self.info,
                                   path_mc=path_mc).get_mc_raw(tag=self.tag)
        self.data_com = GetRaw(bins=self.bins, hist_range=self.hist_range, info=self.info, path_mc=path_mc,
                               path_ru=path_ru).get_mc_com()
        self.combined_errors = np.zeros(len(self.data_com.x_range))
        self.data_raw_fit_dict = {}
        self.func_com = {}
    
    def create_raw_fit(self, used_func, used_int_func=None):
        self.create_combined_mc_errors()
        initial_y_error = self.combined_errors
        temp_y_error = initial_y_error
        error_scale_factor = 1.0
        last_used_temp_y_error = temp_y_error
        last_updated_cost_ndf = 0
        dummy = 1
        lock = False
        lock_count = 0
        flip = False
        if self.error_type_model != "relative":
            error_scale_factor = 0.0
            dummy = 2
        while True:
            temp_hist = K2.HistContainer(n_bins=self.data_raw.bins,
                                         bin_range=self.data_raw.range_interval,
                                         bin_edges=list(self.data_raw.range))
            if self.tag == "background":
                temp_hist.set_bins(self.data_com.data["mc_bac"])
            if self.tag == "signal":
                temp_hist.set_bins(self.data_com.data["mc_sig"])
            
            temp_hist.add_simple_error(err_val=temp_y_error)
            temp_hist_fit = K2.HistFit(data=temp_hist,
                                       model_density_function=used_func,
                                       model_density_antiderivative=used_int_func,
                                       cost_function=K2.HistCostFunction_Chi2(errors_to_use="covariance"),
                                       minimizer="iminuit")
            
            if used_func.__name__ == "gauss":
                temp_hist_fit.limit_parameter("mu", (110, 135))
                temp_hist_fit.limit_parameter("sigma", (0.25, 20.0))
            if used_func.__name__ == "cauchy":
                temp_hist_fit.limit_parameter("t", (109, 142))
                temp_hist_fit.limit_parameter("s", (0.25, 20.0))
            if used_func.__name__ == "single_side_crystal_ball":
                temp_hist_fit.limit_parameter("m", (1.0001, 10.0))
                temp_hist_fit.limit_parameter("beta", (0.0001, 10.0))
                temp_hist_fit.limit_parameter("loc", (109.0, 142.0))
                temp_hist_fit.limit_parameter("scale", (0.25, 10.0))
            if used_func.__name__ == "DSCB":
                temp_hist_fit.limit_parameter("sigma", (0.15, 10.0))
                temp_hist_fit.limit_parameter("mu", (120.0, 130.0))
                temp_hist_fit.limit_parameter("alpha_l", (0.0001, 15.0))
                temp_hist_fit.limit_parameter("alpha_r", (0.0001, 15.0))
                temp_hist_fit.limit_parameter("n_l", (1.0001, 15.0))
                temp_hist_fit.limit_parameter("n_r", (1.0001, 15.0))
            if used_func.__name__ == "voigt":
                temp_hist_fit.limit_parameter("mu", (124.0, 126.0))
            
            temp_hist_fit.do_fit()
            
            ndf = float(len(temp_hist.data) - len(temp_hist_fit.parameter_values))
            initial_c_ndf = temp_hist_fit.cost_function_value / ndf
            
            if self.to_chi2_one and self.error_type_model == "relative":
                c_ndf = temp_hist_fit.cost_function_value / ndf
                if c_ndf > 1.0:
                    if not lock:
                        error_scale_factor = np.sqrt(c_ndf - 0.0001)
                        lock = True
                    error_scale_factor += 10 ** (-dummy)
                    last_used_temp_y_error = temp_y_error
                    temp_y_error = np.array(initial_y_error) * error_scale_factor
                
                if c_ndf < 1.0:
                    error_scale_factor -= 1 * 10 ** (-dummy)
                    temp_y_error = np.array(initial_y_error) * error_scale_factor
                
                if (abs(c_ndf - 1) < 0.0001 and c_ndf > 1.0):
                    break
                if last_updated_cost_ndf > 1 and c_ndf < 1:
                    dummy += 1
                
                print(error_scale_factor, c_ndf)
                last_updated_cost_ndf = temp_hist_fit.cost_function_value / ndf
            
            if self.to_chi2_one and "sqrt_addition" in self.error_type_model:
                ndf = float(len(temp_hist.data) - len(temp_hist_fit.parameter_values))
                c_ndf = temp_hist_fit.cost_function_value / ndf
                if c_ndf > 1.0:
                    error_scale_factor += 10 ** (-dummy)
                    last_used_temp_y_error = temp_y_error
                    if self.error_type_model == "sqrt_addition_absolute":
                        temp_y_error = np.sqrt((error_scale_factor) ** 2 + initial_y_error ** 2)
                    if self.error_type_model == "sqrt_addition_relative":
                        temp_y_error = np.sqrt((error_scale_factor * initial_y_error) ** 2 + initial_y_error ** 2)
                if c_ndf < 1.0:
                    temp_y_error = last_used_temp_y_error
                    error_scale_factor -= 1 * 10 ** (-dummy)
                    if error_scale_factor < 0.0:
                        error_scale_factor = 0.0
                        dummy += 2
                    dummy += 1
                if (abs(c_ndf - 1) < 0.000001 and c_ndf > 1.0) or last_updated_cost_ndf == c_ndf:
                    break
                if last_updated_cost_ndf == c_ndf:
                    lock_count += 1
                last_updated_cost_ndf = temp_hist_fit.cost_function_value / ndf
            if not self.to_chi2_one:
                break
        error_ratio = error_scale_factor
        if self.to_chi2_one and self.verbose:
            print("{}: {} {}".format(used_func.__name__, error_scale_factor, last_updated_cost_ndf))
        
        self.data_raw_fit_dict[used_func.__name__] = {
            "used_hist": temp_hist,
            "fit": temp_hist_fit,
            "bin_width": self.data_raw.bin_width,
            "error_scale_fac": error_ratio,
            "initial_chi2_ndf": initial_c_ndf,
            "scale_factor": error_scale_factor}
    
    def get_raw_fit_results_as_dict(self, used_func, used_int_func=None):
        if used_func.__name__ not in list(self.data_raw_fit_dict.keys()):
            self.create_raw_fit(used_func=used_func, used_int_func=used_int_func)
        temp_fit_results = self.data_raw_fit_dict[used_func.__name__]["fit"]
        return_dict = {"parameter_values": temp_fit_results.parameter_values,
                       "parameter_errors": temp_fit_results.parameter_errors,
                       "asymmetric_parameter_errors": temp_fit_results.asymmetric_parameter_errors}
        return return_dict
    
    def print_report(self, used_func):
        self.data_raw_fit_dict[used_func.__name__]["fit"].report(asymmetric_parameter_errors=False)
    
    def save_report_to_yaml(self):
        for func_name in list(self.data_raw_fit_dict.keys()):
            temp_name = "fit_results.yml"
            if not self.to_chi2_one:
                temp_name = "fit__kind_{}__year_{}__used_func_{}__not_scaled.yml".format(self.tag, 2012, func_name)
            
            if self.to_chi2_one:
                temp_name = "fit__kind_{}__year_{}__usedFunc_{}__scaleKind_{}__InitChiNdf_{}__scaleFac_{}.yml".format(
                    self.tag, 2012, func_name, self.error_type_model,
                    round(self.data_raw_fit_dict[func_name]["initial_chi2_ndf"], 9),
                    round(self.data_raw_fit_dict[func_name]["scale_factor"], 9))
            
            self.data_raw_fit_dict[func_name]["fit"].to_file(os.path.join(self.save_dir, temp_name),
                                                             calculate_asymmetric_errors=True)
    
    def create_combined_mc_errors(self):
        for i in range(len(self.data_raw.x_range)):
            pass_y_error_temp = 0
            for key in list(self.data_raw.data.keys()):
                corr_fac_square = (self.data_raw.data[key]["corr_fac"]) ** 2
                pass_y_error_temp += (self.data_raw.data[key]["raw_hist"][i] * corr_fac_square)
            self.combined_errors[i] = np.sqrt(pass_y_error_temp)
    
    def get_combined_mc_errors(self):
        if len(self.combined_errors) == 0:
            self.create_combined_mc_errors()
        return self.combined_errors
    
    def create_total_function(self, used_func, used_int_func=None):
        if used_func.__name__ not in list(self.data_raw_fit_dict.keys()):
            self.create_raw_fit(used_func=used_func, used_int_func=used_int_func)
        
        fit_details = self.get_raw_fit_results_as_dict(used_func=used_func,
                                                       used_int_func=used_int_func)
        
        def return_func(x):
            f = used_func(x, *fit_details["parameter_values"])
            return f
        
        self.func_com[used_func.__name__] = return_func
    
    def get_total_function(self, used_func=None, used_int_func=None):
        if used_func.__name__ not in list(self.func_com.keys()):
            self.create_total_function(used_func=used_func, used_int_func=used_int_func)
        return self.func_com[used_func.__name__]
    
    def create_mc_fit_plots(self, used_func, with_ratio=False, plot_conture=False, to_file=False, separate=True,
                            used_dict=None,
                            make_title=False):
        
        if type(used_func).__name__ != "list":
            if type(used_func).__name__ != "str":
                used_func = [used_func]
        
        if type(used_dict).__name__ == "NoneType":
            sys.exit("used_dict have to be {'f_name': '<function name>', 'f_expr': '<function expression>'}")
        
        fit_show_list = []
        fit_func_list = []
        fit_save_func_list = []
        for func_name in list(used_dict.keys()):
            for func in used_func:
                if func.__name__ == func_name:
                    exec('fit_{} = self.data_raw_fit_dict[func_name]["fit"]'.format(func_name))
                    exec('fit_{}.assign_model_function_latex_name(used_dict[func_name]["f_name"])'.format(func_name))
                    if "grade_0" in func_name:
                        exec('fit_{}.assign_parameter_latex_names(x="x", a="a_0")'.format(func_name))
                    if "grade_1" in func_name:
                        exec('fit_{}.assign_parameter_latex_names(x="x", a="a_0", b="a_1")'.format(func_name))
                    if "grade_2" in func_name:
                        exec('fit_{}.assign_parameter_latex_names(x="x", a="a_0", b="a_1", c="a_2")'.format(func_name))
                    if "grade_3" in func_name:
                        exec('fit_{}.assign_parameter_latex_names(x="x", a="a_0", b="a_1", c="a_2", d="a_3")'.format(
                            func_name))
                    if "grade_4" in func_name:
                        exec(
                            'fit_{}.assign_parameter_latex_names(x="x", a="a_0", b="a_1", c="a_2", d="a_3", e="a_4")'.format(
                                func_name))
                    if "grade_5" in func_name:
                        exec(
                            'fit_{}.assign_parameter_latex_names(x="x", a="a_0", b="a_1", c="a_2", d="a_3", e="a_4", f="a_5")'.format(
                                func_name))
                    if "grade_6" in func_name:
                        exec(
                            'fit_{}.assign_parameter_latex_names(x="x", a="a_0", b="a_1", c="a_2", d="a_3", e="a_4", f="a_5", g="a_6")'.format(
                                func_name))
                    if "cauchy" in func_name:
                        exec(r'fit_{}.assign_parameter_latex_names(x="x", s="s", t=r"\mu")'.format(func_name))
                    if "gauss" in func_name:
                        exec(r'fit_{}.assign_parameter_latex_names(x="x", sigma=r"\sigma", mu="\mu")'.format(func_name))
                    if "DSCB" in func_name:
                        exec(
                            r'fit_{}.assign_parameter_latex_names(x="x", sigma=r"\sigma", mu=r"\mu", alpha_l=r"\alpha_L", alpha_r=r"\alpha_R", n_l=r"n_L", n_r=r"n_r")'.format(
                                func_name))
                    if "single_side_crystal_ball" in func_name:
                        exec(
                            r'fit_{}.assign_parameter_latex_names(x="x", beta=r"\beta", m=r"m", loc="\mu", scale=r"\sigma")'.format(
                                func_name))
                    if "voigt" in func_name:
                        exec(
                            'fit_{}.assign_parameter_latex_names(x="x", sigma=r"\sigma", mu=r"\mu", gamma=r"\gamma")'.format(
                                func_name))
                    
                    exec('fit_{}.assign_model_function_latex_expression(used_dict[func_name]["f_expr"])'.format(
                        func_name))
                    exec('fit_show_list.append(fit_{})'.format(func_name))
                    fit_func_list.append(used_dict[func_name]["f_name"])
                    fit_save_func_list.append(func_name)
        
        h_plot = K2.Plot(fit_objects=fit_show_list, separate_figures=False)
        
        for item in fit_show_list:
            item._model_function._formatter._latex_x_name = "m_{4\ell}"
        
        if len(fit_show_list) == 1:
            h_plot.customize("data", "label", ["MC Simulation"])
            h_plot.customize("model_density", "label", ["Dichte"])
            h_plot.customize("model", "label", ["Modell"])
        
        if len(fit_show_list) > 1:
            label_list = [None for item in fit_show_list]
            # noinspection PyTypeChecker
            label_list[0] = "MC Simulation"
            label_color = ["k" for item in fit_show_list]
            h_plot.customize("data", "label", label_list).customize("data", "color", label_color)
        
        h_plot.plot(with_ratio=with_ratio, ratio_range=(0.0, 2.0))
        if with_ratio:
            h_plot.axes[0]["ratio"].set_xlabel(r"$m_{4\ell}$ in GeV")
            h_plot.axes[0]["ratio"].hlines(1.0, self.hist_range[0], self.hist_range[1], color="black", alpha=1,
                                           linewidth=0.75)
        
        if not with_ratio:
            h_plot.axes[0]["main"].set_xlabel(r"$m_{4\ell}$ in GeV")
        h_plot.axes[0]["main"].set_ylabel("Bineinträge")
        
        fig = h_plot.figures[-1]
        
        if make_title:
            fig.set_tight_layout(False)
        
        if self.to_chi2_one and make_title:
            used_scale_fac = ""
            for key in self.data_raw_fit_dict.keys():
                used_scale_fac += "$k_{{\\sigma, {}}}$ =  {}, ".format(used_dict[key]["f_name"],
                                                                       round(self.data_raw_fit_dict[key][
                                                                                 "error_scale_fac"], 3))
            
            fig.suptitle("{}, {} channel: $4\\mu, \\ 4e, \\ 2e2\\mu$ \n {}".format(2012, self.tag, used_scale_fac[:-1]))
        if not self.to_chi2_one and make_title:
            fig.suptitle("{}, {}, used channel: $4\\mu + \\ 4e + \\ 2e2\\mu$".format(2012, self.tag))
        
        if to_file:
            
            if separate:
                for item in fit_save_func_list:
                    if not self.to_chi2_one:
                        temp_name = "kind_{}__year_{}__used_func_{}__not_scaled.png".format(
                            self.tag, 2012, item)
                        plt.savefig(os.path.join(self.save_dir, temp_name))
                    if self.to_chi2_one:
                        temp_name = "kind_{}__year_{}__usedFunc_{}__scaleKind_{}__InitChiNdf_{}__scaleFac_{}.png".format(
                            self.tag, 2012, item,
                            self.error_type_model, round(self.data_raw_fit_dict[item]["initial_chi2_ndf"], 9),
                            round(self.data_raw_fit_dict[item]["scale_factor"], 9))
                        plt.savefig(os.path.join(self.save_dir, temp_name))
            
            if not separate:
                used_func_for_plot = ""
                used_chi2Ndf = ""
                used_scale_fac = ""
                for item, in fit_save_func_list:
                    used_func_for_plot += "{}_".format(item)
                    used_chi2Ndf += "{}_".format(round(self.data_raw_fit_dict[item]["initial_chi2_ndf"], 9))
                    used_scale_fac += "{}_".format(round(self.data_raw_fit_dict[item]["scale_factor"], 9))
                    if not self.to_chi2_one:
                        temp_name = "kind_{}__year_{}__used_func_{}__not_scaled.png".format(
                            self.tag, 2012, used_func_for_plot[:-1])
                        plt.savefig(os.path.join(self.save_dir, temp_name))
                    
                    if self.to_chi2_one:
                        temp_name = "kind_{}__year_{}__usedFunc_{}__scaleKind_{}__InitChiNdf_{}__scaleFac_{}.png".format(
                            self.tag, 2012, used_func_for_plot[:-1],
                            self.error_type_model, used_chi2Ndf, used_scale_fac)
                        plt.savefig(os.path.join(self.save_dir, temp_name))
        if not to_file:
            plt.show()
        
        if plot_conture:
            for item, func_name, str_func_name in zip(fit_show_list, fit_func_list, fit_save_func_list):
                cpf = K2.ContoursProfiler(item, contour_sigma_values=(1, 2))
                cpf.plot_profiles_contours_matrix()
                fig_cpf = cpf.figures[-1]
                fig_cpf.set_tight_layout(False)
                fig_cpf.suptitle("{}, {}, used channel: 4\\mu + 2e2\\mu + 4e \n Function: {}".format(2012, self.tag,
                                                                                                     func_name.split(
                                                                                                         " ")[0]))
                
                if to_file:
                    plt.savefig(os.path.join(self.save_dir, "{}_{}_{}.png".format(self.tag, 2012, str_func_name)))
                if not to_file:
                    plt.show()


class StatTest(object):
    """
    Class for statistical tests (p0, signal strength alpha_S or mu_H)
    """
    
    def __init__(self, bins, hist_range, to_chi2_one=False,
                 info=None, path_ru=None, path_mc=None,
                 signal_func=None, background_func=None, verbose=True):
        self.save_dir = "./stat_tests"
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.verbose = verbose
        self.info = info
        self.path_ru = path_ru
        self.path_mc = path_mc
        self.data_com = GetRaw(bins=bins, hist_range=hist_range, info=self.info, path_ru=path_ru,
                               path_mc=path_mc).get_mc_com()
        self.mc_bac_fit = 0
        self.mc_sig_fit = 0
        self.bin_width = (hist_range[1] - hist_range[0]) / bins
        self.bins = bins
        self.hist_range = hist_range
        self.to_chi2_one = to_chi2_one
        self.func_func_param = {}
        self.bac_func_mc_errors = {}
        self.sig_func_mc_errors = {}
        self.data_raw = GetRaw(bins=self.bins * 1000, hist_range=self.hist_range, info=self.info, path_ru=self.path_ru,
                               path_mc=self.path_mc).get_data_raw()
        
        Poly.set_x_mean((self.hist_range[0] + self.hist_range[1]) / 2.)
        self.calculated_dump = {}
        
        self.func_names = {}
        
        self.signal_func = signal_func
        self.background_func = background_func
        if type(self.signal_func).__name__ == "NoneType" or type(self.background_func).__name__ == "NoneType":
            sys.exit("Error: signal and background functions need to be specified!")
        
        self.__set_func_val()
    
    def __set_func_val(self):
        """
        Function to set variables for used functions from fit.
        Can be overwritten by get_background_func_param_from_fit or/and get_signal_func_param_from_fit, but its
        more time consuming when the fit function is complicated.
        If the given Function is not in the list it wont work

        :return: self.func_func_param dict
        """
        
        year = self.info[0]
        if len(year) == 1:
            self.func_func_param["legendre_grade_2"] = np.array(
                [0.02428954322341774, 0.00020021879360914812, -8.663487231700586e-06])
            
            self.func_func_param["gauss"] = np.array([1.9727411383681588, 124.81209770469063])
            self.func_func_param["DSCB"] = np.array(
                [1.3388805157158652, 124.98071139688055, 0.9278644003292827, 1.1053232991255295, 3.2720915177175414,
                 6.771701327613522])
            
            self.func_names["legendre_grade_2"] = r"$\rm{LegendrePoly}_{n=2}$"
            self.func_names["DSCB"] = r"$\rm{DSCB}$"
    
    def __create_mh_scan_array_for_p0(self):
        to_pass_array = np.array([])
        to_pass_array = np.append(to_pass_array, np.arange(max(self.hist_range[0], 106), 124.5, 0.25))  # 0.25))
        to_pass_array = np.append(to_pass_array, np.arange(124.75, 126.0, 0.25))  # 0.25))
        to_pass_array = np.append(to_pass_array, np.arange(126.5, min(self.hist_range[1], 151), 0.5))  # 0.25))
        return to_pass_array
    
    @staticmethod
    def __create_gauss_sigma(passed_mass_array):
        """
        Function to create a sigma array that corresponds to a given mass array based on a function that is created
        from a fit of different sigmas from simulation. Parametrisation function may be updated.

        :param passed_mass_array: numpy array of used mass points for the p0 scan
        :return: numpy array of sigmas that corresponds to mass points
        """
        
        def used_func(x):
            a, b = 2.079854189377624, 0.013654169521349959
            return a + (x - 130.0564929849343) * b
        
        return used_func(passed_mass_array)
    
    @staticmethod
    def __create_3_interval_array(start=0.0, end=1.0, point_1=0.25, point_2=0.75, num_i1=100, num_i2=100, num_i3=100):
        """
        :param start: startpoint (float)
        :param end: endpoint (float)
        :param point_1: first breakpoint (float)
        :param point_2: second breakpoint (float)
        :param num_i1: number of points in the first interval (int)
        :param num_i2: number of points in the second interval (int)
        :param num_i3: number of points in the third interval (int)
        :return: numpy array with length (num_i1 + num_i2 + num_i3)
        """
        
        pass_array = np.concatenate([np.linspace(start=start, stop=point_1, num=num_i1, endpoint=False),
                                     np.linspace(start=point_1, stop=point_2, num=num_i2, endpoint=False),
                                     np.linspace(start=point_2, stop=end, num=num_i3, endpoint=True)])
        return pass_array
    
    @staticmethod
    def __get_inters_points(scan_array, value_array, num=1.0):
        """
        Function to calculate the points where -2ln(L) is equal/near to num

        :param scan_array: -2ln(L) array that corresponds to value_array
        :param value_array: array of variable of interest
        :param num: 1 for 1*sigma or 4 for 3*sigma
        :return: list of 3 points [left point equal to num, minimum, right point equal to num]
        """
        
        left_part = value_array[:np.argmin(value_array)]
        right_part = value_array[np.argmin(value_array):]
        
        left_p = scan_array[np.argmin(abs(left_part - num))]
        right_p = scan_array[np.argmin(abs(right_part - num)) + len(left_part)]
        middle_p = scan_array[np.argmin(value_array)]
        return [left_p, middle_p, right_p]
    
    # Get func param if not in __set_func_val
    # ------------------------------------------------
    def get_background_func_param_from_fit(self):
        """
        If you need to update the fitted values set in __set_func_val or the wanted funtion is not listed there.

        :return: fill func_func_param dict with corresponding name and values
        """
        
        if self.verbose:
            print("Fit range of background MC is ({}, {})".format(self.hist_range[0], self.hist_range[1]))
            print("number of bins: {}".format(3 * self.bins))
        self.mc_bac_fit = MCFitChi2NotSep(bins=3 * self.bins, hist_range=self.hist_range, tag="background",
                                          to_chi2_one=self.to_chi2_one, info=self.info)
        used_func = []
        if type(self.background_func).__name__ != "list":
            used_func = [self.background_func]
        
        for func in used_func:
            self.mc_bac_fit.create_total_function(used_func=func)
            self.func_func_param[func.__name__] = self.mc_bac_fit.data_raw_fit_dict[func.__name__][
                "fit"].parameter_values
            self.bac_func_mc_errors[func.__name__] = self.mc_bac_fit.data_raw_fit_dict[func.__name__]["used_hist"].err
            self.mc_bac_fit.save_report_to_yaml()
            if self.verbose:
                print("Fitted parameter to function {}: {}".format(func.__name__, self.func_func_param[func.__name__]))
    
    def get_signal_func_param_from_fit(self):
        """
        If you need to update the fitted values set in __set_func_val or the wanted funtion is not listed there.

        :return: fill func_func_param dict with corresponding name and values
        """
        
        if self.verbose:
            print("Fit range of signal MC is ({}, {})".format(106, 145))
            print("number of bins: {}".format(4 * self.bins))
        self.mc_sig_fit = MCFitChi2NotSep(bins=4 * self.bins, hist_range=(106, 145), tag="signal",
                                          to_chi2_one=self.to_chi2_one, info=self.info)
        used_func = []
        if type(self.signal_func).__name__ != "list":
            used_func = [self.signal_func]
        
        for func in used_func:
            self.mc_sig_fit.create_total_function(used_func=func)
            self.func_func_param[func.__name__] = self.mc_sig_fit.data_raw_fit_dict[func.__name__][
                "fit"].parameter_values
            self.sig_func_mc_errors[func.__name__] = self.mc_sig_fit.data_raw_fit_dict[func.__name__]["used_hist"].err
            self.mc_sig_fit.save_report_to_yaml()
            if self.verbose:
                print("Fitted parameter to function {}: {}".format(func.__name__, self.func_func_param[func.__name__]))
    
    # Calc alpha
    # ------------------------------------------------
    def calc_alpha_binned_integrated(self):
        """
        Function to calculate signal strength alpha_s in a binned variant.
        Attention: might be wrong
        #TODO: check if calculated values are consistent - consistent; will not be mentioned in thesis

        :return: fill calculated_dump dict with calculated results
        """
        
        print("calc_alpha_binned_integrated")
        signal_func = self.signal_func
        background_func = self.background_func
        
        if "calc_alpha_binned_integrated" not in list(self.calculated_dump.keys()):
            self.calculated_dump["calc_alpha_binned_integrated"] = {}
        
        alpha = np.linspace(0, 1.0, 1000)
        
        used_x_point = np.array(self.data_com.x_range)
        used_d_point = self.data_com.data["data"]
        n_mc_b = np.sum(self.data_com.data["mc_bac"])
        n_mc_s = np.sum(self.data_com.data["mc_sig"])
        
        signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                         args=tuple(self.func_func_param[signal_func.__name__]))[0]
        background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                             args=tuple(self.func_func_param[background_func.__name__]))[0]
        
        used_b_int_bac = {}
        used_b_int_sig = {}
        
        @np.vectorize
        def b_int_bac(x_point):
            bac_func_args = self.func_func_param[background_func.__name__]
            area = 0.0
            if x_point not in used_b_int_bac.keys():
                area = background_func_norm * sci.quad(background_func,
                                                       x_point - self.bin_width / 2.,
                                                       x_point + self.bin_width / 2.,
                                                       args=tuple(bac_func_args))[0]
                used_b_int_bac[x_point] = area
            
            if x_point in used_b_int_bac.keys():
                area = used_b_int_bac[x_point]
            
            return area
        
        @np.vectorize
        def b_int_sig(x_point):
            sig_func_args = self.func_func_param[signal_func.__name__]
            area = 0.0
            if x_point not in used_b_int_sig.keys():
                area = signal_func_norm * sci.quad(signal_func,
                                                   x_point - self.bin_width / 2., x_point + self.bin_width / 2.,
                                                   args=tuple(sig_func_args))[0]
                used_b_int_sig[x_point] = area
            
            if x_point in used_b_int_sig.keys():
                area = used_b_int_sig[x_point]
            
            return area
        
        fac = np.sum(used_d_point)  # (n_mc_s + n_mc_b)
        
        @np.vectorize
        def used_func_(alpha_point, x_point_):
            
            return alpha_point * b_int_sig(x_point_) + (1 - alpha_point) * b_int_bac(x_point_)
        
        p = np.meshgrid(alpha, used_x_point)
        # noinspection PyTypeChecker
        ln = np.sum(
            np.multiply(used_d_point * self.bin_width, np.log(fac * used_func_(p[0], p[1])).T) - fac * used_func_(p[0],
                                                                                                                  p[
                                                                                                                      1]).T,
            axis=1)
        
        draw_ln = - 2 * (ln - np.amax(ln))
        
        if self.verbose:
            print("Range=({},{}), Bins={}, Functions: '{};{}', Signifikanz={}".format(self.hist_range[0],
                                                                                      self.hist_range[1],
                                                                                      self.bins,
                                                                                      signal_func.__name__,
                                                                                      background_func.__name__,
                                                                                      round(np.sqrt(draw_ln[0]), 7)))
        
        self.calculated_dump["calc_alpha_binned_integrated"] = {
            "alpha_array": alpha, "nll_array": draw_ln,
            "sigma_min_sigma": self.__get_inters_points(alpha, draw_ln),
            "significance": np.sqrt(draw_ln[0])}
    
    def calc_alpha_unbinned_point_wise(self, scan_points=500, significance_only=False):
        """
        Function to calculate the signal strength alpha with the undbinned extended likelihood variant

        :param scan_points: number of points that the arrays will have during the funtion
        :param significance_only: for error estimation only
        :return: fill calculated_dump dict with calculated results or return significance if set so
        """
        
        print("calc_alpha_unbinned_point_wise")
        signal_func = self.signal_func
        background_func = self.background_func
        
        if "calc_alpha_unbinned_point_wise" not in list(self.calculated_dump.keys()):
            self.calculated_dump["calc_alpha_unbinned_point_wise"] = {}
        
        used_x_point = self.data_raw.data
        alpha = np.array([])
        if significance_only:
            alpha = self.__create_3_interval_array(start=0, end=1, point_1=0.075, point_2=0.65,
                                                   num_i1=int(scan_points * (1. / 24)),
                                                   num_i2=int(scan_points * (22. / 24)),
                                                   num_i3=int(scan_points * (1. / 24)))
        if not significance_only:
            alpha = self.__create_3_interval_array(start=0, end=1, point_1=0.075, point_2=0.65,
                                                   num_i1=int(scan_points * (1. / 4)),
                                                   num_i2=int(scan_points * (2. / 4)),
                                                   num_i3=int(scan_points * (1. / 4)))
        
        sig_param = self.func_func_param[signal_func.__name__]
        bac_param = self.func_func_param[background_func.__name__]
        
        background_func_norm = 1. / \
                               sci.quad(background_func, self.hist_range[0], self.hist_range[1], args=tuple(bac_param))[
                                   0]
        signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1], args=tuple(sig_param))[0]
        bac_func_args = self.func_func_param[background_func.__name__]
        
        used_b_point_bac = {}
        used_b_point_sig = {}
        
        @np.vectorize
        def b_point_bac(x_point):
            
            if x_point not in used_b_point_bac.keys():
                pass_val = background_func_norm * background_func(x_point, *bac_func_args)
                used_b_point_bac[x_point] = pass_val
                return pass_val
            
            if x_point in used_b_point_bac.keys():
                pass_val = used_b_point_bac[x_point]
                return pass_val
        
        @np.vectorize
        def b_point_sig(x_point):
            
            if x_point not in used_b_point_sig.keys():
                pass_val = signal_func_norm * signal_func(x_point, *sig_param)
                used_b_point_sig[x_point] = pass_val
                return pass_val
            
            if x_point in used_b_point_sig.keys():
                pass_val = used_b_point_sig[x_point]
                
                return pass_val
        
        @np.vectorize
        def used_func(alpha_point, x_point):
            return alpha_point * b_point_sig(x_point) + (1 - alpha_point) * b_point_bac(x_point)
        
        used_exp_val = {}
        
        @np.vectorize
        def expectation_value(alpha_point):
            def to_int_func(x_point):
                return x_point * (alpha_point * b_point_sig(x_point) + (1 - alpha_point) * b_point_bac(x_point))
            
            pass_exp_val = np.sum(self.data_com.data["data"])
            if alpha_point not in used_exp_val.keys():
                pass_exp_val = sci.quad(to_int_func, self.hist_range[0], self.hist_range[1],
                                        points=np.linspace(self.hist_range[0], self.hist_range[1], 42),
                                        epsrel=1e-16)[0]
                
                used_exp_val[alpha_point] = pass_exp_val
            
            if alpha_point in used_exp_val.keys():
                pass_exp_val = used_exp_val[alpha_point]
            
            return pass_exp_val
        
        p = np.meshgrid(alpha, used_x_point)
        exp_val = expectation_value(alpha)
        ln = - exp_val + np.sum(np.log(np.multiply(exp_val, used_func(p[0], p[1]))), axis=0)
        draw_ln = - 2 * (ln - np.amax(ln))
        
        if significance_only:
            return np.sqrt(draw_ln[0])
        
        self.calculated_dump["calc_alpha_unbinned_point_wise"] = {
            "alpha_array": alpha, "nll_array": draw_ln,
            "sigma_min_sigma": self.__get_inters_points(alpha, draw_ln),
            "significance": np.sqrt(draw_ln[0])}
        
        if self.verbose:
            print("Range=({},{}), Bins={}, Functions: '{};{}', Signifikanz={}".format(self.hist_range[0],
                                                                                      self.hist_range[1],
                                                                                      self.bins,
                                                                                      signal_func.__name__,
                                                                                      background_func.__name__,
                                                                                      round(np.sqrt(draw_ln[0]), 7)))
    
    # Calc signal strength
    # ------------------------------------------------
    def calc_signal_strength_binned_integrated(self):
        """
        Function to calculate the signal strength mu_H to check if the corresponding signal MC is correct.
        binned variant

        :return: fill calculated_dump dict with calculated results
        """
        
        print("calc_q0_binned_integrated")
        mu_len = 89
        signal_func = self.signal_func
        background_func = self.background_func
        
        n_mc_b = np.sum(self.data_com.data["mc_bac"])
        n_mc_s = np.sum(self.data_com.data["mc_sig"])
        used_x_point = self.data_com.x_range
        used_d_point = self.data_com.data["data"]
        
        background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                             args=tuple(self.func_func_param[background_func.__name__]),
                                             epsrel=1e-16)[0]
        background_func_param = self.func_func_param[background_func.__name__]
        signal_func_param = self.func_func_param[signal_func.__name__]
        signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                         args=tuple(signal_func_param), epsrel=1e-16)[0]
        
        used_bac_int = {}
        used_sig_int = {}
        
        @np.vectorize
        def b_int_bac(_x):
            def to_int_func(_xx):
                return background_func_norm * background_func(_xx, *background_func_param)
            
            if _x not in used_bac_int.keys():
                area = sci.quad(to_int_func, _x - self.bin_width / 2., _x + self.bin_width / 2.,
                                epsrel=1e-16)[0]
                used_bac_int[_x] = area
                return area * n_mc_b * self.bin_width
            
            if _x in used_bac_int.keys():
                area = used_bac_int[_x]
                return area * n_mc_b * self.bin_width
        
        @np.vectorize
        def b_int_sig(_x):
            def to_int_func(_xx):
                return signal_func_norm * signal_func(_xx, *signal_func_param)
            
            if _x not in used_sig_int.keys():
                area = sci.quad(to_int_func, _x - self.bin_width / 2., _x + self.bin_width / 2.,
                                epsrel=1e-16)[0]
                used_sig_int[_x] = area
                return area * n_mc_s * self.bin_width * 0.5
            
            if _x in used_sig_int.keys():
                area = used_sig_int[_x]
                return area * n_mc_s * self.bin_width * 0.5
        
        @np.vectorize
        def used_func(_mu, _x):
            return _mu * b_int_sig(_x) + b_int_bac(_x)
        
        mu_array = np.linspace(0.0, 3.5, mu_len)
        
        p = np.meshgrid(mu_array, used_x_point)
        # noinspection PyTypeChecker
        ll_array = np.sum(
            np.multiply(used_d_point * self.bin_width, np.log(used_func(p[0], p[1]).T)) - used_func(p[0], p[1]).T,
            axis=1)
        
        ll_array = - 2 * (ll_array - np.amax(ll_array))
        plt.plot(mu_array, ll_array)
        plt.title(self.__get_inters_points(mu_array, ll_array, 1))
        plt.show()
    
    def calc_signal_strength_unbinned_point_wise(self):
        """
        Function to calculate the signal strength mu_H to check if the corresponding signal MC is correct.
        unbinned variant

        :return: fill calculated_dump dict with calculated results
        """
        
        mu_len = 73
        signal_func = self.signal_func
        background_func = self.background_func
        
        n_mc_b = np.sum(self.data_com.data["mc_bac"])
        n_mc_s = np.sum(self.data_com.data["mc_sig"])
        
        used_x_point = self.data_raw.data
        
        background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                             args=tuple(self.func_func_param[background_func.__name__]),
                                             epsrel=1e-12)[0]
        background_func_param = self.func_func_param[background_func.__name__]
        
        signal_func_param = self.func_func_param[signal_func.__name__]
        signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                         args=tuple(signal_func_param), epsrel=1e-12)[0]
        
        used_norm = {}
        used_expectation_val = {}
        
        @np.vectorize
        def used_func(_mu, _x):
            
            def to_int_func(_xx):
                sig_part = _mu * signal_func_norm * signal_func(_xx, *signal_func_param) * n_mc_s
                bac_part = background_func_norm * background_func(_xx, *background_func_param) * n_mc_b
                return sig_part + bac_part
            
            used_func_norm = 1.0
            if _mu not in used_norm.keys():
                used_func_norm = 1. / sci.quad(to_int_func, self.hist_range[0], self.hist_range[1], epsrel=1e-12)[0]
                used_norm[_mu] = used_func_norm
            if _mu in used_norm.keys():
                used_func_norm = used_norm[_mu]
            
            return used_func_norm * to_int_func(_x)
        
        @np.vectorize
        def expectation_value(_mu):
            def to_int_func(_xx):
                return _xx * used_func(_mu, _xx)
            
            if _mu not in used_expectation_val.keys():
                pass_exp_val = sci.quad(to_int_func, self.hist_range[0], self.hist_range[1],
                                        epsrel=1e-16)[0]
                used_expectation_val[_mu] = pass_exp_val
                return pass_exp_val
            
            if _mu in used_expectation_val.keys():
                pass_exp_val = used_expectation_val[_mu]
                return pass_exp_val
        
        mu_array = np.linspace(0.0, 2.5, mu_len)
        p = np.meshgrid(mu_array, used_x_point)
        
        exp_val = expectation_value(mu_array)
        ll_array = - exp_val + np.sum(np.log(np.multiply(exp_val, used_func(p[0], p[1]))), axis=0)
        
        ll_array = - 2 * (ll_array - np.amax(ll_array))
        plt.plot(mu_array, ll_array)
        plt.title(self.__get_inters_points(mu_array, ll_array, 1))
        plt.show()
    
    # calc signal strength with mass variation
    # ------------------------------------------------
    def calc_signal_strength_mass_unbinned_point_wise(self, scan_points=300):
        """
        Function to calculate the signal strength mu_H to check if the corresponding signal MC is correct.
        In addition: the mass is also a variable that will be optimized.
        unbinned variant

        :return: fill calculated_dump dict with calculated results
        """
        
        if "calc_signal_strength_mass_unbinned_point_wise" not in self.calculated_dump.keys():
            self.calculated_dump["calc_signal_strength_mass_unbinned_point_wise"] = {}
        
        signal_func = self.signal_func
        background_func = self.background_func
        used_mass_array = self.__create_3_interval_array(start=111.0, end=141.0, point_1=124.5, point_2=126.5,
                                                         num_i1=int(scan_points / 4 + 1),
                                                         num_i2=int(scan_points / 2 + 1),
                                                         num_i3=int(scan_points / 4 + 1))
        
        n_mc_b = np.sum(self.data_com.data["mc_bac"])
        n_mc_s = np.sum(self.data_com.data["mc_sig"])
        used_x_point = self.data_raw.data
        
        def get_ln_mu_min(mass_array):
            ln_mu_min = np.array([])
            
            background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(self.func_func_param[background_func.__name__]),
                                                 epsrel=1e-12)[0]
            background_func_param = self.func_func_param[background_func.__name__]
            used_b_point_bac = {}
            
            for i in range(len(mass_array)):
                mu_temp = self.__create_3_interval_array(start=0.0, end=5.0, point_1=5. / 3, point_2=2 * 5. / 3,
                                                         num_i1=int(scan_points / 3.), num_i2=int(scan_points / 3.),
                                                         num_i3=int(scan_points / 3.))
                
                if self.verbose:
                    print("min_mas_scan: {} / {}".format(i, len(mass_array)), end="\r", flush=True)
                
                signal_func_param = self.func_func_param[signal_func.__name__]
                signal_func_param[1] = mass_array[i]
                signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(signal_func_param), epsrel=1e-12)[0]
                
                used_norm = {}
                used_exp_val = {}
                used_b_point_sig = {}
                
                @np.vectorize
                def b_point_bac(x_point):
                    if x_point not in used_b_point_bac.keys():
                        pass_val = background_func_norm * background_func(x_point, *background_func_param)
                        used_b_point_bac[x_point] = pass_val
                        return pass_val * n_mc_b
                    if x_point in used_b_point_bac.keys():
                        pass_val = used_b_point_bac[x_point]
                        
                        return pass_val * n_mc_b
                
                @np.vectorize
                def b_point_sig(x_point):
                    if x_point not in used_b_point_sig.keys():
                        pass_val = signal_func_norm * signal_func(x_point, *signal_func_param)
                        used_b_point_sig[x_point] = pass_val
                        return pass_val * n_mc_s
                    if x_point in used_b_point_sig.keys():
                        pass_val = used_b_point_sig[x_point]
                        return pass_val * n_mc_s
                
                @np.vectorize
                def used_func(_mu, _x):
                    def to_int_func(_xx):
                        return _mu * b_point_sig(_xx) + b_point_bac(_xx)
                    
                    if _mu not in used_norm.keys():
                        norm = 1. / sci.quad(to_int_func, self.hist_range[0], self.hist_range[1], epsrel=1e-16)[0]
                        used_norm[_mu] = norm
                        return norm * to_int_func(_x)
                    if _mu in used_norm.keys():
                        norm = used_norm[_mu]
                        return norm * to_int_func(_x)
                
                @np.vectorize
                def expectation_value(_mu):
                    def to_int_func(_xx):
                        return _xx * used_func(_mu, _xx)
                    
                    if _mu not in used_exp_val.keys():
                        pass_exp_val = sci.quad(to_int_func, self.hist_range[0], self.hist_range[1], epsrel=1e-16)[0]
                        used_exp_val[_mu] = pass_exp_val
                        return pass_exp_val
                    if _mu in used_exp_val.keys():
                        pass_exp_val = used_exp_val[_mu]
                        return pass_exp_val
                
                p = np.meshgrid(mu_temp, used_x_point)
                exp_val = expectation_value(mu_temp)
                ln = - exp_val + np.sum(np.log(np.multiply(exp_val, used_func(p[0], p[1]))), axis=0)
                drawn_ln = -2 * ln
                ln_mu_min = np.append(ln_mu_min, np.amin(drawn_ln))
            
            return ln_mu_min
        
        def get_ln_mu_min_on_fixed_mass(mu_array, min_mass):
            mu_temp = mu_array
            background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(self.func_func_param[background_func.__name__]),
                                                 epsrel=1e-12)[0]
            background_func_param = self.func_func_param[background_func.__name__]
            
            signal_func_param = self.func_func_param[signal_func.__name__]
            signal_func_param[1] = min_mass
            signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                             args=tuple(signal_func_param), epsrel=1e-12)[0]
            
            used_norm = {}
            used_exp_val = {}
            used_b_point_bac = {}
            used_b_point_sig = {}
            
            @np.vectorize
            def b_point_bac(x_point):
                if x_point not in used_b_point_bac.keys():
                    pass_val = background_func_norm * background_func(x_point, *background_func_param)
                    used_b_point_bac[x_point] = pass_val
                    return pass_val * n_mc_b
                if x_point in used_b_point_bac.keys():
                    pass_val = used_b_point_bac[x_point]
                    
                    return pass_val * n_mc_b
            
            @np.vectorize
            def b_point_sig(x_point):
                if x_point not in used_b_point_sig.keys():
                    pass_val = signal_func_norm * signal_func(x_point, *signal_func_param)
                    used_b_point_sig[x_point] = pass_val
                    return pass_val * n_mc_s
                if x_point in used_b_point_sig.keys():
                    pass_val = used_b_point_sig[x_point]
                    return pass_val * n_mc_s
            
            @np.vectorize
            def used_func(_mu, _x):
                def to_int_func(_xx):
                    return _mu * b_point_sig(_xx) + b_point_bac(_xx)
                
                if _mu not in used_norm.keys():
                    norm = 1. / sci.quad(to_int_func, self.hist_range[0], self.hist_range[1], epsrel=1e-16)[0]
                    used_norm[_mu] = norm
                    return norm * to_int_func(_x)
                if _mu in used_norm.keys():
                    norm = used_norm[_mu]
                    return norm * to_int_func(_x)
            
            @np.vectorize
            def expectation_value(_mu):
                def to_int_func(_xx):
                    return _xx * used_func(_mu, _xx)
                
                if _mu not in used_exp_val.keys():
                    pass_exp_val = sci.quad(to_int_func, self.hist_range[0], self.hist_range[1], epsrel=1e-16)[0]
                    used_exp_val[_mu] = pass_exp_val
                    return pass_exp_val
                if _mu in used_exp_val.keys():
                    pass_exp_val = used_exp_val[_mu]
                    return pass_exp_val
            
            p = np.meshgrid(mu_temp, used_x_point)
            exp_val = expectation_value(mu_temp)
            ln = - exp_val + np.sum(np.log(np.multiply(exp_val, used_func(p[0], p[1]))), axis=0)
            drawn_ln = -2 * ln
            
            return drawn_ln
        
        p_ln_mu_min = get_ln_mu_min(used_mass_array)
        p_ln_mu_min = p_ln_mu_min - np.amin(p_ln_mu_min)
        
        mu_arr = self.__create_3_interval_array(start=0.0, end=5.0, point_1=5. / 3, point_2=2 * 5. / 3,
                                                num_i1=int(scan_points / 3.), num_i2=int(scan_points / 3.),
                                                num_i3=int(scan_points / 3.))
        p_ln = get_ln_mu_min_on_fixed_mass(mu_arr, used_mass_array[np.argmin(p_ln_mu_min)])
        p_ln_pass_min = np.amin(p_ln)
        p_ln = p_ln - np.amin(p_ln)
        
        self.calculated_dump["calc_signal_strength_mass_unbinned_point_wise"] = {
            "mu_array": mu_arr, "mass_array": used_mass_array,
            "mu_nll_array": p_ln, "mass_nll_array": p_ln_mu_min,
            "mu_sigma_min_sigma": self.__get_inters_points(mu_arr, p_ln, 1),
            "mass_sigma_min_sigma": self.__get_inters_points(used_mass_array, p_ln_mu_min, 1),
            "contour_min": p_ln_pass_min
        }
    
    # calc alpha with mass variation
    # ------------------------------------------------
    def calc_alpha_mass_binned_integrated(self, scan_points=300):
        """
        Funtion to find the optimum signal strength alpha and mass.
        binned variant
        #TODO: check if calculated values are consistent - looks consistent; will not be used in thesis

        :param scan_points: number of points that the arrays will have during the funtion
        :return: fill calculated_dump dict with calculated results
        """
        
        print("calc_alpha_mass_binned_integrated")
        signal_func = self.signal_func
        background_func = self.background_func
        
        if "calc_alpha_mass_binned_integrated" not in list(self.calculated_dump.keys()):
            self.calculated_dump["calc_alpha_mass_binned_integrated"] = {}
        
        used_mass_array = self.__create_3_interval_array(start=111.0, end=141.0, point_1=124.5, point_2=126.5,
                                                         num_i1=int(scan_points / 4 + 1),
                                                         num_i2=int(scan_points / 2 + 1),
                                                         num_i3=int(scan_points / 4 + 1))
        
        used_x_point = np.array(self.data_com.x_range)
        used_d_point = self.data_com.data["data"]
        n_mc_b = np.sum(self.data_com.data["mc_bac"])
        n_mc_s = np.sum(self.data_com.data["mc_sig"])
        
        def get_ln_alpha_min(mass_array):
            
            ln_alpha_min = np.array([])
            
            background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(self.func_func_param[background_func.__name__]))[0]
            sig_param = self.func_func_param[signal_func.__name__]
            
            for i in range(len(mass_array)):
                alpha_temp = self.__create_3_interval_array(start=0, end=1, point_1=0.075, point_2=0.65,
                                                            num_i1=int(scan_points * (1. / 4)),
                                                            num_i2=int(scan_points * (2. / 4)),
                                                            num_i3=int(scan_points * (1. / 4)))
                if self.verbose:
                    print("min_mas_scan: {} / {}".format(i, len(mass_array)), end="\r", flush=True)
                
                sig_param[1] = mass_array[i]
                signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(sig_param))[0]
                
                used_b_int_bac = {}
                used_b_int_sig = {}
                
                @np.vectorize
                def b_int_bac(x_point):
                    bac_func_args = self.func_func_param[background_func.__name__]
                    area = 0.0
                    if x_point not in used_b_int_bac.keys():
                        area = background_func_norm * sci.quad(background_func,
                                                               x_point - self.bin_width / 2.,
                                                               x_point + self.bin_width / 2.,
                                                               args=tuple(bac_func_args))[0] * n_mc_b * self.bin_width
                        used_b_int_bac[x_point] = area
                    
                    if x_point in used_b_int_bac.keys():
                        area = used_b_int_bac[x_point]
                    
                    return area
                
                @np.vectorize
                def b_int_sig(x_point):
                    sig_func_args = self.func_func_param[signal_func.__name__]
                    area = 0.0
                    if x_point not in used_b_int_sig.keys():
                        area = signal_func_norm * sci.quad(signal_func,
                                                           x_point - self.bin_width / 2., x_point + self.bin_width / 2.,
                                                           args=tuple(sig_func_args))[0] * n_mc_s * self.bin_width
                        used_b_int_sig[x_point] = area
                    
                    if x_point in used_b_int_sig.keys():
                        area = used_b_int_sig[x_point]
                    
                    return area
                
                @np.vectorize
                def used_func(alpha_point, x_point):
                    return alpha_point * b_int_sig(x_point) + (1 - alpha_point) * b_int_bac(x_point)
                
                p = np.meshgrid(alpha_temp, used_x_point)
                # noinspection PyTypeChecker
                ln = np.sum(
                    np.multiply(used_d_point * self.bin_width, np.log(used_func(p[0], p[1])).T) - used_func(p[0],
                                                                                                            p[1]).T,
                    axis=1)
                
                draw_ln = - 2 * ln
                ln_alpha_min = np.append(ln_alpha_min, np.amin(draw_ln))
            
            return ln_alpha_min
        
        def get_ln_alpha_min_on_fixed_mass(alpha_array, min_mass):
            
            alpha_temp = alpha_array
            
            background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(self.func_func_param[background_func.__name__]))[0]
            
            sig_param = self.func_func_param[signal_func.__name__]
            sig_param[1] = min_mass
            signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                             args=tuple(sig_param))[0]
            
            used_b_int_bac = {}
            used_b_int_sig = {}
            
            @np.vectorize
            def b_int_bac(x_point):
                bac_func_args = self.func_func_param[background_func.__name__]
                area = 0.0
                if x_point not in used_b_int_bac.keys():
                    area = background_func_norm * sci.quad(background_func,
                                                           x_point - self.bin_width / 2.,
                                                           x_point + self.bin_width / 2.,
                                                           args=tuple(bac_func_args))[0] * n_mc_b * self.bin_width
                    used_b_int_bac[x_point] = area
                
                if x_point in used_b_int_bac.keys():
                    area = used_b_int_bac[x_point]
                
                return area
            
            @np.vectorize
            def b_int_sig(x_point):
                sig_func_args = self.func_func_param[signal_func.__name__]
                sig_func_args[1] = min_mass
                area = 0.0
                if x_point not in used_b_int_sig.keys():
                    area = signal_func_norm * sci.quad(signal_func,
                                                       x_point - self.bin_width / 2., x_point + self.bin_width / 2.,
                                                       args=tuple(sig_func_args))[0] * n_mc_s * self.bin_width
                    used_b_int_sig[x_point] = area
                
                if x_point in used_b_int_sig.keys():
                    area = used_b_int_sig[x_point]
                
                return area
            
            @np.vectorize
            def used_func(alpha_point, x_point):
                return alpha_point * b_int_sig(x_point) + (1 - alpha_point) * b_int_bac(x_point)
            
            p = np.meshgrid(alpha_temp, used_x_point)
            # noinspection PyTypeChecker
            ln = np.sum(
                np.multiply(used_d_point * self.bin_width, np.log(used_func(p[0], p[1])).T) - used_func(p[0], p[1]).T,
                axis=1)
            
            draw_ln = - 2 * ln
            
            return draw_ln
        
        p_ln_alpha_min = get_ln_alpha_min(used_mass_array)
        p_ln_alpha_min = p_ln_alpha_min - np.amin(p_ln_alpha_min)
        
        alpha_p = self.__create_3_interval_array(start=0, end=1, point_1=0.075, point_2=0.65,
                                                 num_i1=int(scan_points * (10. / 4)),
                                                 num_i2=int(scan_points * (20. / 4)),
                                                 num_i3=int(scan_points * (10. / 4)))
        p_ln = get_ln_alpha_min_on_fixed_mass(alpha_p, used_mass_array[np.argmin(p_ln_alpha_min)])
        p_ln_pass_min = np.amin(p_ln)
        p_ln = p_ln - np.amin(p_ln)
        
        self.calculated_dump["calc_alpha_mass_binned_integrated"] = {
            "alpha_array": alpha_p, "mass_array": used_mass_array,
            "alpha_nll_array": p_ln, "mass_nll_array": p_ln_alpha_min,
            "alpha_sigma_min_sigma": self.__get_inters_points(alpha_p, p_ln, 1),
            "mass_sigma_min_sigma": self.__get_inters_points(used_mass_array, p_ln_alpha_min, 1),
            "contour_min": p_ln_pass_min
        }
    
    def calc_alpha_mass_unbinned_point_wise(self, scan_points=300, significance_only=False):
        """
        Function to find the optimum signal strength alpha and mass.
        unbinned variant

        :param scan_points: number of points that the arrays will have during the function
        :param significance_only: for error estimation only
        :return: fill calculated_dump dict with calculated results
        """
        
        print("calc_alpha_mass_unbinned_point_wise")
        signal_func = self.signal_func
        background_func = self.background_func
        
        if "calc_alpha_mass_unbinned_point_wise" not in list(self.calculated_dump.keys()):
            self.calculated_dump["calc_alpha_mass_unbinned_point_wise"] = {}
        
        scan_n_points = scan_points
        used_mass_array = self.__create_3_interval_array(start=111.0, end=141.0, point_1=124.5, point_2=126.5,
                                                         num_i1=int(scan_points / 4 + 1),
                                                         num_i2=int(scan_points / 2 + 1),
                                                         num_i3=int(scan_points / 4 + 1))
        
        if significance_only:
            used_mass_array = np.linspace(123.5, 126.5, int(scan_points))
        
        def get_ln_alpha_min(mass_array):
            
            ln_alpha_min = np.array([])
            
            used_x_point = self.data_raw.data
            
            for i in range(len(mass_array)):
                alpha_temp = self.__create_3_interval_array(start=0, end=1, point_1=0.075, point_2=0.65,
                                                            num_i1=int(scan_points * (1. / 4)),
                                                            num_i2=int(scan_points * (2. / 4)),
                                                            num_i3=int(scan_points * (1. / 4)))
                
                if significance_only:
                    alpha_temp = self.__create_3_interval_array(start=0, end=1, point_1=0.075, point_2=0.65,
                                                                num_i1=int(scan_points * (1. / 24)),
                                                                num_i2=int(scan_points * (22. / 24)),
                                                                num_i3=int(scan_points * (1. / 24)))
                
                if self.verbose:
                    print("min_mas_scan: {} / {}".format(i, len(mass_array)), end="\r", flush=True)
                background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                                     args=tuple(self.func_func_param[background_func.__name__]))[0]
                
                sig_param = self.func_func_param[signal_func.__name__]
                sig_param[1] = mass_array[i]
                signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(sig_param))[0]
                bac_func_args = self.func_func_param[background_func.__name__]
                
                used_b_point_bac = {}
                used_b_point_sig = {}
                
                @np.vectorize
                def b_point_bac(x_point):
                    
                    if x_point not in used_b_point_bac.keys():
                        pass_val = background_func_norm * background_func(x_point, *bac_func_args)
                        used_b_point_bac[x_point] = pass_val
                        return pass_val
                    if x_point in used_b_point_bac.keys():
                        pass_val = used_b_point_bac[x_point]
                        
                        return pass_val
                
                @np.vectorize
                def b_point_sig(x_point):
                    
                    if x_point not in used_b_point_sig.keys():
                        pass_val = signal_func_norm * signal_func(x_point, *sig_param)
                        used_b_point_sig[x_point] = pass_val
                        return pass_val
                    if x_point in used_b_point_sig.keys():
                        pass_val = used_b_point_sig[x_point]
                        
                        return pass_val
                
                @np.vectorize
                def used_func(alpha_point, x_point):
                    return alpha_point * b_point_sig(x_point) + (1 - alpha_point) * b_point_bac(x_point)
                
                used_exp_val = {}
                
                @np.vectorize
                def expectation_value(alpha_point):
                    def to_int_func(x_point):
                        return x_point * (alpha_point * b_point_sig(x_point) + (1 - alpha_point) * b_point_bac(x_point))
                    
                    pass_exp_val = np.sum(self.data_com.data["data"])
                    if alpha_point not in used_exp_val.keys():
                        pass_exp_val = sci.quad(to_int_func, self.hist_range[0], self.hist_range[1],
                                                points=np.linspace(self.hist_range[0], self.hist_range[1], 42),
                                                epsrel=1e-16)[0]
                        used_exp_val[alpha_point] = pass_exp_val
                    
                    if alpha_point in used_exp_val.keys():
                        pass_exp_val = used_exp_val[alpha_point]
                    
                    return pass_exp_val
                
                p = np.meshgrid(alpha_temp, used_x_point)
                exp_val = expectation_value(alpha_temp)
                ln = - exp_val + np.sum(np.log(np.multiply(exp_val, used_func(p[0], p[1]))), axis=0)
                
                draw_ln = - 2 * ln
                ln_alpha_min = np.append(ln_alpha_min, np.amin(draw_ln))
            
            return ln_alpha_min
        
        def get_ln_alpha_min_on_fixed_mass(alpha_array, min_mass):
            
            used_x_point = self.data_raw.data
            
            alpha_temp = alpha_array
            
            background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(self.func_func_param[background_func.__name__]))[0]
            
            sig_param = self.func_func_param[signal_func.__name__]
            sig_param[1] = min_mass
            signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                             args=tuple(sig_param))[0]
            
            @np.vectorize
            def b_point_bac(x_point):
                bac_func_args = self.func_func_param[background_func.__name__]
                return background_func_norm * background_func(x_point, *bac_func_args)
            
            @np.vectorize
            def b_point_sig(x_point):
                return signal_func_norm * signal_func(x_point, *sig_param)
            
            @np.vectorize
            def used_func(alpha_point, x_point):
                return alpha_point * b_point_sig(x_point) + (1 - alpha_point) * b_point_bac(x_point)
            
            @np.vectorize
            def expectation_value(alpha_point):
                def to_int_func(x_point):
                    return x_point * (alpha_point * b_point_sig(x_point) + (1 - alpha_point) * b_point_bac(x_point))
                
                return sci.quad(to_int_func, self.hist_range[0], self.hist_range[1],
                                points=np.linspace(self.hist_range[0], self.hist_range[1], 42), epsrel=1e-10)[0]
            
            p = np.meshgrid(alpha_temp, used_x_point)
            exp_val = expectation_value(alpha_temp)
            ln = - exp_val + np.sum(np.log(np.multiply(exp_val, used_func(p[0], p[1]))), axis=0)
            
            draw_ln = - 2 * ln
            
            return draw_ln
        
        p_ln_alpha_min = get_ln_alpha_min(used_mass_array)
        p_ln_alpha_min = p_ln_alpha_min - np.amin(p_ln_alpha_min)
        
        alpha = self.__create_3_interval_array(start=0, end=1, point_1=0.075, point_2=0.65,
                                               num_i1=int(scan_points * (1. / 4)),
                                               num_i2=int(scan_points * (2. / 4)),
                                               num_i3=int(scan_points * (1. / 4)))
        if significance_only:
            alpha = self.__create_3_interval_array(start=0, end=1, point_1=0.075, point_2=0.65,
                                                   num_i1=int(scan_points * (1. / 24)),
                                                   num_i2=int(scan_points * (22. / 24)),
                                                   num_i3=int(scan_points * (1. / 24)))
        
        p_ln = get_ln_alpha_min_on_fixed_mass(alpha, used_mass_array[np.argmin(p_ln_alpha_min)])
        p_ln_pass_min = np.amin(p_ln)
        p_ln = p_ln - np.amin(p_ln)
        
        if significance_only:
            return np.sqrt(p_ln[0])
        
        self.calculated_dump["calc_alpha_mass_unbinned_point_wise"] = {
            "alpha_array": alpha, "mass_array": used_mass_array,
            "alpha_nll_array": p_ln, "mass_nll_array": p_ln_alpha_min,
            "alpha_sigma_min_sigma": self.__get_inters_points(alpha, p_ln, 1),
            "mass_sigma_min_sigma": self.__get_inters_points(used_mass_array, p_ln_alpha_min, 1),
            "contour_min": p_ln_pass_min
        }
    
    # calc q0 and p0 in 1d (mass scan)
    # ------------------------------------------------
    def calc_q0_binned_integrated(self, to_file=False):
        """
        Funtion to calculate the ln L-ratio and the corresponding p0 - value.
        binned variant-

        :param to_file: saves the calculation to a file
        :return: fill calculated_dump dict with calculated results
        """
        
        print("calc_q0_binned_integrated")
        mu_len = 101
        signal_func = self.signal_func
        background_func = self.background_func
        
        if self.signal_func.__name__ != "gauss":
            sys.exit("Test it with gauss!")
        
        n_mc_b = np.sum(self.data_com.data["mc_bac"])
        n_mc_s = np.sum(self.data_com.data["mc_sig"])
        used_x_point = self.data_com.x_range
        used_d_point = self.data_com.data["data"]
        
        background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                             args=tuple(self.func_func_param[background_func.__name__]),
                                             epsrel=1e-12)[0]
        background_func_param = self.func_func_param[background_func.__name__]
        
        used_int_bac_points = {}
        
        @np.vectorize
        def b_int_bac(x_point):
            def normed_bac_func(_x):
                return background_func(_x, *background_func_param) * background_func_norm  # * n_mc_b * self.bin_width
            
            if x_point not in used_int_bac_points.keys():
                area = sci.quad(normed_bac_func, x_point - self.bin_width / 2., x_point + self.bin_width / 2.,
                                epsrel=1e-12)[0]
                used_int_bac_points[x_point] = area
                return area
            if x_point in used_int_bac_points.keys():
                area = used_int_bac_points[x_point]
                return area
        
        def calc_h0_hypothesis_as_ll():
            fac = (n_mc_s + n_mc_b) * self.bin_width
            ll = np.sum(
                np.multiply(used_d_point * self.bin_width, np.log(fac * b_int_bac(used_x_point))) - fac * b_int_bac(
                    used_x_point))
            return np.double(ll)
        
        h0_hypothesis = calc_h0_hypothesis_as_ll()
        
        mass_scan_array = self.__create_mh_scan_array_for_p0()
        sigma_scan_array = self.__create_gauss_sigma(mass_scan_array)
        
        q0_array = np.array([], dtype=np.longdouble)
        p0_array = np.array([], dtype=np.longdouble)
        
        for _i in range(len(mass_scan_array)):
            
            signal_func_param = [sigma_scan_array[_i], mass_scan_array[_i]]
            signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                             args=tuple(signal_func_param), epsrel=1e-16)[0]
            
            def calc_h1_hypothesis_as_ll():
                
                used_int_sig_points = {}
                
                @np.vectorize
                def b_int_sig(x_point):
                    def normed_sig_func(_x):
                        return signal_func(_x, *signal_func_param) * signal_func_norm  # * n_mc_s * self.bin_width
                    
                    if x_point not in used_int_sig_points.keys():
                        area = sci.quad(normed_sig_func, x_point - self.bin_width / 2., x_point + self.bin_width / 2.,
                                        epsrel=1e-16)[0]
                        
                        used_int_sig_points[x_point] = area
                        return area
                    if x_point in used_int_sig_points.keys():
                        area = used_int_sig_points[x_point]
                        return area
                
                @np.vectorize
                def used_func(_mu, _x):
                    return (_mu * b_int_sig(_x) + b_int_bac(_x))
                
                mu_array = np.linspace(0, 1, mu_len)
                fac = (n_mc_s + n_mc_b) * self.bin_width
                p = np.meshgrid(mu_array, used_x_point)
                # noinspection PyTypeChecker
                ll_array = np.sum(
                    np.multiply(used_d_point * self.bin_width, np.log(fac * used_func(p[0], p[1]).T)) - fac * used_func(p[0],
                                                                                                                        p[1]).T,
                    axis=1)
                
                min_mu = mu_array[np.argmax(ll_array)]
                if min_mu <= 0.00:
                    return np.double(h0_hypothesis), 0.0
                
                ll = np.sum(
                    used_d_point * self.bin_width * np.log(fac * used_func(min_mu, used_x_point)) - fac * used_func(
                        min_mu,
                        used_x_point))
                return np.double(ll), min_mu
            
            h1_hypothesis, best_mu = calc_h1_hypothesis_as_ll()
            
            q0_array = np.append(q0_array, np.double(- 2 * (h0_hypothesis - h1_hypothesis)))
            p0_array = np.append(p0_array, 0.5 * math.erfc(np.sqrt(np.double(- (h0_hypothesis - h1_hypothesis)))))
            
            if self.verbose:
                print("mh: {}; q0: {}; p0: {}; mu: {}; Z: {};  {} / {}".format(round(mass_scan_array[_i], 5),
                                                                               round(q0_array[_i], 5),
                                                                               round(p0_array[_i], 5),
                                                                               round(best_mu, 5),
                                                                               round(np.sqrt(abs(q0_array[_i])), 5),
                                                                               _i, len(mass_scan_array)),
                      end="\r", flush=True)
        
        c = np.vstack((mass_scan_array, q0_array, p0_array))
        if to_file:
            np.savetxt(os.path.join(self.save_dir,
                                    "q0_p0_over_mass_scan_binned_integrated_{}.csv".format(round(self.bin_width, 2))),
                       c.T,
                       delimiter=",")
        if not to_file:
            self.calculated_dump["q0_p0_over_mass_scan_binned_integrated_{}.csv".format(round(self.bin_width, 2))] = {
                "mass": mass_scan_array, "q0": q0_array, "p0": p0_array
            }
    
    def calc_q0_unbinned_point_wise(self, to_file=False):
        """
        Funtion to calculate the ln L-ratio and the corresponding p0 - value.
        unbinned variant-

        :param to_file: saves the calculation to a file
        :return: fill calculated_dump dict with calculated results
        """
        
        print("calc_q0_unbinned_point_wise")
        
        if "calc_q0_unbinned_point_wise" not in self.calculated_dump.keys():
            self.calculated_dump["calc_q0_unbinned_point_wise"] = {}
        
        mu_len = 50
        
        signal_func = self.signal_func
        background_func = self.background_func
        
        if self.signal_func.__name__ != "gauss":
            sys.exit("Test it with gauss!")
        
        used_x_point = self.data_raw.data
        n_mc_b = np.sum(self.data_com.data["mc_bac"])
        n_mc_s = np.sum(self.data_com.data["mc_sig"])
        
        background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                             args=tuple(self.func_func_param[background_func.__name__]),
                                             epsrel=1e-16)[0]
        background_func_param = self.func_func_param[background_func.__name__]
        
        def calc_h0_hypothesis_as_ll():
            
            def used_func(x):
                return background_func_norm * background_func(x, *background_func_param)
            
            def expectation_value():
                def to_int_func(_xx):
                    return _xx * used_func(_xx)
                
                return sci.quad(to_int_func, self.hist_range[0], self.hist_range[1],
                                points=np.linspace(self.hist_range[0], self.hist_range[1], 42), epsrel=1e-16)[0]
            
            exp_val = expectation_value()
            ll = - exp_val + np.sum(np.log(exp_val * used_func(used_x_point)))
            
            return ll
        
        h0_hypothesis = calc_h0_hypothesis_as_ll()
        
        mass_scan_array = self.__create_mh_scan_array_for_p0()
        sigma_scan_array = self.__create_gauss_sigma(mass_scan_array)
        
        q0_array = np.zeros(len(mass_scan_array), dtype=np.longdouble)
        p0_array = np.zeros(len(mass_scan_array), dtype=np.longdouble)
        
        for ii in range(len(mass_scan_array)):
            
            def calc_h1_hypothesis_as_ll():
                signal_func_param = [sigma_scan_array[ii], mass_scan_array[ii]]
                signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(signal_func_param), epsrel=1e-16)[0]
                
                used_norm = {}
                
                @np.vectorize
                def used_func(_mu, _x):
                    def to_int_func(_xx):
                        sig_part = _mu * signal_func_norm * signal_func(_xx, *signal_func_param) * n_mc_s
                        bac_part = background_func_norm * background_func(_xx, *background_func_param) * n_mc_b
                        return sig_part + bac_part
                    
                    if _mu not in used_norm.keys():
                        used_func_norm = 1. / sci.quad(to_int_func, self.hist_range[0], self.hist_range[1],
                                                       points=np.linspace(self.hist_range[0], self.hist_range[1], 42),
                                                       epsrel=1e-16)[0]
                        used_norm[_mu] = used_func_norm
                        
                        return used_func_norm * to_int_func(_x)
                    
                    if _mu in used_norm.keys():
                        used_func_norm = used_norm[_mu]
                        
                        return used_func_norm * to_int_func(_x)
                
                @np.vectorize
                def expectation_value(_mu):
                    def to_int_func(_xx):
                        return _xx * used_func(_mu, _xx)
                    
                    return sci.quad(to_int_func, self.hist_range[0], self.hist_range[1],
                                    points=np.linspace(self.hist_range[0], self.hist_range[1], 42), epsrel=1e-16)[0]
                
                mu_array = np.linspace(0, 1.75, mu_len)
                p = np.meshgrid(mu_array, used_x_point)
                
                exp_val = expectation_value(mu_array)
                ll_array = - exp_val + np.sum(np.log(np.multiply(exp_val, used_func(p[0], p[1]))), axis=0)
                
                min_mu = mu_array[np.argmax(ll_array)]
                if min_mu <= 0.0:
                    return h0_hypothesis
                
                exp_val_min_mu = expectation_value(min_mu)
                ll = - exp_val_min_mu + np.sum(np.log(exp_val_min_mu * used_func(min_mu, used_x_point)))
                
                del used_norm
                return ll
            
            h1_hypothesis = calc_h1_hypothesis_as_ll()
            
            q0_array[ii] = - 2 * (h0_hypothesis - h1_hypothesis)
            p0_array[ii] = 0.5 * math.erfc(np.sqrt(- (h0_hypothesis - h1_hypothesis)))
            
            if self.verbose:
                print("mh: {}; q0: {};  {} / {}".format(mass_scan_array[ii], q0_array[ii], ii, len(mass_scan_array)),
                      end="\r", flush=True)
        
        c = np.vstack((mass_scan_array, q0_array, p0_array))
        if to_file:
            np.savetxt(os.path.join(self.save_dir, "q0_p0_over_mass_scan_unbinned.csv"), c.T, delimiter=",")
        self.calculated_dump["calc_q0_unbinned_point_wise"] = {
            "mass": mass_scan_array, "q0": q0_array, "p0": p0_array
        }
    
    # 1d plots
    # ------------------------------------------------
    def plot_alpha_only(self, tag="unbinned", to_file=False, title_=False):
        """
        Function to plot the calculated signal strength only calculation

        :param tag: "binned" or "unbinned"
        :param to_file: saves to File if True
        """
        
        print("plot_alpha_only, tag: {}, to_file: {}".format(tag, to_file))
        
        used_part = {}
        if tag == "binned":
            try:
                used_part = self.calculated_dump["calc_alpha_binned_integrated"]
            except KeyError:
                self.calc_alpha_binned_integrated()
                used_part = self.calculated_dump["calc_alpha_binned_integrated"]
        
        if tag == "unbinned":
            try:
                used_part = self.calculated_dump["calc_alpha_unbinned_point_wise"]
            except KeyError:
                self.calc_alpha_unbinned_point_wise()
                used_part = self.calculated_dump["calc_alpha_unbinned_point_wise"]
        
        alpha = used_part["alpha_array"]
        nll = used_part["nll_array"]
        
        fig, ax = plt.subplots()
        ax.plot(alpha, nll, label=r"$-2\ln(\mathcal{L})-\rm{profile}$")
        ax.set_xlabel(r"$\alpha_{\rm{s}}$")
        ax.set_ylabel(r"$- 2 \ln \left( \frac{\mathcal{L}}{\mathcal{L}_{\rm{min}}} \right)$")
        ax.legend(loc="upper left")
        
        keys_func_names = list(self.func_names.keys())
        if tag == "unbinned" and title_:
            ax.set_title(
                "{}: {} + {}".format(tag, self.func_names[keys_func_names[0]], self.func_names[keys_func_names[1]]))
        if tag == "binned" and title_:
            ax.set_title("{} ({} GeV per bin): {} + {}".format(tag, np.round(self.bin_width, 2),
                                                               self.func_names[keys_func_names[0]],
                                                               self.func_names[keys_func_names[1]]))
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 40)
        
        observed_z = str(round(used_part["significance"], 3))
        ax.annotate("{}".format(observed_z) + r"$\sigma$",
                    xy=(0.0, used_part["nll_array"][0]), xytext=(0.14, 14),
                    arrowprops=dict(width=1.0, color="black"))
        ax.hlines([4, 9, 16, 25], 0, 0.05, color="red")
        ax.text(0.0525, 3.7, r"$2 \sigma$", color="red")
        ax.text(0.0525, 8.7, r"$3 \sigma$", color="red")
        ax.text(0.0525, 15.7, r"$4 \sigma$", color="red")
        ax.text(0.0525, 24.7, r"$5 \sigma$", color="red")
        
        axins = inset_axes(ax, width='50%', height='40%', loc="upper right")
        axins.plot(alpha, nll)
        axins.grid(alpha=1)
        
        x_ins_limit = self.__get_inters_points(alpha, nll, 1.5)
        
        axins.set_xlim(x_ins_limit[0], x_ins_limit[2])
        axins.set_ylim(0.0, 1.5)
        
        axins.set_yticks([0.0, 1.0])
        axins.set_xticks(used_part["sigma_min_sigma"])
        
        plt.setp(axins.get_xticklabels(), visible=True)
        plt.setp(axins.get_yticklabels(), visible=True)
        
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="grey", lw=0.5, alpha=0.5)
        
        if to_file:
            if tag == "unbinned":
                plt.savefig(os.path.join(self.save_dir, "alpha_only_scan_{}.png".format(tag)))
            if tag == "binned":
                plt.savefig(
                    os.path.join(self.save_dir, "alpha_only_scan_{}_{}.png".format(tag, round(self.bin_width, 2))))
        if not to_file:
            plt.show()
    
    # 2d plots
    # ------------------------------------------------
    def plot_2d_likelihood_contour(self, tag="unbinned", nll_scan_points=300, contour_scan_points=50,
                                   to_file=False):
        """
        Function to Plot two dimensional parameter minimisation. Signal strength alpha and mass

        :param tag: "binned" or "unbinned"
        :param nll_scan_points: number of points that the arrays will have during the function
        :param contour_scan_points: number of points used for contour
        :param to_file: saves to File if True
        """
        
        print("plot_2d_likelihood_contour, tag: {}, to_file: {}".format(tag, to_file))
        global count
        count = 0
        signal_func = self.signal_func
        background_func = self.background_func
        
        used_dict = {}
        if tag == "binned":
            if "calc_alpha_mass_binned_integrated" not in list(self.calculated_dump.keys()):
                self.calc_alpha_mass_binned_integrated(scan_points=nll_scan_points)
                used_dict = self.calculated_dump["calc_alpha_mass_binned_integrated"]
        if tag == "unbinned":
            if "calc_alpha_mass_unbinned_point_wise" not in list(self.calculated_dump.keys()):
                self.calc_alpha_mass_unbinned_point_wise(scan_points=nll_scan_points)
                used_dict = self.calculated_dump["calc_alpha_mass_unbinned_point_wise"]
        
        n_scale = np.sum(self.data_com.data["data"])
        used_x_point = self.data_com.x_range
        used_d_point = self.data_com.data["data"]
        n_mc_b = np.sum(self.data_com.data["mc_bac"])
        n_mc_s = np.sum(self.data_com.data["mc_sig"])
        
        def calc_ll_point_wise(x, y):
            print("error")
            return x, y
        
        if tag == "binned":
            background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(self.func_func_param[background_func.__name__]))[0]
            
            @np.vectorize
            def calc_ll_point_wise(alpha_point, mass_point):
                global count
                count += 1
                if count % 1 == 0 and self.verbose:
                    print("Create grid: {} / {}".format(count, contour_scan_points ** 2), end="\r",
                          flush=True)
                
                sig_param = self.func_func_param[signal_func.__name__]
                sig_param[1] = mass_point
                signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(sig_param))[0]
                
                @np.vectorize
                def b_int_bac(x_point):
                    bac_func_args = self.func_func_param[background_func.__name__]
                    pass_val = background_func_norm * sci.quad(background_func,
                                                               x_point - self.bin_width / 2.,
                                                               x_point + self.bin_width / 2.,
                                                               args=tuple(bac_func_args))[0] * n_mc_b * self.bin_width
                    return pass_val
                
                @np.vectorize
                def b_int_sig(x_point):
                    sig_func_args = self.func_func_param[signal_func.__name__]
                    return signal_func_norm * sci.quad(signal_func,
                                                       x_point - self.bin_width / 2., x_point + self.bin_width / 2.,
                                                       args=tuple(sig_func_args))[0] * n_mc_s * self.bin_width
                
                @np.vectorize
                def used_func(_alpha_point, _x_point):
                    return (_alpha_point * b_int_sig(_x_point) + (1 - _alpha_point) * b_int_bac(_x_point))
                
                ln_temp = np.sum(
                    used_d_point * self.bin_width * np.log(used_func(alpha_point, used_x_point)) - used_func(
                        alpha_point, used_x_point))
                
                return -2 * ln_temp - used_dict["contour_min"]
        
        if tag == "unbinned":
            background_func_norm = 1. / sci.quad(background_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(self.func_func_param[background_func.__name__]))[0]
            
            @np.vectorize
            def calc_ll_point_wise(alpha_point, mass_point):
                
                global count
                count += 1
                if count % 100 == 0 and self.verbose:
                    print("Create grid: {} / {}".format(count, contour_scan_points ** 2), end="\r",
                          flush=True)
                
                used_x = self.data_raw.data
                sig_param = self.func_func_param[signal_func.__name__]
                sig_param[1] = mass_point
                signal_func_norm = 1. / sci.quad(signal_func, self.hist_range[0], self.hist_range[1],
                                                 args=tuple(sig_param))[0]
                
                @np.vectorize
                def b_point_bac(x_point):
                    bac_func_args = self.func_func_param[background_func.__name__]
                    return background_func_norm * background_func(x_point, *bac_func_args)
                
                @np.vectorize
                def b_point_sig(x_point):
                    return signal_func_norm * signal_func(x_point, *sig_param)
                
                @np.vectorize
                def used_func(alpha_p, x_point):
                    return alpha_p * b_point_sig(x_point) + (1 - alpha_p) * b_point_bac(x_point)
                
                used_exp_val = {}
                
                @np.vectorize
                def expectation_value(alpha_p):
                    def to_int_func(x_point):
                        return x_point * (alpha_p * b_point_sig(x_point) + (1 - alpha_p) * b_point_bac(x_point))
                    
                    if alpha_p not in used_exp_val.keys():
                        area = sci.quad(to_int_func, self.hist_range[0], self.hist_range[1],
                                        points=np.linspace(self.hist_range[0], self.hist_range[1], 42), epsrel=1e-10)[0]
                        used_exp_val[alpha_p] = area
                        return area
                    if alpha_p in used_exp_val.keys():
                        return used_exp_val[alpha_p]
                
                exp_val = expectation_value(alpha_point)
                p = np.meshgrid(alpha_point, used_x)
                ln = -exp_val + np.sum(np.log(exp_val * used_func(p[0], p[1])))
                
                draw_ln = - 2 * ln - used_dict["contour_min"]
                return draw_ln
        
        mass_points = self.__get_inters_points(used_dict["mass_array"], used_dict["mass_nll_array"], 1)
        mass_points_2 = self.__get_inters_points(used_dict["mass_array"], used_dict["mass_nll_array"], 4)
        alpha_points = self.__get_inters_points(used_dict["alpha_array"], used_dict["alpha_nll_array"], 1)
        alpha_points_2 = self.__get_inters_points(used_dict["alpha_array"], used_dict["alpha_nll_array"], 4)
        observed_z = str(round(np.sqrt(used_dict["alpha_nll_array"][0]), 3))
        
        mass_array = np.linspace(mass_points_2[0] - 0.25, mass_points_2[-1] + 0.25, contour_scan_points)
        alpha_array = np.linspace(alpha_points_2[0] - 0.15, alpha_points_2[-1] + 0.15, contour_scan_points)
        
        X, Y = np.meshgrid(alpha_array, mass_array)
        print("do_contour")
        Z = calc_ll_point_wise(X, Y)
        
        f = plt.figure(figsize=(14, 8))
        f.subplots_adjust(left=0.15, bottom=0.07, right=0.98, top=0.95, wspace=0.23, hspace=1.0)
        gs = gridspec.GridSpec(1, 1, figure=f)
        gs0 = gridspec.GridSpecFromSubplotSpec(7, 2, subplot_spec=gs[0])
        
        ax1 = f.add_subplot(gs0[:-3, :])
        ax1.plot(used_dict["alpha_array"], used_dict["alpha_nll_array"], label=r"$-2\ln(\mathcal{L})-\rm{profile}$")
        ax1.annotate("{}".format(observed_z) + r"$\sigma$",
                     xy=(0.0, used_dict["alpha_nll_array"][0]), xytext=(0.14, 14),
                     arrowprops=dict(width=1.0, color="black"))
        ax1.hlines([4, 9, 16, 25], 0, 0.05, color="red")
        ax1.text(0.0525, 3.7, r"$2 \sigma$", color="red")
        ax1.text(0.0525, 8.7, r"$3 \sigma$", color="red")
        ax1.text(0.0525, 15.7, r"$4 \sigma$", color="red")
        ax1.text(0.0525, 24.7, r"$5 \sigma$", color="red")
        ax1.set_ylim(0, 40)
        ax1.set_xlim(0, 0.8)
        ax1.set_xlabel(r"$\alpha_{\rm{s}}$")
        ax1.set_ylabel(r"$- 2 \ln \left( \frac{\mathcal{L}}{\mathcal{L}_{\rm{min}}} \right)$")
        ax1.legend(loc="upper left")
        keys_func_names = list(self.func_names.keys())
        if tag == "unbinned":
            ax1.set_title("{}: {} + {}".format(tag, self.func_names[keys_func_names[0]],
                                               self.func_names[keys_func_names[1]]))
        if tag == "binned":
            ax1.set_title("{} ({} GeV per bin): {} + {}".format(tag, np.round(self.bin_width, 2),
                                                                self.func_names[keys_func_names[0]],
                                                                self.func_names[keys_func_names[1]]))
        
        axins = inset_axes(ax1, width='45%', height='50%', loc="upper right")
        
        axins.plot(used_dict["alpha_array"], used_dict["alpha_nll_array"], label=r"$-2\ln(\mathcal{L})-\rm{profile}$")
        mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="grey", lw=0.5, alpha=0.5)
        axins.grid(alpha=0.5)
        axins.set_xlim(alpha_points[0] - 0.05, alpha_points[2] + 0.05)
        axins.set_ylim(0.0, 1.5)
        axins.set_yticks([0.0, 1.0])
        axins.set_xticks(alpha_points)
        axins.grid(alpha=1)
        plt.setp(axins.get_xticklabels(), visible=True)
        plt.setp(axins.get_yticklabels(), visible=True)
        
        ax2 = f.add_subplot(gs0[4:7, :-1])
        cs = ax2.contourf(X, Y, Z, levels=[0, 1, 4], colors=('red', 'blue', 'white'), alpha=0.125)
        
        error_bar = ax2.errorbar(alpha_points[1], mass_points[1],
                                 xerr=[[alpha_points[1] - alpha_points[0]], [alpha_points[2] - alpha_points[1]]],
                                 yerr=[[mass_points[1] - mass_points[0]], [mass_points[2] - mass_points[1]]],
                                 fmt="kx", capsize=5, elinewidth=0.5, markeredgewidth=0.5, label=r"$\rm{fit}$")
        
        x_ax_set = np.sort(np.array(list(set(np.concatenate([alpha_points, alpha_points_2])))))
        y_ax_set = np.sort(np.array(list(set(np.concatenate([mass_points, mass_points_2])))))
        
        ax2.set_xlim(x_ax_set[0] - 0.05, x_ax_set[-1] + 0.15)
        ax2.set_ylim(y_ax_set[0] - 0.125, y_ax_set[-1] + 0.125)
        ax2.set_xticks(x_ax_set)
        ax2.set_xticklabels([r"$-2\sigma$", r"$-\sigma$", r"$0\sigma$", r"$\sigma$", r"$2\sigma$"])
        ax2.set_yticks(y_ax_set)
        ax2.set_yticklabels([r"$-2\sigma$", r"$-\sigma$", r"$0\sigma$", r"$\sigma$", r"$2\sigma$"])
        ax2.set_ylabel(r"$m_{\rm{H}}$")
        ax2.set_xlabel(r"$\alpha_{\rm{s}}$")
        
        proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in cs.collections]
        proxy.append(error_bar)
        
        ax2.legend(proxy, [r"$\sigma$", r"$2\sigma$", r"$\rm{fit}$", "t"])
        
        ax3 = f.add_subplot(gs0[4:7, -1])
        ax3.plot(used_dict["mass_array"], used_dict["mass_nll_array"], label=r"$-2\ln(\mathcal{L})-\rm{profile}$")
        ax3.set_ylim(0, 1.5)
        ax3.set_xlim(mass_points[0] - 0.125, mass_points[-1] + 0.125)
        
        ax3.set_xticks(mass_points)
        ax3.set_yticks([0, 1])
        ax3.grid(alpha=1)
        ax3.legend(loc="upper center")
        ax3.set_xlabel(r"$m_{\rm{H}}$")
        ax3.set_ylabel(r"$- 2 \ln \left( \frac{\mathcal{L}}{\mathcal{L}_{\rm{min}}} \right)$")
        
        if to_file:
            if tag == "unbinned":
                plt.savefig(os.path.join(self.save_dir, "alpha_mass_scan_{}.png".format(tag)))
            if tag == "binned":
                plt.savefig(
                    os.path.join(self.save_dir, "alpha_mass_scan_{}_{}.png".format(tag, round(self.bin_width, 2))))
        if not to_file:
            plt.show()
    
    def plot_p0_estimation(self, unbinned=None, binned=None, to_file=False, filename=None):
        """
        Funtion to plot p0 estimation

        :param unbinned: None or filename as string
        :param binned: if list of strings then those files will be used. If None, then a calculation with a preset
                       number of bins will be done
        :param to_file: save as file if True
        """
        used_unbinned = [[0], [0], [0]]
        if type(unbinned).__name__ == "NoneType":
            self.calc_q0_unbinned_point_wise()
            used_unbinned = self.calculated_dump["calc_q0_unbinned_point_wise"]
            plt.plot(list((used_unbinned["mass"])), used_unbinned["p0"], color="black", label="Unbinned")
        
        if type(unbinned).__name__ == "str":
            used_unbinned = np.loadtxt(unbinned, delimiter=",").T
            plt.plot(list((used_unbinned[0])), used_unbinned[2], color="black", label=r"$p_0$ Abschätzung")
        
        if type(binned).__name__ == "list":
            ls_list = ["--", "-.", ":"]
            for i in range(len(binned)):
                used_temp = np.loadtxt(binned[i], delimiter=",").T
                plt.plot(used_temp[0], used_temp[2], color="black", alpha=0.5, ls=ls_list[i],
                         label="Histogramm mit {},{} GeV/bin".format(binned[i].split(".")[0].split("_")[-1],
                                                                     binned[i].split(".")[-2]))
        
        s1, s2, s3 = 0.317310507863 / 2., 0.045500263896 / 2., 0.002699796063 / 2.
        s4, s5, s6 = 0.000063342484 / 2., 0.000000573303 / 2., 0.000000001973 / 2.
        mass_scan_array = self.__create_mh_scan_array_for_p0()
        
        plt.hlines([s1, s2, s3, s5, s6], mass_scan_array[0], mass_scan_array[-1], colors="red", alpha=0.75)
        plt.hlines([s4], mass_scan_array[0], 132.0, colors="red", alpha=0.75)
        plt.text(mass_scan_array[1], s1 * 1.1, r"$\sigma$", color="red")
        plt.text(mass_scan_array[1], s2 * 1.1, r"$2\sigma$", color="red")
        plt.text(mass_scan_array[1], s3 * 1.1, r"$3\sigma$", color="red")
        plt.text(mass_scan_array[1], s4 * 1.1, r"$4\sigma$", color="red")
        plt.text(mass_scan_array[1], s5 * 1.1, r"$5\sigma$", color="red")
        plt.text(mass_scan_array[1], s6 * 1.1, r"$6\sigma$", color="red")
        plt.yscale("log")
        plt.xlim(mass_scan_array[0], mass_scan_array[-1])
        plt.ylim(s4 * 0.5, 1.25)
        plt.xlabel(r"$m_{4l}$ in GeV")
        plt.ylabel(r"$p_{0}$")
        plt.legend(loc="lower right")
        
        # noinspection PyTypeChecker
        print("Massenminimum: {} | p0: {}| Z: {}".format(
            used_unbinned[0][np.argmin(used_unbinned[2])], np.amin(used_unbinned[2]),
            np.sqrt(np.amax(used_unbinned[1]))))
        
        if to_file:
            if type(filename).__name__ == "NoneType":
                plt.savefig(os.path.join(self.save_dir, "p0_estimation.png"))
            if type(filename).__name__ != "NoneType":
                plt.savefig(os.path.join(self.save_dir, filename))
        if not to_file:
            plt.show()
    
    # Preparation for sigma
    # ------------------------------------------------
    def plot_sigma(self, filepath=None, only_set_val=False, to_file=True, filename=None):
        """
        Function to calculate sigma gauss estimation out of multiple signal MC.

        :param filepath: used path witch fit results
        :param only_set_val: just return the fitted values
        :param to_file: save plot to file
        :param filename: name that will be given to the saved plot
        :return: fitted values if only_set_val=True
        """
        
        def get_sigmas_and_mus(filepath, errors="simple"):
            used_file_names = os.listdir(filepath)
            used_file_names = [os.path.join(filepath, item) for item in used_file_names]
            
            sigma_gauss = np.zeros(len(used_file_names))
            mu_gauss = np.zeros(len(used_file_names))
            sigma_errors = np.zeros((2, len(used_file_names)))
            mu_errors = np.zeros((2, len(used_file_names)))
            
            sigma_simple_errors = np.zeros(len(used_file_names))
            mu_simple_errors = np.zeros(len(used_file_names))
            
            for i in range(len(used_file_names)):
                with open(used_file_names[i]) as f:
                    temp = list(yaml.unsafe_load_all(f.read()))[0]["fit_results"]
                    sigma_gauss[i] = temp["parameter_values"][0]
                    mu_gauss[i] = temp["parameter_values"][1]
                    sigma_errors[0][i] = abs(temp["asymmetric_parameter_errors"][0][0])
                    sigma_errors[1][i] = abs(temp["asymmetric_parameter_errors"][0][1])
                    
                    mu_errors[0][i] = abs(temp["asymmetric_parameter_errors"][1][0])
                    mu_errors[1][i] = abs(temp["asymmetric_parameter_errors"][1][1])
                    
                    sigma_simple_errors[i] = temp['parameter_errors'][0]
                    mu_simple_errors[i] = temp['parameter_errors'][1]
            
            if errors == "simple":
                return sigma_gauss, sigma_simple_errors, mu_gauss, mu_simple_errors
            if errors == "asymmetric":
                return sigma_gauss, sigma_errors, mu_gauss, mu_errors
        
        if type(filepath).__name__ == "NoneType":
            filepath = os.path.join(self.save_dir, "gauss_mc_fits_for_p0")
        
        sig, sig_err, mu, mu_err = get_sigmas_and_mus(filepath=filepath)
        Poly.set_x_mean(np.mean(mu))
        xy_d = K2.XYContainer(mu, sig)
        xy_d.add_simple_error("x", np.array(sig_err))
        xy_d.add_simple_error("y", np.array(mu_err))
        xy_fit = K2.XYFit(xy_d, Poly.legendre_grade_1)
        xy_fit.do_fit()
        xy_fit.to_file("sigma_gauss_estimate_fit_results.yml", calculate_asymmetric_errors=True)
        xy_fit.assign_parameter_latex_names(x="x", a="a", b="b")
        xy_fit._model_function._formatter._latex_x_name = "m_{4\ell}"
        xy_fit.assign_model_function_latex_name("f_1")
        xy_fit.assign_model_function_latex_expression("{0}({x}- \\bar{{m}}_{{4\ell}}) + {1}")
        if only_set_val:
            pass_var = xy_fit.parameter_values
            xy_fit.to_file(os.path.join(self.save_dir, "fit_results_from_sigma_estimation.yaml"))
            return pass_var
        
        p = K2.Plot(xy_fit)
        p.customize("data", "label", ["Messung"])
        p.customize("model_line", "label", ["Modell"]).customize("model_line", "color", ["red"])
        p.customize("model_error_band", "label", ["Modellunsicherheit"])
        p.plot(with_ratio=True, with_legend=True, with_fit_info=True, with_asymmetric_parameter_errors=False)
        p.axes[0]["main"].set_ylabel(r"$\sigma_{\rm{G}}$ in GeV")
        p.axes[0]["ratio"].set_xlabel(r"$m_{4\ell}$ in GeV")
        # fig = p.figures[-1]
        # fig.set_tight_layout(False)
        
        if to_file:
            if type(filename).__name__ == "NoneType":
                plt.savefig(os.path.join(self.save_dir, "p0_estimation.png"))
            if type(filename).__name__ != "NoneType":
                plt.savefig(os.path.join(self.save_dir, filename))
        if not to_file:
            plt.show()


class CalcErrorZ(object):
    """
    Class to calculate the error on significance.
    """
    
    pass_test_obj = None
    pass_scan_points = None
    pass_save_dir = None
    
    def __init__(self, testing_obj, scan_points=200, reduce_list=None):
        self.save_dir = ["./z_err", "./z_mass_err"]
        for item in self.save_dir:
            if not os.path.exists(item):
                os.mkdir(item)
        
        self.testing_obj = testing_obj
        self.scan_points = scan_points
        self.__set_vars()
        self.reduce_list = reduce_list
    
    def __set_vars(self):
        """
        Set Variables for @staticmethods
        """
        CalcErrorZ.pass_save_dir = self.save_dir
        CalcErrorZ.pass_test_obj = self.testing_obj
        CalcErrorZ.pass_scan_points = self.scan_points
    
    def __get_all_possible_sig_bac_params(self, use_mass_error=True):
        """
        Function to create all possible combinations of parameter variation.

        :param use_mass_error: True == mass is locked; False == mass is a free fit parameter
        :return: a list of tuples of all possible combinations
        """
        
        init_signal_par = self.testing_obj.func_func_param[self.testing_obj.signal_func.__name__]
        init_backgr_par = self.testing_obj.func_func_param[self.testing_obj.background_func.__name__]
        
        init_signal_err = np.array([])
        if use_mass_error:
            init_signal_err = [[-0.04744222795601664, 0.0, 0.043539248834648166],
                               [-0.023900703200297076, 0.0, 0.024157017066654077],
                               [-0.050153560018870884, 0.0, 0.04955412676125166],
                               [-0.06811186251199337, 0.0, 0.06734114740543501],
                               [-0.27967528810147, 0.0, 0.32848165191960554],
                               [-1.0452913055092519, 0.0, 1.4616049313897979]]
        if not use_mass_error:
            init_signal_err = [[-0.04744222795601664, 0.0, 0.043539248834648166],
                               [0.0, 0.0, 0.0],
                               [-0.050153560018870884, 0.0, 0.04955412676125166],
                               [-0.06811186251199337, 0.0, 0.06734114740543501],
                               [-0.27967528810147, 0.0, 0.32848165191960554],
                               [-1.0452913055092519, 0.0, 1.4616049313897979]]
        init_backgr_err = [[-0.00034863678940386816, 0.0, 0.0003486367894076063],
                           [-1.6557538709537503e-05, 0.0, 1.6557538709602562e-05],
                           [-9.635762833577886e-07, 0.0, 9.635762833416374e-07]]
        
        sig_pars = [[] for item in init_signal_par]
        for i in range(len(init_signal_err[0])):
            for j in range(len(init_signal_par)):
                sig_pars[j].append(init_signal_par[j] + init_signal_err[j][i])
        
        used_sig_par = []
        # TODO: find a more elegant way
        for p0 in itertools.permutations(sig_pars[0], 1):
            for p1 in itertools.permutations(sig_pars[1], 1):
                for p2 in itertools.permutations(sig_pars[2], 1):
                    for p3 in itertools.permutations(sig_pars[3], 1):
                        for p4 in itertools.permutations(sig_pars[4], 1):
                            for p5 in itertools.permutations(sig_pars[5], 1):
                                used_sig_par.append((p0[0], p1[0], p2[0], p3[0], p4[0], p5[0]))
        used_sig_par = list(set(used_sig_par))
        
        bac_pars = [[] for item in init_backgr_par]
        for i in range(len(init_backgr_err[0])):
            for j in range(len(init_backgr_par)):
                bac_pars[j].append(init_backgr_par[j] + init_backgr_err[j][i])
        
        used_bac_par = []
        # TODO: find a more elegant way
        for p0 in itertools.permutations(bac_pars[0], 1):
            for p1 in itertools.permutations(bac_pars[1], 1):
                for p2 in itertools.permutations(bac_pars[2], 1):
                    used_bac_par.append((p0[0], p1[0], p2[0]))
        
        used_bac_par = list(set(used_bac_par))
        
        sig_bac = []
        for ps in itertools.permutations(used_sig_par, 1):
            for pb in itertools.permutations(used_bac_par, 1):
                sig_bac.append((ps[0], pb[0]))
        
        sig_bac = list(set(sig_bac))
        
        return sig_bac
    
    def get_pairs(self, use_mass_error=True, used_reduce_list=None):
        """
        Function to create pairs list of pairs like (signal_param, background_param, num)

        :param use_mass_error: True == mass is locked; False == mass is a free fit parameter
        :return: a list of tuples of all possible combinations
        """
        
        temp_p = self.__get_all_possible_sig_bac_params(use_mass_error=use_mass_error)
        temp_num = [i for i in range(len(temp_p))]
        temp_pairs = list(zip(temp_p, temp_num))
        if type(used_reduce_list).__name__ != "NoneType":
            used_file_list_num = np.unique(np.sort(np.array(np.load(used_reduce_list), dtype=int)))
            to_pass_pairs = []
            for item in temp_pairs:
                if item[1] in used_file_list_num:
                    continue
                to_pass_pairs.append(item)
            return to_pass_pairs
        return temp_pairs
    
    @staticmethod
    def calc_z_error_on_alpha_only(pair):
        """
        Function to calculate a significance and save thecorresponding parameters and the significance
        alpha only

        :param pair: used pair
        """
        
        temp_testing_obj = CalcErrorZ.pass_test_obj
        temp_testing_obj.func_func_param[temp_testing_obj.signal_func.__name__] = np.array(list(pair[0][0]))
        temp_testing_obj.func_func_param[temp_testing_obj.background_func.__name__] = np.array(list(pair[0][1]))
        z_temp = temp_testing_obj.calc_alpha_unbinned_point_wise(scan_points=CalcErrorZ.pass_scan_points,
                                                                 significance_only=True)
        with open(os.path.join(CalcErrorZ.pass_save_dir[0], "{}__{}.csv".format(pair[1], z_temp)), "w") as my_empty_csv:
            my_empty_csv.write(str(pair))
    
    @staticmethod
    def calc_z_error_on_alpha_and_mass(pair):
        """
        Function to calculate a significance and save thecorresponding parameters and the significance
        alpha and mass

        :param pair: used pair
        """
        
        temp_testing_obj = CalcErrorZ.pass_test_obj
        temp_testing_obj.func_func_param[temp_testing_obj.signal_func.__name__] = np.array(list(pair[0][0]))
        temp_testing_obj.func_func_param[temp_testing_obj.background_func.__name__] = np.array(list(pair[0][1]))
        z_temp = temp_testing_obj.calc_alpha_mass_unbinned_point_wise(scan_points=CalcErrorZ.pass_scan_points,
                                                                      significance_only=True)
        with open(os.path.join(CalcErrorZ.pass_save_dir[0], "z_mass_err/{}__{}.csv".format(pair[1], z_temp)),
                  "w") as my_empty_csv:
            my_empty_csv.write(str(pair))


class ProcessHelper(object):
    
    @staticmethod
    def create_tuple(file_path):
        data_type = [("ru", "FourElectron", "FourMuon"), ("mc", "_4el_", "_4mu_")]
        filename = [file_path + item for item in os.listdir(file_path)]
        filetuple = []
        used_names = data_type[0]
        pair = namedtuple("Pair", "name, particle")
        if "mc_" in file_path:
            used_names = data_type[1]
        for item in filename:
            if used_names[1] in item:
                filetuple.append(pair(item, "electron"))
                continue
            if used_names[2] in item:
                filetuple.append((item, "muon", used_names[0]))
                continue
            else:
                filetuple.append(pair(item, "both"))
        return filetuple
    
    @staticmethod
    def strip_affix(name):
        name, name_format = name.split(".")[0], name.split(".")[-1]
        name = "_".join(name.split("_")[:-1])
        return "{}.{}".format(name, name_format)
    
    @staticmethod
    def change_on_affix(name, affix):
        to_place_path, name_ = os.path.split(name)
        name_ = ProcessHelper.strip_affix(name_)
        name_ = "{}_{}.{}".format(name_.split(".")[0], affix, name_.split(".")[-1])
        to_place_path = "{}_{}".format(to_place_path.split("_")[0], affix)
        return os.path.join(to_place_path, name_).replace("\\", "/")




