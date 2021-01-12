# -*- coding: UTF-8 -*-
from copy import deepcopy

import awkward1 as awk
import numpy as np


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray

npawk = lambda x: np.array(awk.to_list(x))
from_to_awk = lambda x: awk.from_iter(awk.to_list(x))


class FilterInit(object):
    """
    Class that introduces certain cuts and thus restricts the leptons in the events.
    """
    
    a_calc_instance = None
    a_filter_instance = None
    
    def __init__(self, *, calc_obj=None, filter_obj=None):
        FilterInit.a_filter_instance = FilterInit if filter_obj is None else filter_obj
        FilterInit.a_calc_instance = CalcInit if calc_obj is None else calc_obj
    
    # Used allowed parameter in class
    va = {'relative_isolation': None, 'misshit': 1.0,
          'pt_1': None, 'pt_2': None, 'pt_min_mu': 5.0, 'pt_min_el': 7.0,
          'max_pseudorapidity_mu': None, 'max_pseudorapidity_el': None,
          'type_mu': 'G', 'type_el': 'T',
          'min_2l_m': 4.0, 'min_4l_m': 70.0, 'max_4l_m': 180.0,
          'sip3d': None, 'dxy': None, 'dz': None,
          'z1_min': None, 'z1_max': None, 'z2_min': None, 'z2_max': None,
          'delta_r': 0.02}
    
    @staticmethod
    def combined_charge(charge, combine_num):
        """
        Tests whether an electrically neutral charge combination is possible.

        :param charge: ndarray
                       1D array containing data with "int" type.
        :param combine_num: int
                            4 if lepton_type is not "both", 2 else
        :return: bool
        """
    
    @staticmethod
    def relative_isolation(rel_pf_iso):
        """
        Checks if relative_isolation is smaller than the allowed value.

        :param rel_pf_iso: ndarray
                           1D array containing data with `float` type.
        :return: ndarray
                 1D array containing data with `bool` type.
        """
    
    @staticmethod
    def pt_exact(p_t, lepton_type=None):
        """
        Checks if the exact pedingun regarding pt is observed.
        (>20 GeV: >= 1; >10 GeV: >= 2; >Minimum pt: >= 4).

        :param p_t: if lepton_type != "both"
                        ndarray
                        1D array containing data with `float` type.
                    if lepton_type == "both"
                        (ndarray, ndarray): (pt_muon, pt_electron)
                        two 1D array containing data with `float` type.
        :param lepton_type: str
                            "muon" or "electron" or "both"
        :return: bool
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
    def pt_min(p_t, lepton_type):
        """
        Checks whether the minimum transverse impulse of the individual leptons is maintained.
        A case distinction between electrons and muons must be made.

        :param p_t: ndarray
                    1D array containing data with `float` type.
        :param lepton_type: str
                            "muon" or "electron"
        :return: ndarray
                 1D array containing data with `bool` type.
        """
    
    @staticmethod
    def pseudorapidity(pseudorapidity, lepton_type):
        """
        Checks if the pseudorapidity of leptons is valid.

        :param pseudorapidity: ndarray
                    1D array containing data with "float" type.
        :param lepton_type: str
                         "muon"; "electron" or "both"
        :return: ndarray
                 1D array containing data with "bool" type.
        """
    
    @staticmethod
    def impact_parameter(sip3d, dxy, dz):
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
    def delta_r(delta_r):
        """
        Checks if delta_r is smaller than the allowed value.

        :param delta_r: ndarray
                        1D array containing data with `float` type.
        :return: ndarray
                 1D array containing data with `bool` type.
        """
        return delta_r > FilterInit.va["delta_r"]
    
    @staticmethod
    def misshits(misshits):
        """
        Checks if the minimum number of misshits was kept.

        :param misshits:
        :return:
        """
        return misshits <= FilterInit.va["misshit"]
    
    @staticmethod
    def lepton_type(typ, lepton_type):
        """
        Checks for the permitted classification of leptons.

        :param typ: ndarray
                    1D array containing data with "float" type.
        :param lepton_type: str
                         "muon"; "electron" or "both"
        :return: ndarray
                 1D array containing data with "bool" type.
        """
        _typ = FilterInit.va["type_mu"] if lepton_type == "muon" else FilterInit.va["type_el"]
        return np.array((typ == _typ))
    
    @staticmethod
    def invariant_mass(mass_list, number_of_leptons, lepton_type="muon"):
        """
        Checks if a mass (or a combination of masses) meets the condition satisfying that
        0 < min_mass <= mass <= max_mass.

        :param mass_list: ndarray
                          1D array containing data with "float" type.
        :param number_of_leptons: int
                                  4 if lepton_type is not "both", 2 else
        :param lepton_type: str
                         "muon"; "electron" or "both"
        :return: bool
        """
        if number_of_leptons == 4:
            accept_array = (mass_list > FilterInit.va["min_4l_m"]) & (mass_list < FilterInit.va["max_4l_m"])
            return True if np.sum(accept_array) >= 1 else False
        if number_of_leptons == 2:
            accept_array = (mass_list > FilterInit.va["min_2l_m"])
            if lepton_type != "both":
                return True if np.sum(accept_array) >= 2 else False
            if lepton_type == "both":
                return True if np.sum(accept_array) >= 1 else False


class CalcInit(object):
    """
    Class for the calculation of certain sizes that are used for
    the cuts or are essential for the reconstruction.
    """
    
    c_calc_instance = None
    c_filter_instance = None
    
    def __init__(self, *, calc_obj=None, filter_obj=None):
        CalcInit.c_filter_instance = FilterInit if filter_obj is None else filter_obj
        CalcInit.c_calc_instance = CalcInit if calc_obj is None else calc_obj
    
    @staticmethod
    def pseudorapidity(px=None, py=None, pz=None, energy=None):
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
    def invariant_mass_square(px, py, pz, energy=None):
        """
        Calculates the square of the invariant mass.
        Optional with or without energy.
        Optionally with or without pseudorapidity and phi.

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
        if isinstance(phi1, list) or isinstance(phi1, list):
            phi1, phi2 = np.array(phi1), np.array(phi2)
        return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    
    @staticmethod
    def delta_r(pseudorapidity, phi):
        """
        Calculation of delta_r.

        :param pseudorapidity: ndarray
                    1D array containing data with "float" type.
        :param phi: ndarray
                    1D array containing data with "float" type.
        :return: ndarray
                 1D array containing data with "float" type.
        """
        if isinstance(pseudorapidity, list) or isinstance(phi, list):
            pseudorapidity, phi = np.array(pseudorapidity), np.array(phi)
        return np.sqrt((pseudorapidity[0] - pseudorapidity[1]) ** 2 + CalcInit.delta_phi(phi[0], phi[1]) ** 2)
    
    @staticmethod
    def mass_4l_out_zz(awk_array, lepton_type="muon"):
        
        _calc_mass = CalcInit.c_calc_instance.invariant_mass_square
        
        if lepton_type != "both":
            _idx = awk.from_iter([np.append(npawk(ev.z1_index), npawk(ev.z2_index)) for ev in awk_array])
            awk_array["mass_4l"] = np.sqrt(awk.from_iter([_calc_mass(ev[f"{lepton_type}_px"][_id],
                                                                     ev[f"{lepton_type}_py"][_id],
                                                                     ev[f"{lepton_type}_pz"][_id],
                                                                     ev[f"{lepton_type}_energy"][_id]) for ev, _id in zip(awk_array, _idx)]))
            return awk_array
        
        if lepton_type == "both":
            
            _awk_tmp = deepcopy(awk_array)
            _n_vars = ["energy", "px", "py", "pz"]
            _name = lambda x: 'muon' if x == 'm' else 'electron'
            
            for name in _n_vars:
                _awk_tmp[name] = awk.from_iter([np.append(npawk(ev[f"{_name(ev.z1_tag)}_{name}"][npawk(ev.z1_index)]),
                                                          npawk(ev[f"{_name(ev.z2_tag)}_{name}"][npawk(ev.z2_index)])) for ev in _awk_tmp])
            
            awk_array["mass_4l"] = np.sqrt(awk.from_iter([_calc_mass(ev.px, ev.py, ev.pz, ev.energy) for ev in _awk_tmp]))
            
            return awk_array
    
    @staticmethod
    def zz_and_index(awk_array, lepton_type="muon"):
        
        lep = lepton_type
        _n_vars = ["energy", "tag", "px", "py", "pz", "charge", "pt", "phi", "pseudorapidity", "index"]
        _filter_charge = CalcInit.c_filter_instance.combined_charge
        _filer_pt_exact = CalcInit.c_filter_instance.pt_exact
        _calc_mass = CalcInit.c_calc_instance.invariant_mass_square
        _calc_delta_r, _filter_delta_r = CalcInit.delta_r, FilterInit.delta_r
        _filter_zz = CalcInit.c_filter_instance.zz
        _filter_exact_pt = CalcInit.c_filter_instance.pt_exact
        
        z_mass = 91.1876
        
        # add stuff to _awk_tmp
        
        def add_tag(name, tag, dummy, count=False):
            _awk_tmp[name] = awk.from_iter([[f"{tag}{f'{_i}' if count else ''}" for _i, _ in enumerate(it)] for it in _awk_tmp[dummy]])
        
        def add_index(name, dummy):
            _awk_tmp[name] = awk.from_iter([[_i for _i, _ in enumerate(it)] for it in _awk_tmp[dummy]])
        
        # cuts
        
        def _flavour_count(comb, mode="2e & 2mu"):
            _str = ",".join([f"{it}" for it in awk.to_list(comb)])
            if mode == "2e & 2mu":
                return True if (_str.count("e") == 2 and _str.count("m") == 2) else False
            if mode == "2e | 2mu":
                return True if ((_str.count("e") == 2 or _str.count("m") == 2) and len(awk.to_list(comb)) == 2) else False
        
        def _get_cut_on_variable(_awk_array_part, _func, _func_kwargs):
            return awk.from_iter([[_func(npawk(subit), **_func_kwargs) for subit in it] for it in _awk_array_part])
        
        def _get_cut_on_z1_combination(pairs_):
            
            def _contained_in_combination(item, test):
                # for 4e or 4mu channel only
                if item[0] == test[0] and item[1] == test[1]:
                    return False  # True
                if item[0] in test or item[1] in test:
                    return False
                else:
                    return True
            
            return [[_contained_in_combination(npawk(subit), npawk(it.z1_index)) for subit in it[f"{lep}_index"]] for it in pairs_]
        
        # calc stuff
        
        def calc_masses(energy, px, py, pz):
            _masses = []
            for _ev in zip(energy, px, py, pz):
                _ev_mass = np.array([_calc_mass(npawk(_px), npawk(_py), npawk(_pz), npawk(_e)) for _e, _px, _py, _pz in zip(*_ev)])
                _ev_mass[_ev_mass < 0.0] = 0.0
                _masses.append(np.sqrt(_ev_mass))
            return _masses
        
        def calc_delta_r(eta_, phi_):
            return [[_calc_delta_r(npawk(eta), npawk(phi)) for eta, phi in zip(*it)] for it in zip(eta_, phi_)]
        
        def _get_z1_z2_in_ev(__ev, mixed=False):
            
            if not mixed:
                __ev["z1_mass"] = __ev.z_masses[min(abs(__ev.z_masses - z_mass)) == (abs(__ev.z_masses - z_mass))][0]
                __ev["z1_index"] = __ev[f"{lep}_index"][np.where(__ev.z_masses == __ev.z1_mass)[0]][0]
                
                _cut = awk.from_iter(_get_cut_on_z1_combination([__ev])[0])
                
                __ev.z_masses = __ev.z_masses[_cut]
                __ev[f"{lep}_index"] = __ev[f"{lep}_index"][_cut]
                
                __ev["z2_mass"] = max(__ev.z_masses)
                __ev["z2_index"] = __ev["index" if mixed else f"{lep}_index"][np.where(__ev.z_masses == __ev.z2_mass)[0]][0]
            
            if mixed:
                __ev["z1_mass"] = __ev.z_masses[min(abs(__ev.z_masses - z_mass)) == (abs(__ev.z_masses - z_mass))][0]
                __ev["z1_index"] = __ev.index[np.where(__ev.z_masses == __ev.z1_mass)[0]][0]
                
                __ev["z1_tag"] = awk.to_list(__ev.tag[np.where(__ev.z_masses == __ev.z1_mass)[0]][0])[0][0]
                
                _cut = awk.from_iter([npawk(comb)[0][0] != awk.to_list(__ev.z1_tag)[0] for comb in __ev.tag[0]])
                __ev.z_masses = __ev.z_masses[0][_cut]
                __ev.index = __ev.index[0][_cut]
                __ev.tag = __ev.tag[0][_cut]
                
                __ev["z2_mass"] = max(__ev.z_masses)
                
                __ev["z2_index"] = __ev.index[np.where(__ev.z_masses == __ev.z2_mass)[0]]
                __ev["z2_tag"] = awk.to_list(__ev.tag[np.where(__ev.z_masses == __ev.z2_mass)[0]][0])[0][0]
            
            return __ev
        
        def _get_dummy_z1_z2_on_reconstruction_failure(__ev, mixed=False):
            __ev["z1_mass"] = [-1.0]
            __ev["z2_mass"] = [-2.0]
            if not mixed:
                __ev["z1_index"] = __ev[0][f"{lep}_index"]
                __ev["z2_index"] = __ev[0][f"{lep}_index"]
            if mixed:
                __ev["z1_index"] = __ev[0].index
                __ev["z2_index"] = __ev[0].index
                __ev["z1_tag"] = "m"
                __ev["z2_tag"] = "e"
            return __ev
        
        #
        
        _awk_tmp = deepcopy(awk_array)
        
        if lep != "both":
            
            add_index(f"{lep}_index", f"{lep}_energy")
            
            pairs = awk.combinations(_awk_tmp[[it for it in _awk_tmp.fields if lep in it and any(subit in it for subit in _n_vars)]], 2)
            
            pairs = pairs[_get_cut_on_variable(pairs[f"{lep}_charge"], _filter_charge, dict(combine_num=2))]
            
            pairs["delta_r"] = calc_delta_r(pairs[f"{lep}_pseudorapidity"], pairs[f"{lep}_phi"])
            pairs = pairs[_get_cut_on_variable(pairs.delta_r, _filter_delta_r, dict())]
            
            pairs["z_masses"] = calc_masses(pairs[f"{lep}_energy"], pairs[f"{lep}_px"], pairs[f"{lep}_py"], pairs[f"{lep}_pz"])
            
            pairs = pairs[["z_masses", f"{lep}_index"]]
            
            _pairs = []
            
            for i, ev in enumerate(pairs):
                _ev = deepcopy(ev)
                while awk.to_list(ev.z_masses):
                    try:
                        ev = _get_z1_z2_in_ev(deepcopy(ev), mixed=False)
                        break
                    except ValueError:
                        ev = ev[ev.z_masses != ev.z_masses[min(abs(ev.z_masses - z_mass)) == (abs(ev.z_masses - z_mass))][0]]
                if not awk.to_list(ev.z_masses):
                    ev = _get_dummy_z1_z2_on_reconstruction_failure(deepcopy(_ev), mixed=False)
                
                _pairs.append(ev)
            
            pairs = awk.from_iter([awk.to_list(it) for it in _pairs])
            
            for field in pairs.fields:
                if "z1" in field or "z2" in field:
                    pairs[f"tmp_{field}"] = pairs[field]
            
            _swap = awk.concatenate(pairs.z2_mass > pairs.z1_mass)
            for i, j in [(1, 2), (2, 1)]:
                pairs[f"z{i}_mass"] = awk.from_iter(awk.to_list([it[f"tmp_z{j}_mass"] if s else it[f"z{i}_mass"] for it, s in zip(pairs, _swap)]))
                pairs[f"z{i}_index"] = awk.from_iter(awk.to_list([it[f"tmp_z{j}_index"] if s else it[f"z{i}_index"] for it, s in zip(pairs, _swap)]))
            
            for field in [it for it in pairs.fields if ("z1" in it or "z2" in it) and "tmp" not in it]:
                awk_array[field] = awk.from_iter([awk.to_list(it[0]) for it in pairs[field]])
            
            awk_array = awk_array[[_filter_zz(it.z1_mass, it.z2_mass) for it in awk_array]]
            
            awk_array = awk_array[awk.from_iter([_filter_exact_pt(event[f"{lep}_pt"][np.append(npawk(event.z1_index), npawk(event.z2_index))],
                                                                  lepton_type=lep)
                                                 for event in awk_array])]
            
            return awk_array
        
        if lep == "both":
            
            add_tag("muon_tag", "m", "muon_energy", count=True)
            add_tag("electron_tag", "e", "electron_energy", count=True)
            add_index("muon_index", "muon_energy")
            add_index("electron_index", "electron_energy")
            
            for mu_name, el_name in zip([it for it in _awk_tmp.fields if "muon_" in it],
                                        [it for it in _awk_tmp.fields if "electron_" in it]):
                if any(it in mu_name for it in _n_vars):
                    _awk_tmp[mu_name.replace("muon_", "")] = awk.concatenate([_awk_tmp[mu_name], _awk_tmp[el_name]], axis=1)
            
            pairs = awk.combinations(_awk_tmp[[it for it in _n_vars]], 2)
            pairs = pairs[_get_cut_on_variable(pairs.tag, _flavour_count, dict(mode="2e | 2mu"))]
            pairs = pairs[_get_cut_on_variable(pairs.charge, _filter_charge, dict(combine_num=2))]
            
            pairs["delta_r"] = calc_delta_r(pairs.pseudorapidity, pairs.phi)
            pairs = pairs[_get_cut_on_variable(pairs.delta_r, _filter_delta_r, dict())]
            
            pairs["z_masses"] = calc_masses(pairs.energy, pairs.px, pairs.py, pairs.pz)
            
            pairs = pairs[["z_masses", "tag", "index"]]
            
            _pairs = []
            
            for i, ev in enumerate(pairs):
                _ev = deepcopy(ev)
                while awk.to_list(ev.z_masses):
                    try:
                        ev = _get_z1_z2_in_ev(deepcopy(ev), mixed=True)
                        break
                    except ValueError:
                        ev = ev[ev.z_masses != ev.z_masses[min(abs(ev.z_masses - z_mass)) == (abs(ev.z_masses - z_mass))][0]]
                if not awk.to_list(ev.z_masses):
                    ev = _get_dummy_z1_z2_on_reconstruction_failure(deepcopy(_ev), mixed=True)
                
                _pairs.append(ev)
            
            pairs = awk.from_iter([awk.to_list(it) for it in _pairs])
            
            for field in pairs.fields:
                if "z1" in field or "z2" in field:
                    pairs[f"tmp_{field}"] = pairs[field]
            
            _swap = awk.concatenate(pairs.z2_mass > pairs.z1_mass)
            for i, j in [(1, 2), (2, 1)]:
                pairs[f"z{i}_mass"] = awk.from_iter(awk.to_list([it[f"tmp_z{j}_mass"] if s else it[f"z{i}_mass"] for it, s in zip(pairs, _swap)]))
                pairs[f"z{i}_index"] = awk.from_iter(awk.to_list([it[f"tmp_z{j}_index"] if s else it[f"z{i}_index"] for it, s in zip(pairs, _swap)]))
                pairs[f"z{i}_tag"] = awk.from_iter(awk.to_list([it[f"tmp_z{j}_tag"] if s else it[f"z{i}_tag"] for it, s in zip(pairs, _swap)]))
            
            for field in [it for it in pairs.fields if ("z1" in it or "z2" in it) and "tmp" not in it]:
                awk_array[field] = pairs[field]
            
            awk_array = awk_array[[_filter_zz(it.z1_mass, it.z2_mass)[0] for it in awk_array]]
            _name = lambda x: 'muon' if x == 'm' else 'electron'
            
            # awkward array specific...
            try:
                awk_array = awk_array[[_filter_exact_pt((np.array(npawk(awk_array[f"{_name(npawk(ev.z1_tag))}_pt"][npawk(ev.z1_index[0])])[0]),
                                                         np.array(npawk(awk_array[f"{_name(npawk(ev.z1_tag))}_pt"][npawk(ev.z2_index[0])])[0])),
                                                        lepton_type=lep)
                                       for ev in awk_array]]
            except ValueError:
                awk_array = awk_array[[_filter_exact_pt((np.array(npawk(awk_array[f"{_name(npawk(ev.z1_tag))}_pt"][0][npawk(ev.z1_index[0])])),
                                                         np.array(npawk(awk_array[f"{_name(npawk(ev.z1_tag))}_pt"][0][npawk(ev.z2_index[0])]))),
                                                        lepton_type=lep)
                                       for ev in awk_array]]
            
            return awk_array
    
    @staticmethod
    def possible_invariant_masses(awk_array, lepton_type, number_of_leptons=2):
        
        _n_vars = ["energy", "tag", "px", "py", "pz", "charge"]
        _filter_charge = CalcInit.c_filter_instance.combined_charge
        _calc_mass = CalcInit.c_calc_instance.invariant_mass_square
        
        # for 2el2mu channel if number_of_leptons = 4
        def flavour_count(comb):
            if awk.to_list(comb).count("e") == 2 and awk.to_list(comb).count("m") == 2:
                return True
            else:
                return False
        
        def get_masses_from(energy, px, py, pz):
            _masses = []
            for ev in zip(energy, px, py, pz):
                _ev_mass = np.array([_calc_mass(npawk(_px), npawk(_py), npawk(_pz), npawk(_e)) for _e, _px, _py, _pz in zip(*ev)])
                _ev_mass[_ev_mass < 0.0] = 0.0
                _masses.append(np.sqrt(_ev_mass))
            return _masses
        
        if lepton_type != "both":
            pairs = awk.combinations(awk_array[[it for it in awk_array.fields if lepton_type in it and
                                                any(subit in it for subit in _n_vars)]], number_of_leptons)
            
            _charge_cut = awk.from_iter([[_filter_charge(npawk(subit), number_of_leptons) for subit in it] for it in pairs[f"{lepton_type}_charge"]])
            pairs = pairs[_charge_cut]
            
            return awk.from_iter(get_masses_from(pairs[f"{lepton_type}_energy"], pairs[f"{lepton_type}_px"],
                                                 pairs[f"{lepton_type}_py"], pairs[f"{lepton_type}_pz"]))
        
        if lepton_type == "both":
            
            if number_of_leptons == 2:
                raise TypeError("Pass on muons and electrons separately with number_of_leptons=2")
            
            if number_of_leptons == 4:
                
                # "tagging" for flavour check in combinations
                awk_array["muon_tag"] = awk.from_iter([["m" for _ in it] for it in awk_array.muon_energy])
                awk_array["electron_tag"] = awk.from_iter([["e" for _ in it] for it in awk_array.electron_energy])
                
                for muon_name, electron_name in zip([it for it in awk_array.fields if "muon_" in it],
                                                    [it for it in awk_array.fields if "electron_" in it]):
                    if any(it in muon_name for it in _n_vars):
                        awk_array[muon_name.replace("muon_", "")] = awk.concatenate([awk_array[muon_name], awk_array[electron_name]], axis=1)
                
                pairs = awk.combinations(awk_array[[it for it in _n_vars]], number_of_leptons)
                
                _flavour_cut = [[flavour_count(subit) for subit in it] for it in pairs.tag]
                _charge_cut = [[_filter_charge(npawk(subit), number_of_leptons) for subit in it] for it in pairs.charge]
                pairs = pairs[awk.from_iter(_flavour_cut) & awk.from_iter(_charge_cut)]
                
                return awk.from_iter(get_masses_from(pairs.energy, pairs.px, pairs.py, pairs.pz))
