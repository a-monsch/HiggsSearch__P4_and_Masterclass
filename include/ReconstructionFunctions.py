import itertools

import numpy as np
import pandas as pd


def reconstruct_zz(row: pd.Series, pt_dict: dict = None):
    # for kwarg raw=True in pandas.apply for speedup:
    # using index-variant:
    # {0: run_eve_lumisec, 1: channel, 2: leptpns, 3: _z1, 4: _z2, 5: four_lep, 6:changed} -> (*)
    
    # skip reconstruction if nothing changed
    if (not row[6] and len(row[2]) == 4) or (not row[6] and pt_dict is None):
        return row
    
    z_mass = 91.1876
    
    # creating all possible two lepton combinations from a given event and labeling by leptons
    _combinations = {(c[0][0], c[1][0]): {"lep": [c[0][1], c[1][1]]} for c in itertools.combinations(enumerate(row[2]), 2)}
    
    # helper 1
    _valid_charge_combination = lambda _v: (_v["lep"][0].charge + _v["lep"][1].charge) == 0
    _valid_flavour_combination = lambda _v: _v["lep"][0].flavour == _v["lep"][1].flavour
    _valid_delta_r_combination = lambda _v: _v["lep"][0].delta_r(_v["lep"][1]) > 0.02
    
    for k, v in list(_combinations.items()):  # removing unphysical/bad combinations:
        if not _valid_charge_combination(v) or not _valid_flavour_combination(v) or not _valid_delta_r_combination(v):
            _combinations.pop(k)
            continue
        _combinations[k]["f"] = v["lep"][0].flavour  # "combination flavour" for simpler removal later
        _combinations[k]["z"] = np.sum(v["lep"])  # actual z boson of this combination
    
    # helper 2
    _valid_z_pair_masses = lambda _z1, _z2: (120 > _z1.mass > _z2.mass > 12) & (_z1.mass > 40)
    _pt_dict = pt_dict if pt_dict else {1: 0, 2: 0, 3: 0, 4: 0}  # for example {1: 20, 2: 10, 3: 5, 4: 1}
    _valid_z_pair_pt = lambda x: all([np.sum(x > _pt_dict[_key]) >= _key for _key in _pt_dict.keys()])
    _valid_flavour_comp = lambda ch, _d: len(np.unique([_v["f"] for _v in _d.values()])) == 2 if ch == "2e2mu" else True
    
    try:
        while len(_combinations.keys()) >= 2 and _valid_flavour_comp(row[1], _combinations):  # here
            _tmp_combinations = _combinations.copy()
            _z1_idx = min(_tmp_combinations, key=lambda x: abs(_tmp_combinations[x]["z"].mass - z_mass))  # nearest to z_mass
            for k, v in list(_tmp_combinations.items()):  # removing combinations with same leptons or flavour (*)
                try:
                    if row[1] == "2e2mu":  # here
                        _tmp_combinations.pop(k) if _tmp_combinations[_z1_idx]["f"] == v["f"] else None  # keep only other "combination" flavour
                    else:
                        _tmp_combinations.pop(k) if any(i == j for i in _z1_idx for j in k) else None  # remove combination with _z1_idx leptons
                except KeyError:
                    pass
            try:
                _z2_idx = max(_tmp_combinations, key=lambda x: _tmp_combinations[x]["z"].mass)  # maximal remaining
            except ValueError:  # occurs frm (*) {(1, 2), (2, 3)} -> {}
                break
            z1, z2 = _combinations[_z1_idx]["z"], _combinations[_z2_idx]["z"]
            z1, z2 = (z1, z2) if z1.mass > z2.mass else (z2, z1)
            _pt = np.concatenate([np.array([_combinations[k]["lep"][i].pt for i in range(2)]) for k in [_z1_idx, _z2_idx]])
            if _valid_z_pair_pt(_pt) and _valid_z_pair_masses(z1, z2):
                row[3] = z1  # here
                row[4] = z2  # here
                return row
            else:
                _combinations.pop(_z1_idx)  # retry without this one
    except ValueError:
        pass
    row[0] = np.nan
    return row
