import itertools as itt
from copy import deepcopy
from typing import Union, List, Callable

import numpy as np
import pandas as pd
from threaded_buffered_pipeline import buffered_pipeline
from tqdm import tqdm
from uproot3_methods.classes.TLorentzVector import TLorentzVector

from include.Particle import Lepton

tqdm.pandas()


def pipeline(df: pd.DataFrame,  # Current DataFrame on witch the pipeline is build
             func_list: List[Callable[[pd.Series], pd.Series]],  # (filter) functions for the pipeline
             buffer_size: Union[int, List[int]] = 1):  # buffersize of all steps: [int] or specific steps List[int]
    
    _iteritems = df.iterrows()  # Iterate over DataFrame rows as (index, Series) pairs.
    
    _buffered_iteritems = buffered_pipeline()  # initializing a buffered version...
    buffered_iter = _buffered_iteritems(_iteritems, buffer_size)  # ...wrapping existing version
    
    def _gen_like_func(_func, _iteritems):  # passing trough the pipeline
        for i, item in _iteritems:
            yield (i, _func(item))
    
    # building pipeline: ...d(c(b(a(x)))), with a, b, c, d... buffered pipeline steps
    for func, buffer in zip(func_list, buffer_size if isinstance(buffer_size, list) else itt.repeat(buffer_size)):
        buffered_iter = _buffered_iteritems(_gen_like_func(func, buffered_iter), buffer)
    
    return buffered_iter


def load_dataset(file: str,  # path to dataset
                 nrows: int = None):  # reading first nrows of the dataset
    
    print(f"Loading {file}...")
    
    if ".csv" in file:
        
        # in case when a fixed length is needed: place this or np.nan
        _dummy = Lepton(*(TLorentzVector(*(np.nan for _ in range(4))), *(np.nan for _ in range(6))))
        
        def _converter_func(_item):  # converting string to corresponding element
            array, nan = np.array, np.nan  # since print(np.array([1])) -> array(1) and print(np.nan) -> nan
            try:
                return eval(_item)
            except SyntaxError:  # for example in channel column
                return eval(f'"{_item}"')
        
        _converters = {k: _converter_func for k in list(pd.read_csv(file, sep=";", nrows=10).columns)}
        df = pd.read_csv(file, sep=";", converters=_converters, nrows=nrows)
        return df
    else:
        raise NameError("Files should have .csv format...")  # update if other formats are available


def save_dataset(df: pd.DataFrame,
                 file: str):
    if ".csv" in file:
        def make_array_leptons(row):
            # keep the list of leptons as np.ndarray in case it is changed (by __repr__ or __str__)
            row.leptons = f"array({row.leptons})" if "array" not in str(row.leptons) else row.leptons
            return row
        
        _df = deepcopy(df)
        
        _df.leptons = _df.leptons.to_numpy()  # if it were lists before
        _df.leptons = _df.leptons.astype(str)  # converting to string representation
        _df.leptons = _df.leptons.str.replace("\n", ",")  # print(np.array) place "\n" for better looking
        
        _df = _df.progress_apply(lambda x: make_array_leptons(x), axis=1)
        
        _df.to_csv(file, sep=";", index=None)
        print(f"{file} saved.")


def mc_hist_scale_factor(channel: str,  # "4mu", "4e", "2e2mu"
                         process: str):  # "background" or "signal"
    
    _cross_section = {"4mu": 76.91, "4e": 76.91, "2e2mu": 176.7, "H": 6.5}
    _k = {"4mu": 1.386, "4e": 1.386, "2e2mu": 1.386, "H": 1.0}
    _N_mc = {"4mu": 1499064, "4e": 1499093, "2e2mu": 1497445, "H": 299973}
    _luminosity = {"B": np.longdouble(4.429375295985512733), "C": np.longdouble(7.152728016920716286)}
    
    _key = channel if process == "background" else "H"
    
    return (_cross_section[_key] * _k[_key] * sum(_luminosity.values())) / _N_mc[_key]


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
