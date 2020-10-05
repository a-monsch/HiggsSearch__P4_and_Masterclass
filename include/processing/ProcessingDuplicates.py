# -*- coding: UTF-8 -*-
import sys

import numpy as np
import pandas as pd


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray


class ProcessDuplicates(object):
    """
    Class to check if there are possible duplicates of any kind in
    the pd.DataFrames created from the same source files.
    """
    
    def __init__(self, input_1, input_2, name_1=None, name_2=None):
        self.data_1 = input_1
        self.data_2 = input_2
        self.name_1 = name_1
        self.name_2 = name_2
        self.duplicates_found = False
    
    @classmethod
    def from_file(cls, input_1, input_2):
        """
        Load the two pd.DataFrames from two files.

        :param input_1: str
        :param input_2: str
        """
        df1_ = pd.read_csv(input_1, sep=";")
        df2_ = pd.read_csv(input_2, sep=";")
        return cls(input_1=df1_, input_2=df2_, name_1=input_1, name_2=input_2)
    
    def remove_by_best_z1(self):
        """
        Removes double appearing results. The removal is performed on the basis of the z boson mass.
        """
        z_mass = 91.1876
        
        df1_ = self.data_1.run(["run", "event", "z1_mass", "z2_mass"], axis=1)
        df1_["frame_id"] = 1
        df1_["z1_diff"] = np.abs(df1_["z1_mass"] - z_mass)
        
        df2_ = self.data_2.run(["run", "event", "z1_mass", "z2_mass"], axis=1)
        df2_["frame_id"] = 2
        df2_["z1_diff"] = np.abs(df2_["z1_mass"] - z_mass)
        
        dg_ = pd.concat([df1_, df2_])
        
        try:
            dg_dup_ = [g for _, g in dg_.groupby(["run", "event"]) if len(g) > 1]
            df_reduced = pd.concat(frame.sort_values(["z1_diff"]).iloc[:2] for frame in dg_dup_)
            
            d1_reduced = df_reduced[df_reduced["frame_id"] == 1].drop(["frame_id", "z1_diff"], axis=1)
            d2_reduced = df_reduced[df_reduced["frame_id"] == 2].drop(["frame_id", "z1_diff"], axis=1)
            
            print(str(d1_reduced.shape[0]) + " duplicates found!")
            self.duplicates_found = True
            
            keys_1_ = list(d1_reduced.columns.values)
            keys_2_ = list(d2_reduced.columns.values)
            
            index_all_1 = self.data_1.set_index(keys_1_).index
            index_all_2 = self.data_2.set_index(keys_2_).index
            
            index_drop_1 = d1_reduced.set_index(keys_1_).index
            index_drop_2 = d2_reduced.set_index(keys_2_).index
            
            self.data_1 = self.data_1[~index_all_1.isin(index_drop_1)]
            self.data_2 = self.data_2[~index_all_2.isin(index_drop_2)]
        
        except ValueError:
            print("No duplicates found!")
    
    def test_equality(self, print_short_error=True):
        """
        Tests if the two pd.DataFrames are equal.

        :param print_short_error: bool
        """
        self.data_1 = self.data_1.astype({"run": "int64", "event": "int64"})
        self.data_2 = self.data_2.astype({"run": "int64", "event": "int64"})
        if print_short_error:
            try:
                pd.testing.assert_frame_equal(self.data_1, self.data_2)
            except AssertionError:
                print("difference found")
        if not print_short_error:
            pd.testing.assert_frame_equal(self.data_1, self.data_2)
    
    def save_to_csv(self, name_1=None, name_2=None):
        """
        Saves the two pd.DataFrames if duplicates were found.

        :param name_1: str
        :param name_2: str
        """
        if self.duplicates_found:
            save_name_1, save_name_2 = name_1, name_2
            if save_name_1 is None and save_name_2 is None:
                save_name_1, save_name_2 = self.name_1, self.name_2
            if save_name_1 is None and save_name_2 is None:
                sys.exit("Give some names!")
            
            self.data_1.to_csv(save_name_1, sep=";", index=False)
            self.data_2.to_csv(save_name_2, sep=";", index=False)
        else:
            print("No need, because no duplicates were found")
