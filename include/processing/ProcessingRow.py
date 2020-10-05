# -*- coding: UTF-8 -*-
import ast

import numpy as np


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray


class ProcessingRow(object):
    """
    Class for basic manipulation of a "pandas Series" which contains only strings.
    """
    
    def __init__(self, row, name_list):
        self.row = row
        self.names = name_list
        self.pairs = list(zip(name_list, row))
        self.ignore = ["run", "event", "luminosity", "z1_index", "z2_index", "z1_mass", "z2_mass", "z1_tag", "z2_tag"]
    
    @staticmethod
    def __to_str(variable):
        """
        Converts an ndarray to a string without spaces.

        :param variable: ndarray
                         1D array containing data with "float", "str" or "int" type.
        :return: str
        """
        try:
            return ",".join([str(item) for item in variable]).replace(" ", "")
        except TypeError:
            return str(variable)
    
    def __to_str_from_str(self, variable, accept):
        """
        Converts str back to str after applying a boolean run.

        :param variable: str
        :param accept: ndarray
                       1D array containing data with "bool" type.
        :return: str
        """
        try:
            return self.__to_str(np.array(ast.literal_eval(f"[{variable}]"))[accept])
        except:
            return self.__to_str(np.array(variable.split(","))[accept])
    
    def if_column(self):
        """
        Skipt a row if this is the head.
        :return: bool
        """
        return True if self.pairs[0][0] == self.pairs[0][1] else False
    
    def add_raw_to_row(self, variable_name, variable_array):
        """
        Adds an additional size to the pandas series if not available and updates the names of this pandas series.

        :param variable_name: str
        :param variable_array: ndarray
                               1D array containing data with "float", "str" or "int" type.
        """
        if variable_name not in self.names:
            self.row[variable_name] = self.__to_str(variable_array)
            self.names = list(self.row.to_frame().T)
    
    def search_for(self, search_variables, type_variables):
        """
        Searches for the entries in the "pandas series".

        :param search_variables: list
                                 1D list containing names with "str" type.
        :param type_variables: list
                               !d list containing datatypes
        :return: ndarray
                 search_variables x D array containing data with "ndarray" type.
        """
        found_array = []
        for i, var in enumerate(search_variables):
            for pair in self.pairs:
                if var in pair[0]:
                    found_array.append(np.array(pair[1].split(","), dtype=type_variables[i]))
        return found_array
    
    def reduce_row(self, accept_array):
        """
        Kicks all leptons with the corresponding sizes out of the series using the boolean run.

        :param accept_array: ndarray
                             1D or 2D array containing data with "bool" type.
        """
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
        """
        Evaluates using the "True" entries of the boolean run if the minimum
        number of leptons in the event is met and replace "run" with a "np.nan" if not.

        :param accept_array: ndarray
                             1D or 2D array containing data with "int" type.
        :return: bool
        """
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
        """
        Evaluates and reduces on the basis of a boolean run in form of a list
        or a single value whether the "pandas series" should be discarded or only reduced.

        :param to_accept_list: nrray
                               1D or 2D array containing data with "bool" type.
        :param to_accept_bool: bool
        """
        if to_accept_bool is not None:
            if to_accept_bool:
                pass
            else:
                self.row["run"] = np.nan
        if to_accept_list is not None:
            if not self.eval_on_length(to_accept_list):
                self.reduce_row(to_accept_list)
