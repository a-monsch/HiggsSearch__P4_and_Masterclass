# -*- coding: UTF-8 -*-
import math
import warnings

import numpy as np
import scipy.stats as scst

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class HistHelper(object):
    """
    Class with additional functions that can be used for histograming and delete_all related functions.
    """
    
    @staticmethod
    def convert_column(col_, lepton_number=None, filter_by=None, **kwargs):
        """
        Converts a pandas array containing str, which in turn contain the necessary single lepton variables, into np.ndarrays.
        For the filters "mass_4l" or "z1_mass" or "z2_mass" are available.
        All rows that do not match the filter_by will be removed.
        Afterwards the str array is converted into float arrays.

        :param col_: pd.array
                     1D array containing data with "float" or "int" type.
        :param lepton_number: int or list containing the desired leptons
        :param filter_by: list
                        ["column_name", (lower_value_limit, upper_value_limit)]
        :return: ndarray
                 1D array containing data with "float" type.
        """
        pass_array = None
        if filter_by is not None:
            filter_array = (col_[filter_by[0]].values > filter_by[1][0]) & (col_[filter_by[0]].values < filter_by[1][1])
            col_ = col_.drop([filter_by[0]], axis=1)
            pass_array = col_.values
            pass_array = pass_array[filter_array]
        
        if filter_by is None:
            pass_array = col_.values
        
        _shape = pass_array.shape
        
        try:
            pass_array = np.concatenate(pass_array)
        except ValueError:
            pass
        
        try:
            if type(pass_array[0]) == str:
                if lepton_number is not None:
                    
                    _cut = [lepton_number] if isinstance(lepton_number, int) else (lepton_number if isinstance(lepton_number, list) else None)
                    _pass_array = []
                    
                    for item in pass_array:
                        _tmp_full, _tmp_part = np.array(item.split(","), dtype=float), []
                        
                        for n in _cut:
                            try:
                                _tmp_part.append(_tmp_full[n])
                            except IndexError:
                                continue
                        _pass_array.append(np.array(_tmp_part))
                    pass_array = np.array(_pass_array)
                    
                    try:
                        pass_array = np.concatenate(pass_array)
                    except:
                        pass
                else:
                    pass_array = np.concatenate([np.array(item.split(","), dtype=float) for item in pass_array])
        except IndexError:
            return np.array([])
        
        return pass_array
    
    @staticmethod
    def calc_errors_sqrt(array_of_interest):
        """
        Calculates the symmetric uncertainty on the value in a poisson distribution.
        For large numbers the approximation is sufficient.

        :param array_of_interest: ndarray
                                  1D array containing data with "float" type.
        :return: ndarray
                 1D array containing data with "float" type.
        """
        return np.sqrt(np.array(array_of_interest))
    
    @staticmethod
    def calc_errors_poisson(array_of_interest, interval=0.68):
        """
        Calculates the lower and upper uncertainty from the non-continuous poisson distribution.

        :param array_of_interest: ndarray
                                  1D array containing data with "float" type.
        :param interval: float
        :return: ndarray
                 2D array containing data with "float" type.
        """
        lower_cap = (1 - interval) / 2.
        higher_cap = (1 - interval) / 2. + interval
        
        def get_lower_higher_value(x):
            return scst.poisson.ppf(lower_cap, x), scst.poisson.ppf(higher_cap, x)
        
        lower_limit = np.array([get_lower_higher_value(value)[0] for value in array_of_interest])
        higher_limit = np.array([get_lower_higher_value(value)[1] for value in array_of_interest])
        return [lower_limit, higher_limit]
    
    @staticmethod
    def calc_errors_poisson_near_cont(array_of_interest, intervall=0.68):
        """
        Calculates approximately the lower and upper uncertainties for a "continuous" poisson distribution.

        :param array_of_interest: ndarray
                                  1D array containing data with "float" type.
        :param intervall: float
        :return: ndarray
                 2D array containing data with "float" type.
        """
        used_lower = {0: 0,
                      1: 0.42421579360961914, 2: 0.7561395168304443, 3: 1.0175738334655762, 4: 1.24031600356102,
                      5: 1.437682181596756, 6: 1.6401071548461914, 7: 1.828809380531311, 8: 1.9980171918869019,
                      9: 2.1531015038490295, 10: 2.2973418831825256, 11: 2.4329039454460144, 12: 2.573564440011978,
                      13: 2.71704238653183, 14: 2.8509205877780914, 15: 2.9769710302352905, 16: 3.0964967012405396,
                      17: 3.2104830741882324, 18: 3.3196941614151, 19: 3.4247340261936188, 20: 3.5306936502456665,
                      21: 3.6448516845703125, 22: 3.753896266222, 23: 3.858468145132065, 24: 3.959094285964966,
                      25: 4.056209862232208, 26: 4.150178790092468, 27: 4.241308212280273, 28: 4.3298600018024445,
                      29: 4.416058778762817, 30: 4.500114470720291}
        
        used_upper = {0: 0,
                      1: 2.0667134523391724, 2: 2.405105322599411, 3: 2.745358496904373, 4: 3.0266018211841583,
                      5: 3.245048940181732, 6: 3.4299083948135376, 7: 3.6338911056518555, 8: 3.8340370655059814,
                      9: 4.008370667695999, 10: 4.1644439697265625, 11: 4.306962668895721, 12: 4.439002811908722,
                      13: 4.581984728574753, 14: 4.730859637260437, 15: 4.868239730596542, 16: 4.996275305747986,
                      17: 5.1165759563446045, 18: 5.230372965335846, 19: 5.338625460863113, 20: 5.442093729972839,
                      21: 5.551243782043457, 22: 5.66814911365509, 23: 5.779230922460556, 24: 5.885223418474197,
                      25: 5.9867331981658936, 26: 6.084259033203125, 27: 6.17824923992157, 28: 6.26904222369194,
                      29: 6.356954097747803, 30: 6.442250490188599}
        
        @np.vectorize
        def upper_lower_l(mu, intervall_=intervall, eps_=1e-8):
            @np.vectorize
            def __pp_f(x):
                return scst.poisson.pmf(math.floor(x), mu=mu)
            
            @np.vectorize
            def __pp_c(x):
                return scst.poisson.pmf(np.ceil(x), mu=mu)
            
            upper_limit, lower_limit = 0.5 + intervall_ / 2., 0.5 - intervall_ / 2.
            pass_x_upper, pass_x_lower = mu, mu
            
            if float(mu) == 0.0:
                used_upper[mu] = 0
                used_lower[mu] = 0
            
            if mu not in used_upper.keys():
                start_block_, a_ = 0.0, 0.0
                
                while a_ + __pp_f(start_block_) <= upper_limit:
                    a_ += __pp_f(start_block_)
                    start_block_ += 1
                
                x_, dx_, = start_block_, 0.5
                while dx_ > eps_:
                    if a_ + __pp_f(x_) * dx_ <= upper_limit:
                        a_ += __pp_f(x_) * dx_
                        pass_x_upper = x_ - mu + 0.5
                        x_ += dx_
                    else:
                        dx_ *= 0.5
                used_upper[mu] = pass_x_upper
            
            if mu not in used_lower.keys():
                start_block_, a_ = mu, 0.5
                while a_ - __pp_c(start_block_) > lower_limit:
                    a_ -= __pp_c(start_block_)
                    start_block_ -= 1
                x_, dx_ = start_block_, 0.5
                
                while dx_ > eps_:
                    if a_ - __pp_c(start_block_) * dx_ >= lower_limit:
                        a_ -= __pp_c(start_block_) * dx_
                        pass_x_lower = mu - x_ - 0.5
                        x_ -= dx_
                    else:
                        dx_ *= 0.5
                
                used_lower[mu] = pass_x_lower
            
            return [used_upper[mu], used_lower[mu]]
        
        upper_limit_pass = np.array([upper_lower_l(num)[0] for num in array_of_interest])
        lower_limit_pass = np.array([upper_lower_l(num)[1] for num in array_of_interest])
        
        return [lower_limit_pass, upper_limit_pass]
    
    @staticmethod
    def calc_errors_alternative_near_simplified(array_of_interest):
        """
        Alternative calculation of asymetric errorbars - simplified self.calc_errors_poisson_near_cont

        :param array_of_interest: ndarray
                                  1D array containing data with "float" type.
        :return: ndarray
                 2D array containing data with "float" type.
        """
        
        def get_lower_higher_value(x):
            return -0.5 + np.sqrt(x + 0.025), 0.5 + np.sqrt(x + 0.25)
        
        lower_limit = np.array([get_lower_higher_value(value)[0] for value in array_of_interest])
        higher_limit = np.array([get_lower_higher_value(value)[1] for value in array_of_interest])
        return [lower_limit, higher_limit]
