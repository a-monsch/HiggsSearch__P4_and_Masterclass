import math
import os
from collections import OrderedDict
from functools import partial, update_wrapper
from itertools import cycle

import kafe2 as K2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as sci
from iminuit import Minuit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from tqdm import tqdm

from .PlotHelper import PlotHelper
from ..fits.McFit import McFit
from ..histogramm.Hist import Hist
from ..histogramm.HistDataGetter import HistDataGetter as HDG


class StatTest(Hist):
    """
    Class that implements and visualizes the local significance
    once as p0 scan and once as profiled likelihood scan.
    """
    
    def __init__(self,
                 bins=15, hist_range=(106, 151), signal_func=lambda x, *a: x, background_func=lambda x, *a: x,
                 save_dir="./stat_tests",
                 info=None,
                 ru_dir="./data/ru_aftH", mc_dir="./data/mc_aftH",
                 verbose=True,
                 sigma_parameter=None
                 ):
        super(StatTest, self).__init__(bins=bins, hist_range=hist_range)
        self.sfu = signal_func
        self.bfu = background_func
        self.info = info
        self.ru_dir = ru_dir
        self.mc_dir = mc_dir
        
        self.verbose = verbose
        
        self.func_func_param = {}
        self.func_names = {}
        self.iminuit_objects = {}
        self.significance_dict = {}
        
        self.save_dir = save_dir
        self.set_save_dir()
        self.set_func_val(init=True)
        
        self.calculated_dump = {}
        
        self.sigma_parameter = sigma_parameter
        self.set_sigma_func_param()
        self.data_raw = self.__set_data_raw()
        self.temp_calc_ = {}
        self.data_com = None
    
    # set part - 1 - manually or automatically, preparation
    
    def set_quad_defaults(self, item=None):
        """
        Sets default values for the scipy.integrate.quad integration function
        
        :param item: dict
        """
        dict_ = item if item is not None else {"points": np.linspace(self.hist_range[0], self.hist_range[1], 50),
                                               "epsabs": 1e-12,
                                               "epsrel": 1e-12}
        
        f3_ = partial(sci.quad, **dict_)
        update_wrapper(f3_, sci.quad)
        sci.quad = f3_
    
    def __set_data_com(self):
        """
        Creates a Hist object from HistDataGetter.
        Contains the already scaled histogram with all channels.
        
        Note: Might be unnecessary if using unbinned.
              On the other hand it is necessary for the binned variant
        """
        d = HDG(bins=self.bins, hist_range=self.hist_range, info=self.info, mc_dir=self.mc_dir, ru_dir=self.ru_dir)
        return d.get_mc_com()
    
    def __set_data_raw(self):
        """
        Get raw_date as Hist object from HistDataGetter.

        Note: Used for calculation of uncertainties
        """
        d = HDG(bins=self.bins, hist_range=self.hist_range, info=self.info, mc_dir=self.mc_dir, ru_dir=self.ru_dir)
        return d.get_data_raw(column_name="mass_4l")
    
    def set_info(self, item=None):
        """
        Set info for MC SImulation scaling
        
        :param item: list
                     [["year"], ["run"]]
        """
        self.info = [["2012"], ["A-D"]] if item is None or self.info is None else item
    
    def set_func_val(self, func=None, func_param=None, init=False):
        """
        Sets user default or already certain parameter values for the used distributions.
        
        :param func: function
        :param func_param: list
        :param init: bool
        """
        if func is not None and func_param is not None:
            self.func_func_param[func.__name__] = list(func_param)
        if init:
            self.func_func_param["legendre_grade_2"] = np.array(
                #             [0.02428954322341774, 0.00020021879360914812, -8.663487231700586e-06]
                [0.02425287729893847, 0.00020095053916274993, -8.51985432282704e-06]
            )
            
            self.func_func_param["gauss"] = np.array(
                #             [1.9727411383681588, 124.81209770469063]
                [1.9755701776484866, 124.83155449997581]
            )
            self.func_func_param["DSCB"] = np.array(
                [1.3388805157158652, 124.98071139688055, 0.9278644003292827, 1.1053232991255295, 3.2720915177175414,
                 6.771701327613522])
    
    def set_save_dir(self):
        """
        Creates the folder for saving.
        """
        if self.save_dir is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
        if self.save_dir is None:
            if not os.path.exists(""):
                os.makedirs("", exist_ok=True)
            self.save_dir = ""
    
    def set_sigma_func_param(self, item=None):
        """
        set the parameters for the width estimation using a linear
        model centered around the mean value of the x axis.
        
        :param item: tuple
                     (a, b, mean)
        """
        if self.sigma_parameter is None:
            self.sigma_parameter = (2.079854189377624, 0.013654169521349959, 130.0564929849343)
        if item is not None:
            self.sigma_parameter = item
    
    # set part - 2 - for procedure
    
    def __create_gauss_sigma(self, mass_array_):
        """
        creates an array that calculates the gaussian widths
        according to the selected masses using the linear model.
        
        :param mass_array_: ndarray
                            1D array containing data with "float" type.
        :return: ndarray
                 1D array containing data with "float" type.
        """
        
        def used_func(x):
            a, b, mean = self.sigma_parameter
            return a + (x - mean) * b
        
        return used_func(mass_array_)
    
    def __create_mh_for_p0(self):
        """
        Creates a mass array for the p0 estimation.
        :return: ndarray
                 1D array containing data with "float" type.
        """
        arr_ = np.array([])
        arr_ = np.append(arr_, np.arange(max(self.hist_range[0], 106), 124.5, 0.25))  # 0.25))
        arr_ = np.append(arr_, np.arange(124.75, 126.0, 0.25))  # 0.25))
        arr_ = np.append(arr_, np.arange(126.5, min(self.hist_range[1], 151), 0.5))  # 0.25))
        return arr_
    
    def get_func_from_fit(self, tag="background", to_chi2_one=True):
        """
        Executes a fit and determines the parameters for the functions.
        Optional, if the function parameters are not explicitly set beforehand.
        
        :param tag: str
                    "background" or "signal"
        :param to_chi2_one: bool
        """
        temp_ = McFit(bins=self.bins, hist_range=self.hist_range,
                      tag=tag, to_chi2_one=to_chi2_one,
                      mc_dir=self.mc_dir, ru_dir=self.ru_dir, verbose=True,
                      info=self.info, save_dir=self.save_dir)
        if tag == "background":
            val_, err_ = temp_.get_results(self.bfu)
            self.func_func_param[self.bfu.__name__] = list(val_)
        if tag == "signal":
            val_, err_ = temp_.get_results(self.sfu)
            self.func_func_param[self.sfu.__name__] = list(val_)
        temp_.save_report_to_yaml()
    
    #
    
    # calc alpha / alpha - mass scan and significance
    
    def __get_nll_for_alpha_mass_scan(self, variable_=("alpha",)):
        """
        Creates an extended likelihood function, which can be either one-dimensional or two-dimensional.
        
        :param variable_: tuple
                          tuple of strings. ("alpha",) or ("alpha", "mass")
        :return: function
        """
        data_ = self.data_raw.data
        a, b = self.hist_range
        self.set_quad_defaults()
        b_args = self.func_func_param[self.bfu.__name__]
        b_norm = (sci.quad(self.bfu, a, b, args=tuple(b_args))[0]) ** (-1)
        
        ca_ = eval(f'sys._getframe({2}).f_code.co_name')
        self.temp_calc_[ca_] = {"bac": {}, "sig": {}, "exp": {}}
        
        @np.vectorize
        def used_func(x_, alpha_, mass_):
            if x_ not in self.temp_calc_[ca_]["bac"].keys():
                self.temp_calc_[ca_]["bac"][x_] = b_norm * self.bfu(x_, *b_args)
            if (x_, mass_) not in self.temp_calc_[ca_]["sig"].keys():
                temp_s_args = self.func_func_param[self.sfu.__name__]
                s_args = (temp_s_args[0], mass_, *temp_s_args[2:])
                s_norm = (sci.quad(self.sfu, a, b, args=tuple(s_args))[0]) ** (-1)
                self.temp_calc_[ca_]["sig"][(x_, mass_)] = s_norm * self.sfu(x_, *s_args)
            return alpha_ * self.temp_calc_[ca_]["sig"][(x_, mass_)] + (1 - alpha_) * self.temp_calc_[ca_]["bac"][x_]
        
        @np.vectorize
        def exp_val(alpha_, mass_):
            if (alpha_, mass_) not in self.temp_calc_[ca_]["exp"].keys():
                self.temp_calc_[ca_]["exp"][(alpha_, mass_)] = sci.quad(lambda x: x * used_func(x, alpha_, mass_), a, b)[0]
            return self.temp_calc_[ca_]["exp"][(alpha_, mass_)]
        
        def nll_2d(alpha=0.38, mass=125.4):
            ln = - exp_val(alpha, mass) + np.sum(np.log(exp_val(alpha, mass) * used_func(data_, alpha, mass)), axis=0)
            return - 2 * ln
        
        if len(variable_) == 2:
            return nll_2d
        if len(variable_) == 1:
            fixed_mass = self.func_func_param[self.sfu.__name__][1]
            
            def nll_1d(alpha):
                return nll_2d(alpha=alpha, mass=fixed_mass)
            
            return nll_1d
    
    def calc_1d_unbinned_iminuit(self):
        """
        Calculates the optimal parameters for the one-dimensional profiled likelihood using Iminuit.
        """
        nll = self.__get_nll_for_alpha_mass_scan(variable_=("alpha",))
        
        # noinspection PyArgumentList
        m_ = Minuit(nll, limit_alpha=(0, 1.0))
        m_.migrad()
        m_.minos()
        
        self.iminuit_objects[f"1D_unbinned__{self.sfu.__name__}_{self.bfu.__name__}"] = m_, nll
        
        if self.verbose:
            print(m_.get_param_states())
            print(np.sqrt(nll(0) - nll(m_.np_values()[0])))
    
    def calc_2d_unbinned_iminuit(self):
        """
        Calculates the optimal parameters for the two-dimensional profiled likelihood using Iminuit.
        """
        nll = self.__get_nll_for_alpha_mass_scan(variable_=("alpha", "mass"))
        
        # noinspection PyArgumentList
        m_ = Minuit(nll, alpha=0.3740, mass=125.0, limit_alpha=(0.0, 1.0), limit_mass=(123.0, 127.0))
        m_.migrad()
        m_.minos()
        
        self.iminuit_objects[f"2D_unbinned__{self.sfu.__name__}_{self.bfu.__name__}"] = m_, nll
        
        if self.verbose:
            print(m_.get_param_states())
            print(np.sqrt(nll(0.0, m_.np_values()[1]) - nll(m_.np_values()[0], m_.np_values()[1])))
    
    # alpha / alpha - mass scan plots

    def plot_1d_scan_unbinned(self, to_file=True, nll_func=None, iminuit_object=None):
        """
        Plot the result of the one-dimensional profiled likelihood scan.

        :param to_file: bool
        """
        try:
            mt_, nll_func = self.iminuit_objects[f"1D_unbinned__{self.sfu.__name__}_{self.bfu.__name__}"]
        except KeyError:
            try:
                self.calc_1d_unbinned_iminuit()
                mt_, nll_func = self.iminuit_objects[f"1D_unbinned__{self.sfu.__name__}_{self.bfu.__name__}"]
            except KeyError:
                mt_, nll_func = iminuit_object, nll_func
                if mt_ is None or nll_func is None:
                    raise NotImplementedError
        
        @np.vectorize
        def nll_vec(alpha_):
            return nll_func(alpha_)
        
        x_ins_limit = PlotHelper.get_error_points_from_iminuit_obj(mt_, "alpha")
        alpha = np.sort(np.unique(np.append(np.array(x_ins_limit), np.linspace(0.0, 0.9, 100))))
        nll = nll_vec(alpha)
        nll = nll - np.amin(nll)
        
        z_ = np.sqrt(nll_func(0.0) - nll_func(mt_.np_values()[0]))
        fig, ax = plt.subplots()
        PlotHelper.plot_nll_scan_main_window(ax_obj_=ax, x_=alpha, y_=nll, z_=z_)
        
        ax_in = inset_axes(ax, width='50%', height='40%', loc="upper right")
        PlotHelper.plot_nll_scan_inside_window(ax_obj_=ax_in, x_=alpha, y_=nll, x_ticks_=x_ins_limit)
        mark_inset(ax, ax_in, loc1=2, loc2=4, fc="none", ec="grey", lw=0.5, alpha=0.5)
        
        file_ = os.path.join(self.save_dir, f"1D_scan_unbinned_{self.sfu.__name__}_{self.bfu.__name__}.png")
        plt.savefig(file_) if to_file else plt.show()

    def plot_2d_scan_unbinned(self, to_file=True, nll_func=None, iminuit_object=None):
        """
        Plot the result of the one-dimensional profiled likelihood scan.

        :param to_file: bool
        """
        try:
            mt_, nll_func = self.iminuit_objects[f"2D_unbinned__{self.sfu.__name__}_{self.bfu.__name__}"]
        except KeyError:
            try:
                self.calc_1d_unbinned_iminuit()
                mt_, nll_func = self.iminuit_objects[f"2D_unbinned__{self.sfu.__name__}_{self.bfu.__name__}"]
            except KeyError:
                mt_, nll_func = iminuit_object, nll_func
                if mt_ is None or nll_func is None:
                    raise NotImplementedError
        
        @np.vectorize
        def nll_vec(alpha, mass):
            return nll_func(alpha, mass)
        
        p_alpha_ = PlotHelper.get_error_points_from_iminuit_obj(mt_, "alpha")
        p_mass_ = PlotHelper.get_error_points_from_iminuit_obj(mt_, "mass")
        
        if self.verbose: print("Start plotting: Initialize", flush=True, end="\r")
        alpha_ = np.sort(np.unique(np.append(np.array(p_alpha_), np.linspace(0.0, 0.8, 50))))
        nll_alpha = nll_vec(alpha_, mt_.np_values()[1])
        nll_alpha = nll_alpha - np.amin(nll_alpha)
        if self.verbose: print("Plotting: Got alpha main window", flush=True, end="\r")
        mass_ = np.sort(np.unique(np.append(np.array(p_mass_), np.linspace(p_mass_[0] - 0.25, p_mass_[-1] + 0.25, 50))))
        nll_mass = nll_vec(mt_.np_values()[0], mass_)
        nll_mass = nll_mass - np.amin(nll_mass)
        if self.verbose: print("Plotting: Got mass main window", flush=True, end="\r")
        
        z_ = np.sqrt(nll_func(0.0, mt_.np_values()[1]) - nll_func(mt_.np_values()[0], mt_.np_values()[1]))
        
        f = plt.figure(figsize=(14, 8))
        f.subplots_adjust(left=0.15, bottom=0.07, right=0.98, top=0.95, wspace=0.23, hspace=1.0)
        gs = gridspec.GridSpec(1, 1, figure=f)
        
        gs0 = gridspec.GridSpecFromSubplotSpec(7, 2, subplot_spec=gs[0])
        ax1 = f.add_subplot(gs0[:-3, :])
        PlotHelper.plot_nll_scan_main_window(ax_obj_=ax1, x_=alpha_, y_=nll_alpha, z_=z_)
        
        ax_in = inset_axes(ax1, width='50%', height='40%', loc="upper right")
        PlotHelper.plot_nll_scan_inside_window(ax_obj_=ax_in, x_=alpha_, y_=nll_alpha, x_ticks_=p_alpha_)
        mark_inset(ax1, ax_in, loc1=2, loc2=4, fc="none", ec="grey", lw=0.5, alpha=0.5)
        
        if self.verbose: print("Plotting: Starting contour", flush=True, end="\r")
        ax2 = f.add_subplot(gs0[4:7, :-1])
        PlotHelper.contour_own(ax_obj_=ax2, nll_obj_=nll_vec, m_obj_=mt_, var1_="alpha", var2_="mass")
        if self.verbose: print("Plotting: End Contour", flush=True, end="\r")
        
        ax3 = f.add_subplot(gs0[4:7, -1])
        PlotHelper.plot_nll_scan_inside_window(ax_obj_=ax3, x_=mass_, y_=nll_mass,
                                               x_ticks_=p_mass_,
                                               label_=r"$- 2 \ln \left( \frac{\mathcal{L}}{\mathcal{L}_{\rm{min}}} \right)-\rm{profile}$",
                                               x_label=r"$m_{\rm{H}}$",
                                               y_label=r"$- 2 \ln \left( \frac{\mathcal{L}}{\mathcal{L}_{\rm{min}}} \right)$")
        
        file_ = os.path.join(self.save_dir, f"2D_scan_unbinned_{self.sfu.__name__}_{self.bfu.__name__}.png")
        plt.savefig(file_) if to_file else plt.show()
    
    #
    
    # calc p0
    
    def __get_p0_q0_array(self, h1_calc, h1_calc_iminuit, h0_calc, to_file=True, name=""):
        """
        Loop element which calculates the p0 value for different masses and gaussian widths.
        The Iminuit variant of the function must not be np.vectorized, or it must contain
        an explicit function signature. All other functions must be vectorized.
        The minimizer used is iminuit.
        
        :param h1_calc:  np.vectorized function
        :param h1_calc_iminuit: function
        :param h0_calc: np.vectorized function
        :param to_file: bool
        :param name: str
        :return: pd.DataFrame
                 containing mass, sigma, q0 and p0
        """
        mass_array = self.__create_mh_for_p0()
        sigma_array = self.__create_gauss_sigma(mass_array)
        q0_ = np.zeros(len(mass_array), dtype=np.longdouble)
        p0_ = np.zeros(len(mass_array), dtype=np.longdouble)
        
        for i, (m, s) in tqdm(enumerate(zip(mass_array, sigma_array)), total=len(mass_array)):
            # noinspection PyArgumentList
            m_ = Minuit(h1_calc_iminuit, mass=m, sigma=s, mu=-1.0, limit_mu=(0.0, 2.0), fix_mass=True, fix_sigma=True)
            # noinspection PyArgumentList
            m_.migrad(nsplit=6)
            
            h0_hypothesis = h0_calc()
            h1_hypothesis = h1_calc(m, s, m_.np_values()[2])
            if m_.np_values()[2] < 1e-8:
                h1_hypothesis = h0_hypothesis
            
            q0_[i] = - 2 * (h0_hypothesis - h1_hypothesis)
            p0_[i] = 0.5 * math.erfc(np.sqrt(abs(- (h0_hypothesis - h1_hypothesis))))
            
            # ProcessHelper.print_status_bar(i, mass_array)
        
        df_ = pd.DataFrame(data=np.array([mass_array, sigma_array, q0_, p0_, np.sqrt(q0_)]).T,
                           columns=["mass", "gauss_sigma", "q0", "p0", "sqrt_q0"])
        
        if to_file:
            df_.to_csv(os.path.join(self.save_dir, f"p0_q0_estimation_{name}.csv"))
        
        self.calculated_dump[f"p0_estimation_{name}"] = df_
        return df_
    
    def calc_p0_unbinned_iminuit(self, to_file=True):
        """
        Calculates the unbinned variant of the estimate of the p0 value using iminuit.
        
        :param to_file: bool
        """
        data_ = self.data_raw.data
        a, b = self.hist_range
        
        self.set_quad_defaults()
        
        temp_s_args = self.func_func_param[self.sfu.__name__][2:]
        b_args = self.func_func_param[self.bfu.__name__]
        b_norm = (sci.quad(self.bfu, a, b, args=tuple(b_args))[0]) ** (-1)
        
        self.data_com = self.__set_data_com()
        n_mc_b = np.sum(self.data_com.data["mc_bac"])
        n_mc_s = np.sum(self.data_com.data["mc_sig"])
        
        dict_ = {"s": {}, "b": {}, "mu_s_b": {}, "exp_mu_s_b": {}}  # norm_dict
        
        @np.vectorize
        def used_func(x_, mass_, sigma_, mu_):
            s_args = (sigma_, mass_, *temp_s_args)
            t_ = f"{(mass_, sigma_, mu_)}"
            if t_ not in dict_["s"].keys():
                dict_["s"][t_] = (sci.quad(self.sfu, a, b, args=s_args)[0]) ** (-1)
            
            def inner_func(xx_):
                return mu_ * dict_["s"][t_] * n_mc_s * self.sfu(xx_, *s_args) + b_norm * n_mc_b * self.bfu(xx_, *b_args)
            
            if t_ not in dict_["mu_s_b"].keys():
                dict_["mu_s_b"][t_] = (sci.quad(inner_func, a, b)[0]) ** (-1)
            return dict_["mu_s_b"][t_] * inner_func(x_)
        
        @np.vectorize
        def exp_val(mass_, sigma_, mu_):
            t_ = (mass_, sigma_, mu_)
            if f"{t_}" not in dict_["exp_mu_s_b"].keys():
                dict_["exp_mu_s_b"][f"{t_}"] = sci.quad(lambda x: x * used_func(x, *t_), a, b)[0]
            return dict_["exp_mu_s_b"][f"{t_}"]
        
        @np.vectorize
        def nll(mass, sigma, mu):
            t_ = (mass, sigma, mu)
            ln = - exp_val(*t_) + np.sum(np.log(exp_val(*t_) * used_func(data_, *t_)))
            return ln
        
        def calc_h0_hypothesis():
            return nll(125, 1.0, 0.0)
        
        @np.vectorize
        def calc_h1_hypothesis(mass, sigma, mu):
            return nll(mass, sigma, mu)
        
        def _calc_h1_iminuit(mass, sigma, mu):  # for iminuit
            return - nll(mass, sigma, mu)
        
        self.__get_p0_q0_array(h1_calc=calc_h1_hypothesis, h1_calc_iminuit=_calc_h1_iminuit, h0_calc=calc_h0_hypothesis,
                               to_file=to_file, name="unbinned")
    
    def calc_p0_binned_iminuit(self, to_file=True):
        """
        Calculates the binned variant of the estimate of the p0 value using iminuit.

        :param to_file: bool
        """
        a, b = self.hist_range
        w = self.bin_width
        
        self.set_quad_defaults()
        
        temp_s_args = self.func_func_param[self.sfu.__name__][2:]
        b_args = self.func_func_param[self.bfu.__name__]
        b_norm = (sci.quad(self.bfu, a, b, args=tuple(b_args))[0]) ** (-1)
        
        self.data_com = self.__set_data_com()
        data_ = self.data_com.data["data"]
        x_used = np.array(self.data_com.x_range)
        n_mc_b = np.sum(self.data_com.data["mc_bac"])
        n_mc_s = np.sum(self.data_com.data["mc_sig"])
        
        dict_ = {"b_norm": {}, "b_int": {}, "s_norm": {}, "s_int": {}}
        
        @np.vectorize
        def used_func(x_, mass_, sigma_, mu_):
            
            s_args = (sigma_, mass_, *temp_s_args)
            t_ = (x_, sigma_, mass_)
            
            if x_ not in dict_["b_int"]:
                dict_["b_int"][x_] = b_norm * sci.quad(self.bfu, x_ - w / 2., x_ + w / 2., args=tuple(b_args))[0]
            
            if t_ not in dict_["s_norm"]:
                dict_["s_norm"][t_] = (sci.quad(self.sfu, a, b, args=s_args)[0]) ** (-1)
            if t_ not in dict_["s_int"]:
                dict_["s_int"][t_] = dict_["s_norm"][t_] * sci.quad(self.sfu, x_ - w / 2., x_ + w / 2., args=s_args)[0]
            
            return mu_ * dict_["s_int"][t_] + dict_["b_int"][x_]
        
        fac = (n_mc_b + n_mc_s) * w
        
        @np.vectorize
        def nll(mass, sigma, mu):
            ll = np.sum(
                np.multiply(data_ * w, np.log(fac * used_func(x_used, mass, sigma, mu))) - fac * used_func(x_used, mass,
                                                                                                           sigma, mu))
            return ll
        
        def calc_h0_hypothesis():
            return nll(125, 1.0, 0.0)
        
        @np.vectorize
        def calc_h1_hypothesis(mass, sigma, mu):
            return nll(mass, sigma, mu)
        
        def _calc_h1_iminuit(mass, sigma, mu):  # for iminuit
            return - nll(mass, sigma, mu)
        
        self.__get_p0_q0_array(h1_calc=calc_h1_hypothesis, h1_calc_iminuit=_calc_h1_iminuit, h0_calc=calc_h0_hypothesis,
                               to_file=to_file, name=f"binned__{round(w, 2)}_per_bin__bins_{self.bins}")
    
    # p0 plots
    
    def plot_p0_estimation(self, plot_=("unbinned", "binned_bins_15"), show_max_sigma_=4, to_file_=True):
        """
        Draws the p0 estimation. You can choose to draw either the unbinned variant
        or the binned variant with any number of bins. For this purpose an instance with
        the necessary bins is temporarily created and the result is stored in a csv file.
        
        :param plot_: tuple
                      possible tuple content: "unbinned", "binned" or "binned_bins_N" with N > 0 bins.
        :param show_max_sigma_: int
                                lower y-achsis  drawing limit.
        :param to_file_: bool
        """
        fig, ax = plt.subplots(1, 1)
        
        dfs = {}
        
        plot_ = plot_ if isinstance(plot_, tuple) else [plot_]
        
        for p in plot_:
            
            if isinstance(p, str):
                if "unbinned" == p:
                    try:
                        df_ = pd.read_csv(os.path.join(self.save_dir, "p0_q0_estimation_unbinned.csv"))
                    except FileNotFoundError:
                        self.calc_p0_unbinned_iminuit(to_file=True)
                        df_ = pd.read_csv(os.path.join(self.save_dir, "p0_q0_estimation_unbinned.csv"))
                    dfs[r"$p_0$ - unbinned"] = df_
                
                if "binned" in p and p != "unbinned":
                    if "bins" not in p or int(p.split("_")[-1]) == self.bins:
                        bw, b = round(self.bin_width, 2), self.bins
                        f_name_ = f"p0_q0_estimation_binned__{bw}_per_bin__bins_{b}.csv"
                        path_ = os.path.join(self.save_dir, f_name_).replace("\\", "/")
                        
                        try:
                            df_ = pd.read_csv(path_)
                        
                        except FileNotFoundError:
                            self.calc_p0_binned_iminuit(to_file=True)
                            df_ = pd.read_csv(path_)
                        
                        dfs[r"$p_0$ - binned " + f"{round(self.bin_width, 1)} GeV/bin"] = df_
                    
                    if "bins" in p and int(p.split("_")[-1]) != self.bins:
                        b_ = int(p.split("_")[-1])
                        bw_ = round((self.hist_range[1] - self.hist_range[0]) / b_, 2)
                        f_name_ = f"p0_q0_estimation_binned__{bw_}_per_bin__bins_{b_}.csv"
                        c_func_func_param = self.func_func_param
                        path_ = os.path.join(self.save_dir, f_name_).replace("\\", "/")
                        
                        try:
                            df_ = pd.read_csv(path_)
                        
                        except FileNotFoundError:
                            c_ = StatTest(bins=b_, hist_range=self.hist_range, info=self.info,
                                          ru_dir=self.ru_dir, mc_dir=self.mc_dir, signal_func=self.sfu,
                                          background_func=self.bfu)
                            c_.func_func_param = c_func_func_param
                            c_.calc_p0_binned_iminuit(to_file=True)
                            df_ = pd.read_csv(path_)
                        
                        dfs[r"$p_0$ - binned " + f"{round(float(bw_), 1)} GeV/bin"] = df_
                
                else:
                    df_ = pd.read_csv(os.path.join(self.save_dir, p)) if isinstance(p, str) else p
                    dfs[r"$p_0$ - unbinned"] = df_
            
            if not isinstance(p, str):
                df_ = pd.read_csv(os.path.join(self.save_dir, p)) if isinstance(p, str) else p
                dfs[r"$p_0$ - unbinned"] = df_
        lss = OrderedDict([
            # ('solid', (0, ())),
            ('loosely dotted', (0, (1, 10))), ('dotted', (0, (1, 5))), ('densely dotted', (0, (1, 1))),
            ('loosely dashed', (0, (5, 10))), ('dashed', (0, (5, 5))), ('densely dashed', (0, (5, 1))),
            ('loosely dashdotted', (0, (3, 10, 1, 10))),
            ('dashdotted', (0, (3, 5, 1, 5))),
            ('densely dashdotted', (0, (3, 1, 1, 1))),
            ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
            ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
        
        for ((key1, df), (key2, linestyle)) in zip(dfs.items(), cycle(lss.items())):
            if "- unbinned" in key1:
                ax.plot(df["mass"], df["p0"], color="black", label=key1, linestyle="-", zorder=10)
            else:
                ax.plot(df["mass"], df["p0"], color="grey", label=key1, linestyle=linestyle)
        
        ax.set_xlabel(r"$m_{4l}$ in GeV")
        ax.set_ylabel(r"$p_{0}$")
        ax.legend(loc="center right", frameon=True, framealpha=1.0, fancybox=True)
        
        PlotHelper.get_p0_sigma_lines(x_=self.__create_mh_for_p0(), ax_obj_=ax, max_sigma_=show_max_sigma_)
        
        file_ = os.path.join(self.save_dir, f"p0_estimation_{self.sfu.__name__}_{self.bfu.__name__}.png")
        plt.savefig(file_) if to_file_ else plt.show()
    
    # plot sigma
    
    def plot_sigma(self, func_, file_="", set_values=True, plot_=True, to_file_=True, ):
        """
        Fits and draws the parameterization of the gaussian width using a function.
        The data for this is provided in advance.
        
        :param func_: function
        :param file_: str
        :param set_values: bool
        :param plot_: bool
        :param to_file_: bool
        """
        
        def get_sigma_mu_and_err(file_):
            df_ = pd.read_csv(file_)
            return tuple(df_[["sigma", "sigma_err", "mu", "mu_err"]].values.T)
        
        sig, sig_err, mu, mu_err = get_sigma_mu_and_err(file_=file_)
        xy_d = K2.XYContainer(x_data=mu, y_data=sig)
        xy_d.add_simple_error("x", np.array(mu_err))
        xy_d.add_simple_error("y", np.array(sig_err))
        xy_fit = K2.XYFit(xy_d, func_)
        xy_fit.do_fit()
        xy_fit.to_file("sigma_gauss_estimate_fit_results.yml", calculate_asymmetric_errors=True)
        xy_fit.assign_parameter_latex_names(x="x", a="a", b="b")
        xy_fit._model_function._formatter._latex_x_name = "m_{4\ell}"
        xy_fit.assign_model_function_latex_name("f_1")
        xy_fit.assign_model_function_latex_expression("{0}({x}- \\bar{{m}}_{{4\ell}}) + {1}")
        
        if set_values:
            if self.verbose: print(f"Set sigma parameter to: {(*xy_fit.parameter_values, np.mean(mu))}")
            self.set_sigma_func_param(item=(*xy_fit.parameter_values, np.mean(mu)))
        
        p = K2.Plot(xy_fit)
        p.customize("data", "label", ["Messung"])
        p.customize("model_line", "label", ["Modell"]).customize("model_line", "color", ["red"])
        p.customize("model_error_band", "label", ["Modellunsicherheit"])
        p.plot(with_ratio=True, with_legend=True, with_fit_info=True, with_asymmetric_parameter_errors=False)
        p.axes[0]["main"].set_ylabel(r"$\sigma_{\rm{G}}$ in GeV")
        p.axes[0]["ratio"].set_xlabel(r"$m_{4\ell}$ in GeV")
        
        xy_fit.to_file(os.path.join(self.save_dir, "fit_results_sigma_estimation.yaml"))
        file_ = os.path.join(self.save_dir, f"sigma_estimation.png")
        plt.savefig(file_) if to_file_ else plt.show() if plot_ else plt.close()
