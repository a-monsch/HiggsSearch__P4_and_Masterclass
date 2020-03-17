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


class StatTestInit(Hist):
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
        super(StatTestInit, self).__init__(bins=bins, hist_range=hist_range)
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
    
    def set_func_val(self, init=False):
        pass
    
    def set_sigma_func_param(self):
        pass
    
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
        
        # may be implemented by students
        # It is also possible to define this part outside of the class
        
        def nll_func(alpha, mass=125.0):
            return 1.0
        
        return nll_func
    
    def calc_1d_unbinned_iminuit(self):
        """
        Calculates the optimal parameters for the one-dimensional profiled likelihood using Iminuit.
        """
        
        # may be implemented by students
        # It is also possible to define this part outside of the class
    
    def calc_2d_unbinned_iminuit(self):
        """
        Calculates the optimal parameters for the two-dimensional profiled likelihood using Iminuit.
        """
        
        # may be implemented by students
        # It is also possible to define this part outside of the class
    
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
    
    def __get_p0_q0_array(self, *args, **kwargs):
        """
        Loop element which calculates the p0 value for different masses and gaussian widths.
        The Iminuit variant of the function must not be np.vectorized, or it must contain
        an explicit function signature. All other functions should be vectorized.
        
        :return: pd.DataFrame
                 containing mass, sigma, q0 and p0
        """
        
        # may be implemented by students
        # It is also possible to define this part outside of the class
        
        df_ = pd.DataFrame()
        
        return df_
    
    def calc_p0_binned_iminuit(self, to_file=True):
        """
        Calculates the binned variant of the estimate of the p0 value using iminuit.

        :param to_file: bool
        """
        
        # extra
    
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
        
        @np.vectorize
        def nll(mass, sigma, mu):
            # may be implemented by students
            # It is also possible to define this part outside of the class
            
            ln = 1.0
            return ln
        
        def calc_h0_hypothesis():
            return nll(125, 1.0, 0.0)
        
        @np.vectorize
        def calc_h1_hypothesis(mass, sigma, mu):
            return nll(mass, sigma, mu)
        
        def _calc_h1_iminuit(mass, sigma, mu):  # for iminuit
            return - nll(mass, sigma, mu)
        
        # may be implemented by students
        # It is also possible to define this part outside of the class
        
        self.__get_p0_q0_array(h1_calc=calc_h1_hypothesis, h1_calc_iminuit=_calc_h1_iminuit, h0_calc=calc_h0_hypothesis,
                               to_file=to_file, name="unbinned")
    
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
                            c_ = StatTestInit(bins=b_, hist_range=self.hist_range, info=self.info,
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
        
        PlotHelper.get_p0_sigma_lines(x_=dfs[r"$p_0$ - unbinned"].mass.values, ax_obj_=ax, max_sigma_=show_max_sigma_)
        
        file_ = os.path.join(self.save_dir, f"p0_estimation_{self.sfu.__name__}_{self.bfu.__name__}.png")
        plt.savefig(file_) if to_file_ else plt.show()
