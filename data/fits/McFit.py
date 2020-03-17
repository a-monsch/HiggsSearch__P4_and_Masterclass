import os

import kafe2 as K2
import matplotlib.pyplot as plt
import numpy as np

from ..RandomHelper import ToSortHelper as TSH
from ..histogramm.HistDataGetter import HistDataGetter as HDG


def array(*args, **kwargs):
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


_oldarray = np.array
np.array = _oldarray


class McFit(object):
    """
    Class that performs an adjustment of the distribution of the MC simulations using the kafe2 module.
    """
    
    def __init__(self, bins=15, hist_range=(106, 151), tag="background",
                 to_chi2_one=False,
                 mc_dir="./data/mc_aftH", ru_dir="./data/ru_aftH",
                 verbose=True, info=None, error_type_model="relative", save_dir="./mc_fits"):
        self.verbose = verbose
        self.error_type_model = error_type_model
        self.to_chi2_one = to_chi2_one
        self.bins = bins
        self.hist_range = hist_range
        self.tag = tag
        self.info = [["2012"], ["A-D"]] if info is None else info
        self.mc_dir = mc_dir
        self.ru_dir = ru_dir
        self.drfd = {}  # DataRawFitDict
        self.func_com = {}
        self.data_raw = self.set_data_raw()
        self.data_com = self.set_data_com()
        self.combined_errors = self.create_combined_errors()
        self.save_dir = save_dir
        self.create_dir()
    
    def create_dir(self):
        """
        Creates folder to save.
        """
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_data_raw(self):
        """
        Get raw_date as Hist object from HistDataGetter.

        Note: Used for calculation of uncertainties
        """
        d = HDG(bins=self.bins, hist_range=self.hist_range, info=self.info, mc_dir=self.mc_dir, ru_dir=self.ru_dir)
        return d.get_mc_raw(tag=self.tag)
    
    def set_data_com(self):
        """
        Creates a Hist object from HistDataGetter.
        Contains the already scaled histogram with all channels.
        """
        d = HDG(bins=self.bins, hist_range=self.hist_range, info=self.info, mc_dir=self.mc_dir, ru_dir=self.ru_dir)
        return d.get_mc_com()
    
    def create_combined_errors(self):
        """
        Creates an array of seld.data_raw after gaussian error propagation.

        :return: ndarray
                 1D array containing data with "float" type.
        """
        combined_errors = np.zeros(len(self.data_com.x_range))
        for i in range(len(self.data_raw.x_range)):
            pass_y_error_temp = 0
            for key in list(self.data_raw.data.keys()):
                corr_fac_square = (self.data_raw.data[key]["corr_fac"]) ** 2
                pass_y_error_temp += (self.data_raw.data[key]["raw_hist"][i] * corr_fac_square)
            combined_errors[i] = np.sqrt(pass_y_error_temp)
        return combined_errors
    
    def create_raw_fit(self, used_func):
        """
        Fits the function to the distribution.
        Repeats the step with scaled uncertainties if to_chi2_one=True.
        The kafe2 object is cached.

        :param used_func: function
        """
        initial_y_error, temp_y_error = self.combined_errors, self.combined_errors
        error_scale_factor, dummy, lock = 1.0, 1, False
        last_used_temp_y_error, last_updated_cost_ndf = temp_y_error, 0
        
        if self.error_type_model != "relative":
            error_scale_factor, dummy = 0.0, 2
        
        while True:
            temp_hist = K2.HistContainer(n_bins=self.data_raw.bins,
                                         bin_range=self.data_raw.hist_range,
                                         bin_edges=list(self.data_raw.bin_edges))
            if self.tag.lower() == "background": temp_hist.set_bins(self.data_com.data["mc_bac"])
            if self.tag.lower() == "signal": temp_hist.set_bins(self.data_com.data["mc_sig"])
            
            temp_hist.add_simple_error(err_val=temp_y_error)
            temp_hist_fit = K2.HistFit(data=temp_hist, model_density_function=used_func,
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
            if used_func.__name__ == "double_side_crystal_ball_anti_sym":
                temp_hist_fit.limit_parameter("sigma", (0.15, 10.0))
                temp_hist_fit.limit_parameter("mu", (120.0, 130.0))
                temp_hist_fit.limit_parameter("alpha_l", (0.0001, 15.0))
                temp_hist_fit.limit_parameter("alpha_r", (0.0001, 15.0))
                temp_hist_fit.limit_parameter("n_l", (1.0001, 15.0))
                temp_hist_fit.limit_parameter("n_r", (1.0001, 15.0))
            if used_func.__name__ == "voigt":
                temp_hist_fit.limit_parameter("mu", (124.0, 126.0))
                pass
            
            temp_hist_fit.do_fit()
            
            ndf = float(len(temp_hist.data) - len(temp_hist_fit.parameter_values))
            initial_c_ndf = temp_hist_fit.cost_function_value / ndf
            
            if not self.to_chi2_one: break
            
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
                
                if (abs(c_ndf - 1) < 0.0001 and c_ndf > 1.0): break
                if last_updated_cost_ndf > 1 and c_ndf < 1: dummy += 1
                
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
                if (abs(c_ndf - 1) < 0.000001 and c_ndf > 1.0) or last_updated_cost_ndf == c_ndf: break
                
                last_updated_cost_ndf = temp_hist_fit.cost_function_value / ndf
        
        if self.to_chi2_one and self.verbose: print(
            f"{used_func.__name__}: {round(error_scale_factor, 7)} {round(last_updated_cost_ndf, 7)}")
        
        self.drfd[used_func.__name__] = {
            "used_hist": temp_hist,
            "fit": temp_hist_fit,
            "initial_chi2_ndf": initial_c_ndf,
            "scale_factor": error_scale_factor
        }
    
    def save_report_to_yaml(self):
        """
        Saves all fit results of all fits in the cache.

        TODO: Insert function that only certain objects should be saved.
        """
        for fn in self.drfd.keys():
            tn = "fit_results.yml"
            if not self.to_chi2_one: tn = f"fit__{self.tag}__year_{self.info[0][0]}__func_{fn}__not_scaled.yml"
            if self.to_chi2_one:
                ichi2 = str(round(self.drfd[fn]["initial_chi2_ndf"], 5)).replace(".", "-")
                sf = str(round(self.drfd[fn]["scale_factor"], 9)).replace(".", "-")
                tn1 = f"fit__{self.tag}__year_{self.info[0][0]}__func_{fn}"
                tn2 = f"__errmodel_{self.error_type_model}__initchi2ndf_{ichi2}__scalef_{sf}.yml"
                tn = tn1 + tn2
            
            self.drfd[fn]["fit"].to_file(os.path.join(self.save_dir, tn), calculate_asymmetric_errors=True)
    
    def plot_fit(self, used_func, used_func_dict=None,
                 with_ratio=False, plot_conture=False, to_file=False, separate=True,
                 make_title=False):
        """
        Draws the result of the performed kafe2 fit using the kafe2 plot method.

        :param used_func: list
                          1D list containing functions
        :param used_func_dict: list
                               1D list containing dicts for formatting (if not in docstring of function, optional)
        :param with_ratio: bool
        :param plot_conture: bool
        :param to_file: bool
        :param separate: bool
        :param make_title: bool
        """
        if not isinstance(used_func, list): used_func = [used_func]
        
        if used_func_dict is not None:
            if not isinstance(used_func_dict, list): used_func_dict = [used_func_dict]
        if used_func_dict is None:
            used_func_dict = [TSH.dict_out_of_doc(func_) for func_ in used_func]
        
        func_fits = [self.drfd[func.__name__]["fit"] for func in used_func]
        
        for fit, f_dict in zip(func_fits, used_func_dict):
            if f_dict is None:
                continue
            fit.assign_model_function_latex_name(f_dict["func_name"])
            fit.assign_parameter_latex_names(**f_dict["func_param"])
            fit.assign_model_function_latex_expression(f_dict["func_expr"])
            fit._model_function._formatter._latex_x_name = f_dict["x_name"]
        
        h_plot = K2.Plot(fit_objects=func_fits, separate_figures=False)
        if len(func_fits) == 1:
            h_plot.customize("data", "label", ["MC Simulation"])
            h_plot.customize("model_density", "label", ["Dichte"])
            h_plot.customize("model", "label", ["Modell"])
        if len(func_fits) > 1:
            label_list = [None for _ in func_fits]
            # noinspection PyTypeChecker
            label_list[0] = "MC Simulation"
            label_color = ["k" for _ in func_fits]
            h_plot.customize("data", "label", label_list).customize("data", "color", label_color)
            h_plot.customize("model_density", "label", [f"Dichte {i + 1}" for i in range(len(used_func))])
            h_plot.customize("model", "label", [f"Modell {i + 1}" for i in range(len(used_func))])
        
        h_plot.plot(with_ratio=with_ratio, ratio_range=(0.0, 2.0))
        if with_ratio:
            h_plot.axes[0]["ratio"].set_xlabel(r"$m_{4\ell}$ in GeV")
            h_plot.axes[0]["ratio"].hlines(1.0, self.hist_range[0], self.hist_range[1],
                                           color="black", alpha=1, linewidth=0.75)
        if not with_ratio: h_plot.axes[0]["main"].set_xlabel(r"$m_{4\ell}$ in GeV")
        
        h_plot.axes[0]["main"].set_ylabel("Bineinträge")
        
        if make_title:
            fig = h_plot.figures[-1]
            fig.set_tight_layout(False)
            
            pre_, fn_, fv_ = f"{self.info[0][0]}, {self.tag}, $4\\mu + 4e + 2e2\\mu$: ", "\\sqrt{{\\chi^2_{{min}}}}", []
            
            for func, f_dict in zip(used_func, used_func_dict):
                if not self.to_chi2_one:
                    continue
                if self.to_chi2_one:
                    if f_dict is None:
                        fv_.append(f"${func.__name__}$: ${fn_} = {round(self.drfd[func.__name__]['scale_factor'], 3)}$")
                    if f_dict is not None:
                        fv_.append(
                            f"${f_dict['func_name']}$: ${fn_} = {round(self.drfd[func.__name__]['scale_factor'], 3)}$")
            
            fig.suptitle(pre_ + "; ".join(fv_))
        
        if to_file:
            if separate:
                for func in used_func:
                    if self.to_chi2_one:
                        plt.savefig(os.path.join(self.save_dir,
                                                 f"{self.tag}__year_{self.info[0][0]}__func_{func.__name__}__f_{round(self.drfd['scale_factor'], 4)}.png"))
                    if not self.to_chi2_one:
                        plt.savefig(os.path.join(self.save_dir,
                                                 f"{self.tag}__year_{self.info[0][0]}__func_{func.__name__}.png"))
            
            if not separate:
                funcs = ""
                sfs = ""
                for func in used_func:
                    funcs += f"_{func.__name__}"
                    sfs += f"_{round(self.drfd[func.__name__]['scale_factor'], 3)}"
                
                plt.savefig(
                    os.path.join(self.save_dir, f"{self.tag}__year_{self.info[0][0]}__func_{funcs}__f_{sfs}.png"))
        
        if not to_file: plt.show()
        
        if plot_conture:
            for fit, func in zip(func_fits, used_func):
                cpf = K2.ContoursProfiler(fit, contour_sigma_values=(1, 2))
                cpf.plot_profiles_contours_matrix()
                fig_cpf = cpf.figures[-1]
                fig_cpf.set_tight_layout(False)
                fig_cpf.suptitle(f"{self.tag}, {self.info[0][0]}, {func.__name__}")
                
                if to_file:
                    plt.savefig(
                        os.path.join(self.save_dir, f"contour_{self.tag}__{self.info[0][0]}__func_{func.__name__}.png"))
                if not to_file: plt.show()
    
    def get_results(self, used_func):
        """
        Collects and passes the parameters and uncertainties
        to the used_func as a tuple if the fit was performed.

        :param used_func: functon
        :return: tuple
                 (parameter_values, parameter_errors)
        """
        try:
            fit_ = self.drfd[used_func.__name__]["fit"].get_result_dict_for_robots()
            res_ = fit_["parameter_values"], fit_["parameter_errors"]
        except KeyError:
            self.create_raw_fit(used_func=used_func)
            fit_ = self.drfd[used_func.__name__]["fit"].get_result_dict_for_robots()
            res_ = fit_["parameter_values"], fit_["parameter_errors"]
        
        return res_
    
    @property
    def variables_for_xy_fit(self):
        """
        Passes all necessary variables for a standalone xy fit
        (the uncertainties on the data are already calculated).

        :return: tuple
                 (x_data, y_data, y_errors)
        """
        to_pass_tuple = []
        
        to_pass_tuple.append(self.data_com.x_range)
        if self.tag.lower() == "background": to_pass_tuple.append(self.data_com.data["mc_bac"])
        if self.tag.lower() == "signal": to_pass_tuple.append(self.data_com.data["mc_sig"])
        to_pass_tuple.append(self.combined_errors)
        
        return tuple(to_pass_tuple)
    
    @property
    def variables_for_hist_fit(self):
        """
        Passes all necessary variables for a standalone histogram fit
        (the uncertainties on the data are already calculated).

        :return: tuple
                 (bins, histogramm_range, histogramm_data, histogramm_data_errors)
        """
        to_pass_tuple = []
        to_pass_tuple.append(self.bins)
        to_pass_tuple.append(self.hist_range)
        if self.tag.lower() == "background": to_pass_tuple.append(self.data_com.data["mc_bac"])
        if self.tag.lower() == "signal": to_pass_tuple.append(self.data_com.data["mc_sig"])
        to_pass_tuple.append(self.combined_errors)
        return tuple(to_pass_tuple)
