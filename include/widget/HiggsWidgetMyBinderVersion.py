import ast
from copy import deepcopy
from functools import partial

import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import yaml
from IPython.display import clear_output
from ipywidgets import VBox, HBox

from include.RandomHelper import ToSortHelper as TSH
from include.histogramm.Hist import Hist
from include.histogramm.HistHelper import HistHelper
from include.widget.helper.WidgetHistLoader import get_histograms


def _own_stat_eval_func(measurement, background_simulation, signal_simulation,
                        background_name="b", signal_name="s"):
    b_nll = sum(bac_s - m + m * np.log(m / bac_s) for (m, bac_s) in zip(measurement, background_simulation) if float(m) != 0.0)
    bs_nll = sum((bac_s + sig_s) - m + m * np.log(m / (bac_s + sig_s)) for (m, bac_s, sig_s) in zip(measurement,
                                                                                                    background_simulation,
                                                                                                    signal_simulation) if float(m) != 0.0)
    
    nlr_ = 2 * (b_nll - bs_nll)
    q0_ = np.round(nlr_, 3) if nlr_ > 0 else 0
    
    bn_, sn_ = background_name, signal_name
    name_ = f"$ \sqrt{{ 2 \\ln \\left( \\dfrac{{ \\mathcal{{L}}_{{ {bn_} + {sn_} }} }}{{ \\mathcal{{L}}_{{ {bn_} }} }}  \\right) }}$"
    
    return name_, np.sqrt(q0_)


class HiggsWidget(object):
    
    def __init__(self, measurement_list=None, bins=37, hist_range=(70, 181), language="EN", stat_eval_func=None,
                 mc_dir="../data/for_widgets/mc_aftH",
                 mc_other_dir="../data/for_widgets/other_mc/dXXX/mc_aftH"):
        self.mc_other_dir = mc_other_dir
        self.mc_dir = mc_dir
        self.stat_eval_func = _own_stat_eval_func if stat_eval_func is None else stat_eval_func
        self.hist_range = hist_range
        self.bins = bins
        self.measurement_list = np.array([]) if measurement_list is None else measurement_list
        self.mc_sig_name_list = [115, 120, 122, 124, 125, 128, 130, 135, 140, 145, 150]
        
        self.la = language
        self.td = self.load_language_dict()
        
        self.ui_components = self.get_ui_components()
        self.histograms = get_histograms(bins=self.bins, hist_range=self.hist_range,
                                         mc_dir=self.mc_dir, mc_other_dir=self.mc_other_dir)
        self.ui = None
    
    def load_language_dict(self):
        with open('../include/widget/lang/gui.yml', 'r') as outfile:
            my_dict = yaml.full_load(outfile)
        return my_dict
    
    def _measurement_array_to_str(self):
        return ", ".join([str(round(item, 3)) for item in np.sort(self.measurement_list)])
    
    def _measurement_histogram(self, bins=None, hist_range=None):
        return np.histogram(self.measurement_list, bins=self.bins if bins is None else bins,
                            range=self.hist_range if hist_range is None else hist_range,
                            density=False)[0]
    
    def _stat_eval_bac_sig(self, hist_object, num="", mu=""):
        _width = hist_object.bin_width
        filter_lower_ = (hist_object.x_range - _width / 2. > hist_object.hist_range[0])
        filter_upper_ = (hist_object.x_range + _width / 2. < hist_object.hist_range[1])
        filter_ = filter_lower_ & filter_upper_
        
        _background_simulation = hist_object.data["mc_bac"][filter_]
        _signal_simulation = hist_object.data["mc_sig"][filter_]
        
        _measurement = self._measurement_histogram(hist_object.bins, hist_object.hist_range)[filter_]
        
        mu = f"{mu}" if mu != "" and float(mu) != 1.0 else ""
        
        name_, val_ = self.stat_eval_func(measurement=_measurement,
                                          background_simulation=_background_simulation, signal_simulation=_signal_simulation,
                                          background_name="b", signal_name=f"{mu}s_{{{num} \\ \\mathrm{{GeV}}}}")
        
        return f"{name_} = {round(val_, 3)}"
    
    def get_ui_components(self):
        check_boxes_mc = [(num, ipw.Checkbox(False, description=r"$m_{\mathrm{H}}$" + f" = {num} GeV",
                                             layout=ipw.Layout(width="125px", height="30px"),
                                             indent=False)) for num in self.mc_sig_name_list]
        
        ipw.BoundedFloatText = partial(ipw.BoundedFloatText,
                                       disabled=False,
                                       description="",
                                       layout=ipw.Layout(width="175px", height="30px"))
        
        float_text_mu = [(num, ipw.BoundedFloatText(description=f"$\phantom{{{num}}}\mu = $",
                                                    value=1.0, min=0.0, max=1000.0, step=0.05,
                                                    indent=False)) for num in self.mc_sig_name_list]
        
        bins_ = ipw.Text(description="Bins", value=str(int(self.bins)),
                         layout=ipw.Layout(width="175px", height="30px"),
                         intend=False, continuous_update=False)
        hist_range_ = ipw.Text(description=self.td["range"][self.la], value=str(self.hist_range),
                               layout=ipw.Layout(width="175px", height="30px"),
                               intend=False, continuous_update=False)
        stat_eval_ = ipw.Checkbox(False, description=self.td["statistical evaluation"][self.la], intent=False,
                                  layout=ipw.Layout(height="30px", width="250px"))
        
        add_m_val = ipw.Text(description=r"$\phantom{_adm}$", layout=ipw.Layout(width='100px'),
                             style={'description_width': '0px'}, continuous_update=False)
        select_m_option = ipw.Dropdown(description=r"$\phantom{_dws}$",
                                       options=[self.td["add measurement"][self.la],
                                                self.td["delete measurement"][self.la],
                                                self.td["reset measurement"][self.la]],
                                       intend=False)
        
        m_show = ipw.HTML(value=self._measurement_array_to_str(),
                          placeholder='',
                          description=f"    {self.td['measurement'][self.la]}: ",
                          layout=ipw.Layout(border="0px", width="1000px", height="145px"),
                          style={"description_width": "125px"})
        
        return {"check_boxes_mc": check_boxes_mc,
                "float_text_mu": float_text_mu,
                "bins": bins_,
                "hist_range": hist_range_,
                "stat_eval": stat_eval_,
                "add_m_val": add_m_val,
                "select_m_option": select_m_option,
                "m_show": m_show}
    
    def build_out_opt_dict(self):
        _opt_mc_checks = {item[1].description: item[1] for item in self.ui_components["check_boxes_mc"]}
        _opts_mc_scale = {item[1].description: item[1] for item in self.ui_components["float_text_mu"]}
        
        return {**_opt_mc_checks, **_opts_mc_scale,
                self.ui_components["bins"].description: self.ui_components["bins"],
                self.ui_components["hist_range"].description: self.ui_components["hist_range"],
                self.ui_components["stat_eval"].description: self.ui_components["stat_eval"],
                self.ui_components["add_m_val"].description: self.ui_components["add_m_val"],
                self.ui_components["select_m_option"].description: self.ui_components["select_m_option"],
                self.ui_components["m_show"].description: self.ui_components["m_show"]}
    
    def build_out(self):
        return ipw.interactive_output(self.plot, self.build_out_opt_dict())
    
    def build_ui(self):
        _b1 = VBox([item[1] for item in self.ui_components["check_boxes_mc"]])
        _b2 = VBox([item[1] for item in self.ui_components["float_text_mu"]], intend=False)
        _ui_0 = HBox([_b1, _b2], position="left")
        
        _b3 = VBox([self.ui_components["bins"],
                    self.ui_components["hist_range"],
                    self.ui_components["stat_eval"]])
        
        _ui_1 = VBox([_b3, _ui_0])
        
        _out = self.build_out()
        _out.layout.width = "70%"
        _ui_1.layout.width = "30%"
        
        _ui_2 = HBox([self.ui_components["select_m_option"],
                      self.ui_components["add_m_val"],
                      self.ui_components["m_show"]],
                     layout=ipw.Layout(height="150px"))
        
        _ui_final = VBox([HBox([_out, _ui_1]), _ui_2])
        self.ui = _ui_final
    
    def plot(self, *args, **kwargs):
        _stat_eval_sig_bac_string = ""
        fig, ax = plt.subplots(1, 1, figsize=(25, 12))
        
        bins = int(kwargs[self.ui_components["bins"].description])
        hist_range = ast.literal_eval(kwargs[self.ui_components["hist_range"].description])
        
        if self.histograms["mc_bac"].bins != bins or self.histograms["mc_bac"].hist_range != hist_range:
            clear_output()
            self.bins = bins
            self.hist_range = hist_range
            self.histograms = get_histograms(bins=bins, hist_range=hist_range,
                                             mc_dir=self.mc_dir, mc_other_dir=self.mc_other_dir)
        
        h = Hist(bins=bins, hist_range=hist_range)
        colors = ["green", "orange", "yellow", "cyan", "orangered",
                  "magenta", "red", "brown", "dodgerblue", "silver", "lawngreen"]
        _y_lim_max = 0.0
        for i, (num, color) in enumerate(zip(self.mc_sig_name_list, colors)):
            if kwargs[r"$m_{\mathrm{H}}$" + f" = {num} GeV"]:
                temp_hist = deepcopy(self.histograms[f"mc_bac_sig_{num}"])
                _mu = float(kwargs[f"$\phantom{{{num}}}\mu = $"])
                temp_hist.data["mc_sig"] *= _mu
                temp_hist.draw(pass_name=["mc_bac", f"mc_sig"],
                               color=["royalblue", color],
                               label=[f"{self.td['background'][self.la]}:" + r" $ZZ, Z\gamma$",
                                      f"$m_H = ${num} GeV"],
                               alpha=[1, 0.5],
                               figure=fig, ax=ax)
                if np.amax(temp_hist.data["mc_sig"]) > _y_lim_max:
                    _y_lim_max = np.amax(temp_hist.data["mc_sig"])
                if kwargs[self.ui_components["stat_eval"].description]:
                    _sig_bac_temp_str = self._stat_eval_bac_sig(temp_hist, num=num, mu=f"{_mu}")
                    _stat_eval_sig_bac_string += f"{_sig_bac_temp_str},  " if kwargs[self.ui_components["stat_eval"].description] else ""
                del temp_hist
            
            else:
                self.histograms["mc_bac"].draw(pass_name=["mc_bac"],
                                               color=["royalblue"],
                                               label=[f"{self.td['background'][self.la]}:" + r" $ZZ, Z\gamma$"],
                                               figure=fig, ax=ax)
        
        TSH.legend_without_duplicate_labels(ax)
        
        if kwargs[self.ui_components["stat_eval"].description]:
            ax.text(0.5, 0.7, _stat_eval_sig_bac_string[:-3], size=14, bbox=dict(facecolor='white', alpha=0.75),
                    horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
        
        if kwargs[self.ui_components["select_m_option"].description] == self.td["add measurement"][self.la]:
            if kwargs[self.ui_components["add_m_val"].description] != "":
                self.ui_components["add_m_val"].value = ""
                self.measurement_list = np.append(self.measurement_list,
                                                  round(float(kwargs[self.ui_components["add_m_val"].description]), 3))
        
        if kwargs[self.ui_components["select_m_option"].description] == self.td["reset measurement"][self.la]:
            self.ui_components["select_m_option"].value = self.td["add measurement"][self.la]
            self.measurement_list = np.array([])
        
        if kwargs[self.ui_components["select_m_option"].description] == self.td["delete measurement"][self.la]:
            if kwargs[self.ui_components["add_m_val"].description] != "":
                self.ui_components["add_m_val"].value = ""
                try:
                    self.measurement_list = np.delete(self.measurement_list,
                                                      np.where(
                                                          self.measurement_list == round(float(kwargs[self.ui_components["add_m_val"].description])))[
                                                          0][0])
                except IndexError:
                    pass
        
        TSH.legend_without_duplicate_labels(ax)
        
        self.ui_components["m_show"].value = self._measurement_array_to_str()
        _m_hist = self._measurement_histogram(self.bins, self.hist_range)
        
        pass_x, pass_y = np.array([]), np.array([])
        y_err_calc_func = HistHelper.calc_errors_alternative_near_simplified
        for i in range(len(h.x_range)):
            if _m_hist[i] != 0:
                pass_x, pass_y = np.append(pass_x, h.x_range[i]), np.append(pass_y, _m_hist[i])
        
        label_name = self.td["measurement"][self.la] if np.sum(len(_m_hist)) > 0.0 else None
        ax.errorbar(pass_x, pass_y, xerr=0, yerr=y_err_calc_func(pass_y), fmt="o", marker="o",
                    color="black", label=label_name)
        
        TSH.legend_without_duplicate_labels(ax)
        
        y_plot_limits = (0, float(max(np.amax(self.histograms["mc_bac"].data["mc_bac"]) + 1,
                                      _m_hist[np.argmax(_m_hist)] + np.sqrt(_m_hist[np.argmax(_m_hist)]) + 1,
                                      _y_lim_max + 1)))
        ax.set_yticks([0 + 2 * i for i in range(int(y_plot_limits[1]))])
        ax.set_ylim(*y_plot_limits)
        ax.set_xlim(*hist_range)
        
        ax.set_xlabel(r"$m_{4\ell}$ in GeV")
        ax.set_ylabel(self.td["entries"][self.la])
        
        # fig.canvas.draw()
        plt.show()
    
    @property
    def run(self):
        self.build_ui()
        return self.ui
