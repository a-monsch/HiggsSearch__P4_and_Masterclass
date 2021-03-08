import ast
import inspect
import os
from copy import deepcopy
from functools import partial, lru_cache

import iminuit as im
import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import clear_output
from ipywidgets import VBox, HBox
from scipy.integrate import quad

from include.Helper import legend_without_duplicate_labels, mc_hist_scale_factor
from include.Hist import Hist
from include.Hist import _AsymPoissonErr


def _get_histograms(mass_list, mc_dir, bins, hist_range):
    _histograms = {}
    _files = [os.path.join(mc_dir, file) for file in os.listdir(mc_dir) if ".csv" in file]
    _dfs = [(file, pd.read_csv(file)) for file in _files]
    
    for file, df in _dfs:
        _h = Hist(bins=bins, hist_range=hist_range)
        mass = [it for it in mass_list if str(it) in file][0] if "_H_" in file else None
        process, label = ("signal", f"mc_sig") if "_H_" in file else ("background", "mc_bac")
        
        for channel in np.unique(df.channel):
            _h.fill(df.loc[df.channel == channel, "mass"], label=label,
                    global_scale=mc_hist_scale_factor(channel=channel, process=process))
        
        _histograms[f"{label}_{mass}" if mass else label] = _h
    
    for k in _histograms:
        if "mc_sig" in k:
            _histograms[k].set_bins(_histograms["mc_bac"].data["mc_bac"], "mc_bac")
    
    return _histograms


def _flatten_dict(mydict, sep="_", group=True):
    expand = lambda _k, _v: [(f"{_k}{sep}{k}" if group else k, v) for k, v in _flatten_dict(_v).items()] if isinstance(_v, dict) else [(_k, _v)]
    return dict([it for k, v in mydict.items() for it in expand(k, v)])


class _StatEvalFallback(object):
    
    @staticmethod
    def histogram(measurement, background_simulation, signal_simulation,
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
    
    @staticmethod
    def pdf(pdf_func, integration_range, measurement, signal_strength):
        exp_v = lambda a: quad(lambda x: x * pdf_func(a, x), a=integration_range[0], b=integration_range[1])[0]
        _nll_mu = 2 * exp_v(signal_strength) - 2 * np.sum(exp_v(signal_strength) * pdf_func(signal_strength, measurement))
        _nll_0 = 2 * exp_v(0) - 2 * np.sum(exp_v(0) * pdf_func(0, measurement))
        
        _value = round(np.real(np.sqrt(np.array(- _nll_mu + _nll_0, dtype=np.complexfloating))), 3)
        return f"$Z = {_value} \, \\sigma$"


class _BackgroundFunc(object):
    
    @staticmethod
    def poly_grade_0(x, a=0.021739129972412785):
        try:
            return a * np.ones(len(x))
        except:
            return a
    
    @staticmethod
    def poly_grade_1(x, a=-0.004341273513006172, b=0.00020375315644710623):
        return np.polynomial.legendre.legval(x, [a, b])
    
    @staticmethod
    def poly_grade_2(x, a=-0.20416440100593744, b=0.0033598933725970098, c=-8.21911514700874e-06):
        return np.polynomial.legendre.legval(x, [a, b, c])
    
    @staticmethod
    def poly_grade_3(x, a=0.015230934295111354, b=-0.0018495596353961252, c=1.9089241264779767e-05, d=-4.2669306904012716e-08):
        return np.polynomial.legendre.legval(x, [a, b, c, d])


class _SignalFunc(object):
    
    @staticmethod
    def gaussian(x, sigma, mu):
        return np.exp(- 0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2 * np.pi * sigma ** 2)


class _CoreWidget(object):
    measurement = np.array([])
    
    def __init__(self, language="EN", gui_dict="../include/widget/lang/gui.yml", mc_dir="./data/for_widgets"):
        self.mc_dir = mc_dir
        self.gui_dict = gui_dict
        self.la = language  # language
        self.td = self.load_custom_language_dict()
        self.ui_comp = self._get_measurement_ui_components()
        self.measurement_ui = self.get_measurement_ui()
    
    def build_ui(self):
        pass
    
    def _get_measurement_ui_components(self):
        _d = {}
        
        _d["add_measurement_text"] = ipw.Text(description=r"$\phantom{_adm}$", layout=ipw.Layout(width='100px'),
                                              style={'description_width': '0px'}, continuous_update=False)
        
        _d["measurement_options"] = ipw.Dropdown(description=r"$\phantom{_dws}$",
                                                 options=[self.td["add measurement"][self.la],
                                                          self.td["delete measurement"][self.la],
                                                          self.td["reset measurement"][self.la]],
                                                 intend=False)
        
        _d["measurement_show"] = ipw.HTML(value=", ".join([str(round(it, 3)) for it in np.sort(_CoreWidget.measurement)]),
                                          placeholder='',
                                          description=f"    {self.td['measurement'][self.la]}: ",
                                          layout=ipw.Layout(border="0px", width="1000px", height="145px"),
                                          style={"description_width": "125px"})
        
        return _d
    
    def get_measurement_ui(self):
        return HBox([self.ui_comp["measurement_options"], self.ui_comp["add_measurement_text"], self.ui_comp["measurement_show"]],
                    layout=ipw.Layout(height="150px"))
    
    def update_measurement_ui(self, **kwargs):
        
        if kwargs["measurement_options"] == self.td["add measurement"][self.la]:
            if kwargs["add_measurement_text"] != "":
                self.ui_comp["add_measurement_text"].value = ""
                _CoreWidget.measurement = np.append(_CoreWidget.measurement, round(float(kwargs["add_measurement_text"]), 3))
        
        if kwargs["measurement_options"] == self.td["reset measurement"][self.la]:
            self.ui_comp["measurement_options"].value = self.td["add measurement"][self.la]
            _CoreWidget.measurement = np.array([])
        
        if kwargs["measurement_options"] == self.td["delete measurement"][self.la]:
            if kwargs["add_measurement_text"] != "":
                self.ui_comp["add_measurement_text"].value = ""
                try:
                    _CoreWidget.measurement = np.delete(
                        _CoreWidget.measurement,
                        np.where(_CoreWidget.measurement == round(float(kwargs["add_measurement_text"])))[0][0])
                except IndexError:
                    pass
        
        self.ui_comp["measurement_show"].value = ", ".join([str(round(it, 3)) for it in np.sort(_CoreWidget.measurement)])
    
    def load_custom_language_dict(self):
        with open(self.gui_dict, 'r') as outfile:
            my_dict = yaml.full_load(outfile)
        return my_dict
    
    def set_measurement(self, array):
        _CoreWidget.measurement = np.array(array)


class _HiggsPdfWidget(_CoreWidget):
    
    def __init__(self, range=(105, 151), pdf_eval_func=None, **kwargs):
        super().__init__(**kwargs)
        self.pdf_eval_func = pdf_eval_func if pdf_eval_func else _StatEvalFallback.pdf
        self.range = range
        self.ui_comp.update(self._get_ui_components())
        
        self.fit_hist = None
        self._get_hist_to_fit_and_set_background_defaults()
    
    def _get_ui_components(self):
        _d = {}
        
        _d["range_text"] = ipw.Text(description=self.td["range"][self.la], value=str(self.range))
        
        _d["background_model_text"] = ipw.HTML(value=self.td["background model"][self.la])
        
        _d["background_model_options"] = ipw.Dropdown(description="",
                                                      default=self.td["polynomial grade 2"][self.la],
                                                      options=[self.td["polynomial grade 0"][self.la],
                                                               self.td["polynomial grade 1"][self.la],
                                                               self.td["polynomial grade 2"][self.la],
                                                               self.td["polynomial grade 3"][self.la]],
                                                      intend=False)
        
        _d["signal_model_text"] = ipw.HTML(value=self.td["signal model"][self.la])
        
        _d["signal_model_options"] = ipw.Dropdown(description="",
                                                  default=self.td["gaussian"][self.la],
                                                  options=[self.td["gaussian"][self.la]],
                                                  intend=False)
        
        # less laggy variant
        _d["signal_strenght_slider"] = ipw.BoundedFloatText(description=f"$\mu = $", value=0.0, min=0.0, max=2.0, step=0.1)
        _d["signal_width_slider"] = ipw.BoundedFloatText(description=r"$\sigma = $", value=2, min=1.8, max=2.5, step=0.1)
        _d["signal_mu_slider"] = ipw.BoundedFloatText(description=r"$\bar{m} = $", value=125, step=0.1, min=self.range[0], max=self.range[1])
        
        # more laggy variant
        # _d["signal_strenght_slider"] = ipw.FloatSlider(description=r"$\mu$", value=0, step=0.05, min=0, max=2)
        # _d["signal_width_slider"] = ipw.FloatSlider(description=r"$\sigma$", value=2, step=0.05, min=1.8, max=2.5)
        # _d["signal_mu_slider"] = ipw.FloatSlider(description=r"$\bar{m}$", value=125, step=0.05, min=self.range[0], max=self.range[1])
        
        _d["stat_eval_checkbox"] = ipw.Checkbox(False, description=self.td["statistical evaluation"][self.la], intent=False,
                                                layout=ipw.Layout(height="30px", width="250px"))
        
        return _d
    
    def build_ui(self):
        _ui1 = VBox([self.ui_comp["range_text"],
                     self.ui_comp["stat_eval_checkbox"],
                     self.ui_comp["background_model_text"],
                     self.ui_comp["background_model_options"],
                     self.ui_comp["signal_model_text"],
                     self.ui_comp["signal_model_options"],
                     self.ui_comp["signal_strenght_slider"],
                     self.ui_comp["signal_width_slider"],
                     self.ui_comp["signal_mu_slider"]], position="left")
        
        _out = ipw.interactive_output(self.plot, self.ui_comp)
        
        _out.layout.width = "75%"
        _ui1.layout.width = "25%"
        
        return VBox([HBox([_out, _ui1]), self.measurement_ui])
    
    def _get_hist_to_fit_and_set_background_defaults(self, do_fit=True):
        
        if not self.fit_hist:
            self.mc_sigs_mass = [115, 120, 122, 124, 125, 128, 130, 135, 140, 145, 150]
            self.fit_hist = _get_histograms(bins=int((self.range[1] - self.range[0]) / 2),
                                            hist_range=self.range,
                                            mc_dir=self.mc_dir,
                                            mass_list=self.mc_sigs_mass)
            
            self.N_b = np.sum(self.fit_hist["mc_bac"].data["mc_bac"])
            self.N_s = np.sum(self.fit_hist["mc_sig_125"].data["mc_sig"])
        
        if do_fit:
            def change_default_kwargs(func):
                _arg_dict = {it.name: it.default for it in list(inspect.signature(func).parameters.values())[1:]}
                _arg_dict_str = ', '.join(f'{k}={v}' for k, v in _arg_dict.items())
                _arg_dict_keys_str = ', '.join(list(_arg_dict.keys()))
                _data = self.fit_hist["mc_bac"].data["mc_bac"]
                _x = self.fit_hist["mc_bac"].bc
                
                _chi2 = eval(f"lambda {_arg_dict_str}: np.sum((_data - func(_x, {_arg_dict_keys_str})) ** 2)")
                
                _f = im.Minuit(_chi2, **_arg_dict)
                return partial(func, **{k: _f.values[k] for k in _f.parameters})
            
            _BackgroundFunc.poly_grade_0 = change_default_kwargs(_BackgroundFunc.poly_grade_0)
            _BackgroundFunc.poly_grade_1 = change_default_kwargs(_BackgroundFunc.poly_grade_1)
            _BackgroundFunc.poly_grade_2 = change_default_kwargs(_BackgroundFunc.poly_grade_2)
            _BackgroundFunc.poly_grade_3 = change_default_kwargs(_BackgroundFunc.poly_grade_3)
    
    def plot(self, **kwargs):
        
        fig, ax = plt.subplots(1, 1, figsize=(25, 12))
        
        # ----
        
        if ast.literal_eval(kwargs["range_text"]) != self.range:
            clear_output(wait=True)
            self.range = ast.literal_eval(kwargs["range_text"])
            self._get_hist_to_fit_and_set_background_defaults(do_fit=True)
        
        # ----
        
        _db = {self.td["polynomial grade 0"][self.la]: _BackgroundFunc.poly_grade_0,
               self.td["polynomial grade 1"][self.la]: _BackgroundFunc.poly_grade_1,
               self.td["polynomial grade 2"][self.la]: _BackgroundFunc.poly_grade_2,
               self.td["polynomial grade 3"][self.la]: _BackgroundFunc.poly_grade_3}
        _ds = {self.td["gaussian"][self.la]: _SignalFunc.gaussian}
        
        b_pdf = _db[kwargs["background_model_options"]]
        s_pdf = _ds[kwargs["signal_model_options"]]
        
        # ----
        
        sigma = float(kwargs["signal_width_slider"])
        mean = float(kwargs["signal_mu_slider"])
        a = float(kwargs["signal_strenght_slider"])
        
        # ----
        
        @lru_cache
        def norm_s(s, m):
            return (quad(lambda x: s_pdf(x, sigma=s, mu=m), a=self.range[0], b=self.range[1])[0]) ** (-1)
        
        @lru_cache
        def norm_pdf(a):
            return (quad(lambda x: pdf(a, x), a=self.range[0], b=self.range[1])[0]) ** (-1)
        
        @lru_cache
        def pdf(a, x):
            return a * self.N_s * norm_s(s=sigma, m=mean) * s_pdf(x, sigma=sigma, mu=mean) + self.N_b * b_pdf(x)
        
        @np.vectorize
        @lru_cache
        def normed_pdf(a, x):
            return norm_pdf(a) * pdf(a, x)
        
        # ----
        
        _x = np.linspace(*self.range, 1000, endpoint=True)
        _y = normed_pdf(a, _x)
        
        ax.plot(_x, _y, color="royalblue", label=self.td["probability density function"][self.la])
        
        # ----
        
        if kwargs["stat_eval_checkbox"]:
            # pdf_func, integration_range, measurement, signal_strength
            _stat_eval_sig_bac_string = self.pdf_eval_func(normed_pdf, self.range, _CoreWidget.measurement, a)
            ax.text(0.5, 0.7, _stat_eval_sig_bac_string, size=14, bbox=dict(facecolor='white', alpha=0.75),
                    horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
        
        # ----
        
        self.update_measurement_ui(**kwargs)
        
        # ----
        
        if len(_CoreWidget.measurement) > 0:
            _d = _CoreWidget.measurement
            _d = _d[(_d > self.range[0]) & (_d < self.range[1])]
            ax.vlines(_d, 0, 0.1 * np.amax(_y), label=self.td["measurement"][self.la])
        
        # ----
        
        y_plot_limits = (0, np.amax(_y) * 1.2)
        ax.set_yticks([0 + 2 * i for i in range(int(y_plot_limits[1]))])
        ax.set_ylim(*y_plot_limits)
        ax.set_xlim(*self.range)
        
        ax.set_xlabel(r"$m_{4\ell}$ in GeV")
        ax.set_ylabel(self.td["probability"][self.la])
        
        legend_without_duplicate_labels(ax)
        
        fig.canvas.draw()
    
    @property
    def run(self):
        return self.build_ui()


class _HiggsHistogramWidget(_CoreWidget):
    
    def __init__(self, bins=37, hist_range=(70, 181), hist_eval_func=None, **kwargs):
        super().__init__(**kwargs)
        
        self.hist_eval_func = hist_eval_func if hist_eval_func else _StatEvalFallback.histogram
        self.hist_range = hist_range
        self.bins = bins
        
        self.histograms = self._get_mc_histograms()
        self.ui_comp.update(self._get_ui_components())
    
    def set_dirs(self, mc_dir=None, mc_other_dir=None):
        if mc_dir:
            self.mc_dir = mc_dir
        if mc_other_dir:
            self.mc_other_dir = mc_other_dir
        self.histograms = self._get_mc_histograms()
    
    def _get_measurement_histogram(self):
        return np.histogram(_CoreWidget.measurement, bins=self.bins, range=self.hist_range, density=False)[0]
    
    def _get_mc_histograms(self):
        
        self.mc_sigs_mass = [115, 120, 122, 124, 125, 128, 130, 135, 140, 145, 150]
        
        return _get_histograms(bins=self.bins, hist_range=self.hist_range,
                               mc_dir=self.mc_dir, mass_list=self.mc_sigs_mass)
    
    def _get_ui_components(self):
        _d = {}
        
        _d["mc_checkbox"] = {r"$m_{\mathrm{H}}$" + f" = {num} GeV": ipw.Checkbox(False, description=r"$m_{\mathrm{H}}$" + f" = {num} GeV",
                                                                                 layout=ipw.Layout(width="125px", height="30px"),
                                                                                 indent=False) for num in self.mc_sigs_mass}
        
        ipw.BoundedFloatText = partial(ipw.BoundedFloatText, disabled=False, description="",
                                       layout=ipw.Layout(width="175px", height="30px"))
        
        _d["mc_mu_float_text"] = {f"$\phantom{{{num}}}\mu = $": ipw.BoundedFloatText(description=f"$\phantom{{{num}}}\mu = $",
                                                                                     value=1.0, min=0.0, max=1000.0, step=0.05,
                                                                                     indent=False) for num in self.mc_sigs_mass}
        _d["bins_text"] = ipw.Text(description="Bins", value=str(int(self.bins)),
                                   layout=ipw.Layout(width="175px", height="30px"),
                                   intend=False, continuous_update=False)
        
        _d["hist_range_text"] = ipw.Text(description=self.td["range"][self.la], value=str(self.hist_range),
                                         layout=ipw.Layout(width="175px", height="30px"),
                                         intend=False, continuous_update=False)
        
        _d["stat_eval_checkbox"] = ipw.Checkbox(False, description=self.td["statistical evaluation"][self.la], intent=False,
                                                layout=ipw.Layout(height="30px", width="250px"))
        
        return _d
    
    def build_ui(self):
        
        _box1 = VBox([self.ui_comp["bins_text"], self.ui_comp["hist_range_text"], self.ui_comp["stat_eval_checkbox"]])
        _box2 = HBox([VBox([v for v in self.ui_comp["mc_checkbox"].values()]),
                      VBox([v for v in self.ui_comp["mc_mu_float_text"].values()], intend=False)], position="left")
        
        _ui1 = VBox([_box1, _box2])
        
        _out = ipw.interactive_output(self.plot, _flatten_dict(self.ui_comp))
        
        _out.layout.width = "75%"
        _ui1.layout.width = "25%"
        
        return VBox([HBox([_out, _ui1]), self.measurement_ui])
    
    def _hist_eval_func_converter(self, hist_object, num="", mu=""):
        
        _width = hist_object.bin_width
        filter_lower_ = (hist_object.bc - _width / 2. > hist_object.hist_range[0])
        filter_upper_ = (hist_object.bc + _width / 2. < hist_object.hist_range[1])
        
        filter_ = filter_lower_ & filter_upper_
        
        # ----
        
        _background_simulation = hist_object.data["mc_bac"][filter_]
        _signal_simulation = hist_object.data["mc_sig"][filter_]
        _measurement = self._get_measurement_histogram()[filter_]
        
        # ----
        
        mu = f"{mu}" if mu != "" and float(mu) != 1.0 else ""
        
        name_, val_ = self.hist_eval_func(measurement=_measurement,
                                          background_simulation=_background_simulation, signal_simulation=_signal_simulation,
                                          background_name="b", signal_name=f"{mu}s_{{{num} \\ \\mathrm{{GeV}}}}")
        
        return f"{name_} = {round(val_, 3)}"
    
    def plot(self, **kwargs):
        
        _stat_eval_sig_bac_string = ""
        
        fig, ax = plt.subplots(1, 1, figsize=(25, 12))
        
        # ----
        
        self.bins, self.hist_range = int(kwargs["bins_text"]), ast.literal_eval(kwargs["hist_range_text"])
        if self.histograms["mc_bac"].bins != self.bins or self.histograms["mc_bac"].hist_range != self.hist_range:
            clear_output(wait=True)
            self.histograms = self._get_mc_histograms()
        
        # ----
        
        _h, _y_lim_max = Hist(bins=self.bins, hist_range=self.hist_range), 0
        colors = ["green", "orange", "yellow", "cyan", "orangered",
                  "magenta", "red", "brown", "dodgerblue", "silver", "lawngreen"]
        
        for i, (num, color) in enumerate(zip(self.mc_sigs_mass, colors)):
            if kwargs[r"mc_checkbox_$m_{\mathrm{H}}$" + f" = {num} GeV"]:
                _th = deepcopy(self.histograms[f"mc_sig_{num}"])
                _mu = float(kwargs[f"mc_mu_float_text_$\phantom{{{num}}}\mu = $"])
                _th.data["mc_sig"] *= _mu
                _th.draw(label_list=["mc_bac", "mc_sig"],
                         figure=fig, ax=ax,
                         matplotlib_dicts={"mc_bac": {"color": "royalblue",
                                                      "label": f"{self.td['background'][self.la]}:" + r" $ZZ, Z\gamma$",
                                                      "alpha": 1},
                                           "mc_sig": {"color": color,
                                                      "label": f"$m_H = ${num} GeV",
                                                      "alpha": 0.5}})
                
                if np.amax(_th.data["mc_sig"]) > _y_lim_max:
                    _y_lim_max = np.amax(_th.data["mc_sig"])
                if kwargs["stat_eval_checkbox"]:
                    _sig_bac_temp_str = self._hist_eval_func_converter(_th, num=num, mu=f"{_mu}")
                    _stat_eval_sig_bac_string += f"{_sig_bac_temp_str},  " if kwargs["stat_eval_chekbox"] else ""
                del _th
            
            else:
                self.histograms["mc_bac"].draw(label_list=["mc_bac"],
                                               figure=fig, ax=ax,
                                               matplotlib_dicts={"mc_bac": {"label": f"{self.td['background'][self.la]}:" + r" $ZZ, Z\gamma$",
                                                                            "color": "royalblue", "alpha": 1}})
        
        # ----
        
        if kwargs["stat_eval_checkbox"]:
            ax.text(0.5, 0.7, _stat_eval_sig_bac_string[:-3], size=14, bbox=dict(facecolor='white', alpha=0.75),
                    horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
        
        # ----
        
        self.update_measurement_ui(**kwargs)
        
        # ----
        
        _m_hist = self._get_measurement_histogram()
        
        pass_x, pass_y = np.array([]), np.array([])
        for i in range(len(_h.bc)):
            if _m_hist[i] != 0:
                pass_x, pass_y = np.append(pass_x, _h.bc[i]), np.append(pass_y, _m_hist[i])
        
        label_name = self.td["measurement"][self.la] if np.sum(len(_m_hist)) > 0.0 else None
        ax.errorbar(pass_x, pass_y, xerr=0, yerr=_AsymPoissonErr(pass_y).error, fmt="o", marker="o",
                    color="black", label=label_name)
        
        # ----
        
        legend_without_duplicate_labels(ax)
        
        y_plot_limits = (0, float(max(np.amax(self.histograms["mc_bac"].data["mc_bac"]) + 1,
                                      _m_hist[np.argmax(_m_hist)] + np.sqrt(_m_hist[np.argmax(_m_hist)]) + 1,
                                      _y_lim_max + 1)))
        ax.set_yticks([0 + 2 * i for i in range(int(y_plot_limits[1]))])
        ax.set_ylim(*y_plot_limits)
        ax.set_xlim(*self.hist_range)
        
        ax.set_xlabel(r"$m_{4\ell}$ in GeV")
        ax.set_ylabel(self.td["entries"][self.la])
        
        fig.canvas.draw()
    
    @property
    def run(self):
        return self.build_ui()


class HiggsWidget(_CoreWidget):
    
    def __init__(self, language="EN", display="all", **kwargs):
        super().__init__(language=language, **kwargs)
        self.hhw = _HiggsHistogramWidget(language=language, **kwargs)
        self.hpw = _HiggsPdfWidget(language=language, **kwargs)
        self.display = display
    
    def observe_tab(self, x):
        try:
            if self.display == "all":
                self.hhw.run if x["new"]["selected_index"] == 0 else self.hpw.run
            if self.display != "histogram":
                if x["new"]["selected_index"] == 0:
                    if self.display == "histogram":
                        self.hhw.run
                    if self.display == "pdf":
                        self.hpw.run
        except (KeyError, TypeError):
            pass
    
    def build_tab_ui(self):
        if self.display == "all":
            tab = ipw.Tab(children=[self.hhw.run, self.hpw.run], continuous_update=True)
            tab.set_title(0, 'Histogramm')
            tab.set_title(1, 'Pdf')
            tab.observe(lambda x: self.observe_tab(x))
            return tab
        
        if self.display == "histogram":
            tab = ipw.Tab(children=[self.hhw.run], continuous_update=True)
            tab.set_title(0, 'Histogramm')
            tab.observe(lambda x: self.observe_tab(x))
            return tab
        
        if self.display == "pdf":
            tab = ipw.Tab(children=[self.hpw.run], continuous_update=True)
            tab.set_title(0, 'Pdf')
            tab.observe(lambda x: self.observe_tab(x))
            return tab
    
    @property
    def run(self):
        return self.build_tab_ui()
