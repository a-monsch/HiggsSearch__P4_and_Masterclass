import ast

import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import clear_output
from ipywidgets import VBox, HBox

from include.Helper import legend_without_duplicate_labels


class DiceWidget(object):
    
    def __init__(self, language="EN", gui_dict="../include/widget/lang/gui.yml", bins=6):
        self.gui_dict = gui_dict
        self.la = language  # language
        self.td = self._load_language_dict()
        
        self.bins = bins
        self.simulation = np.array([])
        self.measurement = np.array([])
        
        self.own_simulation_func = None
        self.own_measurement_func = None
        self.own_mean_func = None
        self.own_norm_func = None
        self.own_std_all_func = None
        self.own_std_indv_func = None
        self.own_measurement_scale_func = None
        self.own_statistical_evaluation_func = None
        
        self.ui_comp = self._get_ui_components()
        self.ui = None
    
    def _load_language_dict(self):
        with open(self.gui_dict, 'r') as outfile:
            my_dict = yaml.full_load(outfile)
        return my_dict
    
    def _get_ui_components(self):
        _d = {}
        _d["dropdown_options_simulation"] = ipw.Dropdown(description=r"$\phantom{_sd}$",
                                                         options=[f"{self.td['simulate'][self.la].capitalize()} N {self.td['times'][self.la]}:",
                                                                  f"{self.td['reset simulation'][self.la]}"], intend=False,
                                                         layout=ipw.Layout(width="225px", height="30px"),
                                                         style={"description_width": "0px"})
        
        _d["text_simulation"] = ipw.Text(description=r"$\phantom{_st}$", value="", continuous_update=False, intent=False,
                                         layout=ipw.Layout(width="200px", height="30px"),
                                         style={"description_width": "0px"})
        
        _d["dropdown_options_measurement"] = ipw.Dropdown(description=r"$\phantom{_md}$",
                                                          options=[f"{self.td['measure'][self.la].capitalize()} N {self.td['times'][self.la]}:",
                                                                   f"{self.td['add one measurement'][self.la]}:",
                                                                   f"{self.td['add n measurement'][self.la]}:",
                                                                   f"{self.td['reset measurement'][self.la]}"], intend=False,
                                                          layout=ipw.Layout(width="225px", height="30px"),
                                                          style={"description_width": "0px"})
        
        _d["text_measurement"] = ipw.Text(description=r"$\phantom{_mt}$", value="", continuous_update=False, intent=False,
                                          layout=ipw.Layout(width="200px", height="30px"),
                                          style={"description_width": "0px"})
        
        _d["checkbox_mean"] = ipw.Checkbox(False, description=f"{self.td['mean'][self.la]}",
                                           layout=ipw.Layout(width="250px", height="30px"),
                                           indent=False, style={"description_width": "300px"})
        
        _d["checkbox_std_all"] = ipw.Checkbox(False,
                                              description=f"{self.td['standard deviation'][self.la]} ({self.td['all'][self.la]})",
                                              layout=ipw.Layout(width="250px", height="30px"),
                                              indent=False, style={"description_width": "300px"})
        
        _d["checkbox_std_indv"] = ipw.Checkbox(False,
                                               description=f"{self.td['standard deviation'][self.la]} ({self.td['individual'][self.la]})",
                                               layout=ipw.Layout(width="300px", height="30px"),
                                               indent=False, style={"description_width": "300px"})
        
        _d["checkbox_norm"] = ipw.Checkbox(False, description=f"{self.td['simulation normalization'][self.la]}",
                                           layout=ipw.Layout(width="250px", height="30px"),
                                           indent=False, style={"description_width": "300px"})
        
        _d["checkbox_scale_on_measurement"] = ipw.Checkbox(False, description=f"{self.td['simulation scaling'][self.la]}",
                                                           layout=ipw.Layout(width="250px", height="30px"),
                                                           indent=False, style={"description_width": "300px"})
        
        _d["checkbox_stat_eval"] = ipw.Checkbox(False, description=f"{self.td['statistical evaluation'][self.la]}",
                                                layout=ipw.Layout(width="250px", height="30px"),
                                                indent=False, style={"description_width": "300px"})
        
        _d["text_stat_eval"] = ipw.Label(description="Statistical evaluation Label", value="",
                                         layout=ipw.Layout(width="250px", height="30px"),
                                         indent=False, style={"description_width": "0px"})
        
        _d["text_bins"] = ipw.Text(description="Bins", value=str(int(self.bins)),
                                   layout=ipw.Layout(width="175px", height="30px"),
                                   intend=False, continuous_update=False)
        
        _d["text_show_measurement"] = ipw.HTML(value=self._ui_helper__build_table(),
                                               placeholder='',
                                               description=r'$\phantom{history_table}$',
                                               layout=ipw.Layout(border="0px", width="90%", height="100px"),
                                               style={"description_width": "0px"}
                                               )
        
        return _d
    
    def _ui_helper__build_table(self):
        _print_header = ["", *[str(i) for i in range(1, int(self.bins + 1))], "Total"]
        _plot_ys_array = np.histogram(self.simulation, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]
        _ys_sum = np.sum(_plot_ys_array)
        _plot_ys_array_str = ["    " if _ys_sum == 0.0 else item for item in _plot_ys_array]
        
        _plot_ym_array = np.histogram(self.measurement, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]
        _ym_sum = np.sum(_plot_ym_array)
        _plot_ym_array_str = ["    " if _ym_sum == 0.0 else item for item in _plot_ym_array]
        
        _df = pd.DataFrame(columns=_print_header)
        _df = _df.append({k: v for k, v in zip(_print_header, [f"{self.td['simulation'][self.la]}: ",
                                                               *_plot_ys_array_str,
                                                               _ys_sum if float(_ys_sum) != 0.0 else "    "])}, ignore_index=True)
        if _ym_sum != 0.0 or self.own_measurement_scale_func is not None:
            _df = _df.append({k: v for k, v in zip(_print_header, [f"{self.td['measurement'][self.la]}: ",
                                                                   *_plot_ym_array_str,
                                                                   _ym_sum if float(_ym_sum) != 0.0 else "    "])}, ignore_index=True)
        _df = _df.set_index("")
        _df_html = _df.to_html(col_space="55px", notebook=True)
        return _df_html
    
    def _ui_helper__check_visibility(self):
        self.ui_comp["text_simulation"].layout.display = "none" if not self.own_simulation_func else "visible"
        self.ui_comp["dropdown_options_simulation"].layout.display = "none" if self.own_simulation_func is None else "visible"
        
        self.ui_comp["text_measurement"].layout.display = "none" if not self.own_measurement_func else "visible"
        self.ui_comp["dropdown_options_measurement"].layout.display = "none" if self.own_measurement_func is None else "visible"
        
        self.ui_comp["checkbox_mean"].layout.display = "none" if not self.own_mean_func else "visible"
        self.ui_comp["checkbox_norm"].layout.display = "none" if not self.own_norm_func else "visible"
        self.ui_comp["checkbox_scale_on_measurement"].layout.display = "none" if not self.own_measurement_scale_func else "visible"
        self.ui_comp["checkbox_std_all"].layout.display = "none" if not self.own_std_all_func else "visible"
        self.ui_comp["checkbox_std_indv"].layout.display = "none" if not self.own_std_indv_func else "visible"
        
        self.ui_comp["checkbox_stat_eval"].layout.display = "none" if not self.own_statistical_evaluation_func else "visible"
        self.ui_comp["text_stat_eval"].layout.display = "none" if not self.own_statistical_evaluation_func else "visible"
    
    def build_ui(self):
        self._ui_helper__check_visibility()
        
        _ui_0 = HBox([self.ui_comp["text_bins"]])
        
        _b_sim = HBox([self.ui_comp["dropdown_options_simulation"], self.ui_comp["text_simulation"]])
        _b_msm = HBox([self.ui_comp["dropdown_options_measurement"], self.ui_comp["text_measurement"]])
        _b_info = HBox([self.ui_comp["text_show_measurement"]])
        
        _ui_1 = VBox([_b_sim, _b_msm, _b_info], layout=ipw.Layout(border="solid, 5px"))
        
        _ui_2 = VBox([self.ui_comp["checkbox_mean"],
                      self.ui_comp["checkbox_norm"],
                      self.ui_comp["checkbox_std_all"],
                      self.ui_comp["checkbox_std_indv"],
                      self.ui_comp["checkbox_scale_on_measurement"]],
                     layout=ipw.Layout(border="solid, 5px"))
        
        _ui_3 = HBox([self.ui_comp["checkbox_stat_eval"], self.ui_comp["text_stat_eval"]], layout=ipw.Layout(border="solid, 5px"))
        
        _ui_tot = VBox([_ui_0, _ui_1, _ui_2, _ui_3])
        
        _out = ipw.interactive_output(self.plot, self.ui_comp)
        _out.layout.width = "70%"
        _ui_tot.layout.width = "30%"
        
        return VBox([HBox([_out, _ui_tot])])
    
    def _run_n_times(self, n, kind="simulation"):
        if kind == "simulation":
            self.simulation = np.append(self.simulation, self.own_simulation_func(n))
        if kind == "measurement":
            self.measurement = np.append(self.measurement, self.own_measurement_func(n))
    
    def _get_measurement_from_dict(self, dict_):
        for key, value in dict_.items():
            if int(float(value)) == 0:
                continue
            _temp = int(float(key)) * np.ones(int(float(value)))
            self.measurement = np.append(self.measurement, _temp)
    
    def plot(self, **kwargs):
        
        if int(kwargs["text_bins"]) != self.bins:
            self.bins = int(kwargs["text_bins"])
        
        if kwargs["dropdown_options_simulation"] == f"{self.td['simulate'][self.la].capitalize()} N {self.td['times'][self.la]}:":
            self.ui_comp["text_simulation"].value = ""
            if kwargs["text_simulation"] != "":
                self.ui_comp["text_simulation"].value = ""
                sim_num = int(kwargs["text_simulation"])
                self._run_n_times(sim_num, "simulation")
                clear_output(True)
        
        if kwargs["dropdown_options_simulation"] == f"{self.td['reset simulation'][self.la]}":
            self.simulation = np.array([])
            self.ui_comp["dropdown_options_simulation"].value = f"{self.td['simulate'][self.la].capitalize()} N {self.td['times'][self.la]}:"
            clear_output(True)
        
        if kwargs["dropdown_options_measurement"] == f"{self.td['measure'][self.la].capitalize()} N {self.td['times'][self.la]}:":
            self.ui_comp["text_measurement"].value = ""
            if kwargs["text_measurement"] != "":
                self.ui_comp["text_measurement"].value = ""
                try:
                    sim_num = int(kwargs["text_measurement"])
                    self._run_n_times(sim_num, "measurement")
                except ValueError:
                    pass
                clear_output(True)
        
        if kwargs["dropdown_options_measurement"] == f"{self.td['add one measurement'][self.la]}:":
            self.ui_comp["text_measurement"].value = ""
            if kwargs["text_measurement"] != "":
                self.ui_comp["text_measurement"].value = ""
                try:
                    _num = int(kwargs["text_measurement"])
                    self.measurement = np.append(self.measurement, _num)
                except ValueError:
                    pass
                clear_output(True)
        
        if kwargs["dropdown_options_measurement"] == f"{self.td['reset measurement'][self.la]}":
            self.ui_comp["dropdown_options_measurement"].value = f"{self.td['measure'][self.la].capitalize()} N {self.td['times'][self.la]}:"
            print(f"{self.td['to reset measurement type Yes'][self.la]}")
            self.measurement = np.array([])
            clear_output(True)
        
        if kwargs["dropdown_options_measurement"] == f"{self.td['add n measurement'][self.la]}:":
            self.ui_comp["text_measurement"].value = f"""{{{", ".join(f'"{i + 1}": 0' for i in range(self.bins))}}}"""
            if kwargs["text_measurement"] != f"""{{{", ".join(f'"{i + 1}": 0' for i in range(self.bins))}}}""":
                self.ui_comp["text_measurement"].value = f"""{{{", ".join(f'"{i + 1}": 0' for i in range(self.bins))}}}"""
                try:
                    _m_dict = ast.literal_eval(kwargs["text_measurement"])
                    self._get_measurement_from_dict(_m_dict)
                except SyntaxError:
                    pass
                clear_output(True)
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        plot_y_array = np.histogram(self.simulation, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]
        plot_ym_array = np.histogram(self.measurement, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]
        
        size_ = self.own_measurement_scale_func(plot_ym_array) if kwargs["checkbox_scale_on_measurement"] else 1.0
        
        if kwargs["checkbox_norm"] or kwargs["checkbox_scale_on_measurement"]:
            plot_y_array = size_ * self.own_norm_func(plot_y_array)
        
        if kwargs["checkbox_mean"]:
            ax.hlines(self.own_mean_func(plot_y_array), -1, self.bins + 2, ls="--", label=f"{self.td['mean'][self.la]}")
        
        if kwargs["checkbox_std_all"]:
            m_ = self.own_mean_func(plot_y_array)
            std_ = self.own_std_all_func(np.histogram(self.simulation, self.bins, density=False, range=(0.5, self.bins + 0.5))[0])
            if kwargs["checkbox_norm"]:
                std_ *= (1. / np.sum(np.histogram(self.simulation, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]))
            if not kwargs["checkbox_norm"] and kwargs["checkbox_scale_on_measurement"]:
                std_ *= (1. / np.sum(np.histogram(self.simulation, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]))
            std_ *= size_
            
            ax.fill_between([-1, self.bins + 2], [m_ + std_, m_ + std_], [m_ - std_, m_ - std_],
                            color="green", alpha=0.25, label=f"{self.td['standard deviation'][self.la]} ({self.td['all'][self.la]})")
        
        if kwargs["checkbox_std_indv"]:
            y_err_ = self.own_std_indv_func(np.histogram(self.simulation, self.bins, density=False, range=(0.5, self.bins + 0.5))[0])
            if kwargs["checkbox_norm"]:
                y_err_ *= (1. / np.sum(np.histogram(self.simulation, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]))
            if not kwargs["checkbox_norm"] and kwargs["checkbox_scale_on_measurement"]:
                y_err_ *= (1. / np.sum(np.histogram(self.simulation, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]))
            y_err_ *= size_
            
            ax.errorbar(np.arange(1, self.bins + 1, 1), plot_y_array, yerr=y_err_, fmt="bx", marker="",
                        ecolor="royalblue", alpha=1.0, capsize=int((0.125 / 2. * 1000) / self.bins),
                        label=f"{self.td['standard deviation'][self.la]} ({self.td['individual'][self.la]})")
        
        if kwargs["checkbox_stat_eval"]:
            _name, _value = self.own_statistical_evaluation_func(plot_ym_array, plot_y_array)
            self.ui_comp["text_stat_eval"].value = f"{_name} = {round(_value, 3)}"
        
        if np.sum(plot_y_array) > 0.0:
            ax.bar(np.arange(1, self.bins + 1, 1), plot_y_array,
                   color="royalblue", alpha=0.5, label=f"{self.td['simulation'][self.la]}")
            ax.legend()
        
        if np.sum(plot_ym_array) > 0.0:
            ax.errorbar(np.arange(1, self.bins + 1, 1), plot_ym_array, xerr=0,
                        yerr=self.own_std_indv_func(plot_ym_array),
                        fmt="ko", lw=2, label=f"{self.td['measurement'][self.la]}")
            ax.legend()
        
        ax.set_xlim(0.5, 0.5 + self.bins)
        ax.set_xticks(np.arange(1, self.bins + 1, 1))
        ax.set_ylabel(self.td["entries"][self.la])
        ax.set_xlabel("n")
        
        self.ui_comp["text_show_measurement"].value = self._ui_helper__build_table()
        
        legend_without_duplicate_labels(ax)
        
        fig.canvas.draw()
    
    @property
    def run(self):
        return self.build_ui()
