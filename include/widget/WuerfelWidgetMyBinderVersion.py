import ast

import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import yaml
from IPython.display import clear_output
from ipywidgets import VBox, HBox


class WuerfelWidget(object):
    
    def __init__(self, language="EN"):
        
        self.la = language
        self.td = self.load_language_dict()
        
        self.ui_components = self.build_ui_components()
        self.ui = None
        
        self.own_simulation_func = None
        self.own_measurement_func = None
        self.own_mean_func = None
        self.own_norm_func = None
        self.own_std_all_func = None
        self.own_std_indv_func = None
        self.own_measurement_scale_func = None
        self.own_statistical_evaluation_func = None
        
        self.bins = 6
        self.simulation_list = np.array([])
        self.measurement_list = np.array([])
    
    def load_language_dict(self):
        with open('../include/widget/lang/gui.yml', 'r') as outfile:
            my_dict = yaml.full_load(outfile)
        return my_dict
    
    def build_ui_components(self):
        dropdown_simulation = ipw.Dropdown(description=r"$\phantom{_sd}$",
                                           options=[f"{self.td['simulate'][self.la].capitalize()} N {self.td['times'][self.la]}:",
                                                    f"{self.td['reset simulation'][self.la]}"], intend=False,
                                           layout=ipw.Layout(width="225px", height="30px"),
                                           style={"description_width": "0px"})
        
        txtwindw_simulation = ipw.Text(description=r"$\phantom{_st}$", value="", continuous_update=False, intent=False,
                                       layout=ipw.Layout(width="200px", height="30px"),
                                       style={"description_width": "0px"})
        
        dropdown_measurement = ipw.Dropdown(description=r"$\phantom{_md}$",
                                            options=[f"{self.td['measure'][self.la].capitalize()} N {self.td['times'][self.la]}:",
                                                     f"{self.td['add one measurement'][self.la]}:",
                                                     f"{self.td['add n measurement'][self.la]}:",
                                                     f"{self.td['reset measurement'][self.la]}"], intend=False,
                                            layout=ipw.Layout(width="225px", height="30px"),
                                            style={"description_width": "0px"})
        
        txtwindw_measurement = ipw.Text(description=r"$\phantom{_mt}$", value="", continuous_update=False, intent=False,
                                        layout=ipw.Layout(width="200px", height="30px"),
                                        style={"description_width": "0px"})
        
        checkbx_mean = ipw.Checkbox(False, description=f"{self.td['mean'][self.la]}",
                                    layout=ipw.Layout(width="250px", height="30px"),
                                    indent=False, style={"description_width": "300px"})
        
        checkbx_std_all = ipw.Checkbox(False,
                                       description=f"{self.td['standard deviation'][self.la]} ({self.td['all'][self.la]})",
                                       layout=ipw.Layout(width="250px", height="30px"),
                                       indent=False, style={"description_width": "300px"})
        
        checkbx_std_indv = ipw.Checkbox(False,
                                        description=f"{self.td['standard deviation'][self.la]} ({self.td['individual'][self.la]})",
                                        layout=ipw.Layout(width="300px", height="30px"),
                                        indent=False, style={"description_width": "300px"})
        
        checkbx_norm = ipw.Checkbox(False, description=f"{self.td['simulation normalization'][self.la]}",
                                    layout=ipw.Layout(width="250px", height="30px"),
                                    indent=False, style={"description_width": "300px"})
        
        checkbx_scale_on_m = ipw.Checkbox(False, description=f"{self.td['simulation scaling'][self.la]}",
                                          layout=ipw.Layout(width="250px", height="30px"),
                                          indent=False, style={"description_width": "300px"})
        
        checkbx_stat_eval = ipw.Checkbox(False, description=f"{self.td['statistical evaluation'][self.la]}",
                                         layout=ipw.Layout(width="250px", height="30px"),
                                         indent=False, style={"description_width": "300px"})
        
        label_stat_eval = ipw.Label(description="Statistical evaluation Label", value="",
                                    layout=ipw.Layout(width="250px", height="30px"),
                                    indent=False, style={"description_width": "0px"})
        
        bins = ipw.Text(description="Bins", value=str(int(6)),
                        layout=ipw.Layout(width="175px", height="30px"),
                        intend=False, continuous_update=False)
        
        return {"dropdown_simulation": dropdown_simulation, "dropdown_measurement": dropdown_measurement,
                "txtwindw_measurement": txtwindw_measurement, "txtwindw_simulation": txtwindw_simulation,
                "checkbx_mean": checkbx_mean, "checkbx_norm": checkbx_norm, "checkbx_scale_on_m": checkbx_scale_on_m,
                "checkbx_std_all": checkbx_std_all, "checkbx_std_indv": checkbx_std_indv, "checkbx_stat_eval": checkbx_stat_eval,
                "bins": bins, "label_stat_eval": label_stat_eval}
    
    def _check_visibility(self):
        self.ui_components["txtwindw_simulation"].layout.display = "none" if self.own_simulation_func is None else "visible"
        self.ui_components["dropdown_simulation"].layout.display = "none" if self.own_simulation_func is None else "visible"
        
        self.ui_components["txtwindw_measurement"].layout.display = "none" if self.own_measurement_func is None else "visible"
        self.ui_components["dropdown_measurement"].layout.display = "none" if self.own_measurement_func is None else "visible"
        
        self.ui_components["checkbx_mean"].layout.display = "none" if self.own_mean_func is None else "visible"
        self.ui_components["checkbx_norm"].layout.display = "none" if self.own_norm_func is None else "visible"
        self.ui_components["checkbx_scale_on_m"].layout.display = "none" if self.own_measurement_scale_func is None else "visible"
        self.ui_components["checkbx_std_all"].layout.display = "none" if self.own_std_all_func is None else "visible"
        self.ui_components["checkbx_std_indv"].layout.display = "none" if self.own_std_indv_func is None else "visible"
        
        self.ui_components["checkbx_stat_eval"].layout.display = "none" if self.own_statistical_evaluation_func is None else "visible"
        self.ui_components["label_stat_eval"].layout.display = "none" if self.own_statistical_evaluation_func is None else "visible"
    
    def build_out_opt_dict(self):
        return {self.ui_components["txtwindw_simulation"].description: self.ui_components["txtwindw_simulation"],
                self.ui_components["dropdown_simulation"].description: self.ui_components["dropdown_simulation"],
                self.ui_components["txtwindw_measurement"].description: self.ui_components["txtwindw_measurement"],
                self.ui_components["dropdown_measurement"].description: self.ui_components["dropdown_measurement"],
                self.ui_components["checkbx_mean"].description: self.ui_components["checkbx_mean"],
                self.ui_components["checkbx_norm"].description: self.ui_components["checkbx_norm"],
                self.ui_components["checkbx_scale_on_m"].description: self.ui_components["checkbx_scale_on_m"],
                self.ui_components["checkbx_std_all"].description: self.ui_components["checkbx_std_all"],
                self.ui_components["checkbx_std_indv"].description: self.ui_components["checkbx_std_indv"],
                self.ui_components["checkbx_stat_eval"].description: self.ui_components["checkbx_stat_eval"],
                self.ui_components["bins"].description: self.ui_components["bins"],
                self.ui_components["label_stat_eval"].description: self.ui_components["label_stat_eval"]}
    
    def build_out(self):
        return ipw.interactive_output(self.plot, self.build_out_opt_dict())
    
    def build_ui(self):
        self._check_visibility()
        
        _ui_0 = HBox([self.ui_components["bins"]])
        
        _b_sim = HBox([self.ui_components["dropdown_simulation"], self.ui_components["txtwindw_simulation"]])
        _b_msm = HBox([self.ui_components["dropdown_measurement"], self.ui_components["txtwindw_measurement"]])
        
        _ui_1 = VBox([_b_sim, _b_msm], layout=ipw.Layout(border="solid, 5px"))
        
        _ui_2 = VBox([self.ui_components["checkbx_mean"],
                      self.ui_components["checkbx_norm"],
                      self.ui_components["checkbx_std_all"],
                      self.ui_components["checkbx_std_indv"],
                      self.ui_components["checkbx_scale_on_m"]],
                     layout=ipw.Layout(border="solid, 5px"))
        
        _ui_3 = HBox([self.ui_components["checkbx_stat_eval"], self.ui_components["label_stat_eval"]], layout=ipw.Layout(border="solid, 5px"))
        
        _ui_tot = VBox([_ui_0, _ui_1, _ui_2, _ui_3])
        
        _out = self.build_out()
        _out.layout.width = "68%"
        _ui_tot.layout.width = "28%"
        
        _ui_final = VBox([HBox([_out, _ui_tot])])
        self.ui = _ui_final
    
    def _run_n_times_simulation(self, n):
        self.simulation_list = np.append(self.simulation_list, self.own_simulation_func(n))
    
    def _run_n_times_measurement(self, n):
        self.measurement_list = np.append(self.measurement_list, self.own_measurement_func(n))
    
    def _fill_measurement_list_from_dict(self, dict_):
        for key, value in dict_.items():
            if int(float(value)) == 0:
                continue
            _temp = int(float(key)) * np.ones(int(float(value)))
            self.measurement_list = np.append(self.measurement_list, _temp)
    
    def plot(self, *args, **kwargs):
        
        if int(kwargs[self.ui_components["bins"].description]) != self.bins:
            self.bins = int(kwargs[self.ui_components["bins"].description])
        
        if kwargs[
            self.ui_components["dropdown_simulation"].description] == f"{self.td['simulate'][self.la].capitalize()} N {self.td['times'][self.la]}:":
            self.ui_components["txtwindw_simulation"].value = ""
            if kwargs[self.ui_components["txtwindw_simulation"].description] != "":
                self.ui_components["txtwindw_simulation"].value = ""
                sim_num = int(kwargs[self.ui_components["txtwindw_simulation"].description])
                self._run_n_times_simulation(sim_num)
                clear_output(True)
        
        if kwargs[self.ui_components["dropdown_simulation"].description] == f"{self.td['reset simulation'][self.la]}":
            print(f"{self.td['to reset measurement type Yes'][self.la]}")
            self.ui_components["txtwindw_simulation"].value = ""
            if kwargs[self.ui_components["txtwindw_simulation"].description] == "Yes":
                self.ui_components["txtwindw_simulation"].value = ""
                self.simulation_list = np.array([])
                clear_output(True)
        
        if kwargs[
            self.ui_components["dropdown_measurement"].description] == f"{self.td['measure'][self.la].capitalize()} N {self.td['times'][self.la]}:":
            self.ui_components["txtwindw_measurement"].value = ""
            if kwargs[self.ui_components["txtwindw_measurement"].description] != "":
                self.ui_components["txtwindw_measurement"].value = ""
                try:
                    sim_num = int(kwargs[self.ui_components["txtwindw_measurement"].description])
                    self._run_n_times_measurement(sim_num)
                except ValueError:
                    pass
                clear_output(True)
        
        if kwargs[self.ui_components["dropdown_measurement"].description] == f"{self.td['add one measurement'][self.la]}:":
            self.ui_components["txtwindw_measurement"].value = ""
            if kwargs[self.ui_components["txtwindw_measurement"].description] != "":
                self.ui_components["txtwindw_measurement"].value = ""
                try:
                    _num = int(kwargs[self.ui_components["txtwindw_measurement"].description])
                    self.measurement_list = np.append(self.measurement_list, _num)
                except ValueError:
                    pass
                clear_output(True)
        
        if kwargs[self.ui_components["dropdown_measurement"].description] == f"{self.td['reset measurement'][self.la]}":
            print(f"{self.td['to reset measurement type Yes'][self.la]}")
            self.ui_components["txtwindw_measurement"].value = ""
            if kwargs[self.ui_components["txtwindw_measurement"].description] == "Yes":
                self.ui_components["txtwindw_measurement"].value = ""
                self.measurement_list = np.array([])
                clear_output(True)
        
        if kwargs[self.ui_components["dropdown_measurement"].description] == f"{self.td['add n measurement'][self.la]}:":
            self.ui_components["txtwindw_measurement"].value = "{'1':0,'2':0,'3':0,'4':0,'5':0,'6':0}"
            if kwargs[self.ui_components["txtwindw_measurement"].description] != "{'1':0,'2':0,'3':0,'4':0,'5':0,'6':0}":
                self.ui_components["txtwindw_measurement"].value = "{'1':0,'2':0,'3':0,'4':0,'5':0,'6':0}"
                try:
                    _m_dict = ast.literal_eval(kwargs[self.ui_components["txtwindw_measurement"].description])
                    self._fill_measurement_list_from_dict(_m_dict)
                except SyntaxError:
                    pass
                clear_output(True)
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        plot_y_array = np.histogram(self.simulation_list, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]
        plot_ym_array = np.histogram(self.measurement_list, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]
        
        size_ = self.own_measurement_scale_func(plot_ym_array) if kwargs[self.ui_components["checkbx_scale_on_m"].description] else 1.0
        
        if kwargs[self.ui_components["checkbx_norm"].description] or kwargs[self.ui_components["checkbx_scale_on_m"].description]:
            plot_y_array = size_ * self.own_norm_func(plot_y_array)
        
        if kwargs[self.ui_components["checkbx_mean"].description]:
            ax.hlines(self.own_mean_func(plot_y_array), -1, self.bins + 2, ls="--", label=f"{self.td['mean'][self.la]}")
        
        if kwargs[self.ui_components["checkbx_std_all"].description]:
            m_ = self.own_mean_func(plot_y_array)
            std_ = self.own_std_all_func(np.histogram(self.simulation_list, self.bins, density=False, range=(0.5, self.bins + 0.5))[0])
            if kwargs[self.ui_components["checkbx_norm"].description]:
                std_ *= (1. / np.sum(np.histogram(self.simulation_list, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]))
            if not kwargs[self.ui_components["checkbx_norm"].description] and kwargs[self.ui_components["checkbx_scale_on_m"].description]:
                std_ *= (1. / np.sum(np.histogram(self.simulation_list, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]))
            std_ *= size_
            
            ax.fill_between([-1, self.bins + 2], [m_ + std_, m_ + std_], [m_ - std_, m_ - std_],
                            color="green", alpha=0.25, label=f"{self.td['standard deviation'][self.la]} ({self.td['all'][self.la]})")
        
        if kwargs[self.ui_components["checkbx_std_indv"].description]:
            y_err_ = self.own_std_indv_func(np.histogram(self.simulation_list, self.bins, density=False, range=(0.5, self.bins + 0.5))[0])
            if kwargs[self.ui_components["checkbx_norm"].description]:
                y_err_ *= (1. / np.sum(np.histogram(self.simulation_list, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]))
            if not kwargs[self.ui_components["checkbx_norm"].description] and kwargs[self.ui_components["checkbx_scale_on_m"].description]:
                y_err_ *= (1. / np.sum(np.histogram(self.simulation_list, self.bins, density=False, range=(0.5, self.bins + 0.5))[0]))
            y_err_ *= size_
            
            ax.errorbar(np.arange(1, self.bins + 1, 1), plot_y_array, yerr=y_err_, fmt="bx", marker="",
                        ecolor="royalblue", alpha=1.0, capsize=int((0.125 / 2. * 1000) / self.bins),
                        label=f"{self.td['standard deviation'][self.la]} ({self.td['individual'][self.la]})")
        
        if kwargs[self.ui_components["checkbx_stat_eval"].description]:
            _name, _value = self.own_statistical_evaluation_func(plot_ym_array, plot_y_array)
            self.ui_components["label_stat_eval"].value = f"{_name} = {round(_value, 3)}"
        
        if np.sum(plot_ym_array) > 0.0:
            ax.errorbar(np.arange(1, self.bins + 1, 1), plot_ym_array, xerr=0,
                        yerr=self.own_std_indv_func(plot_ym_array),
                        fmt="ko", lw=2, label=f"{self.td['measurement'][self.la]}")
            ax.legend()
        if np.sum(plot_y_array) > 0.0:
            ax.bar(np.arange(1, self.bins + 1, 1), plot_y_array,
                   color="royalblue", alpha=0.5, label=f"{self.td['simulation'][self.la]}")
            ax.legend()
        
        ax.set_xlim(0.5, 0.5 + self.bins)
        #ax.set_ylim(0, max(np.amax(plot_y_array) + 1 * np.sqrt(np.amax(plot_y_array)),
        #                   np.amax(plot_ym_array) + 1 * np.sqrt(np.amax(plot_ym_array)),
        #                   0.1))
        ax.set_xticks(np.arange(1, self.bins + 1, 1))
        ax.set_ylabel(self.td["entries"][self.la])
        ax.set_xlabel("n")
        
        fig.canvas.draw()
    
    @property
    def run(self):
        self.build_ui()
        return self.ui
