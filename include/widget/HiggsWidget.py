import math
import sys
from copy import deepcopy
from functools import partial

import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QLineEdit
from matplotlib.widgets import Cursor
from tqdm import tqdm

from include.RandomHelper import ToSortHelper as TSH
from include.histogramm.Hist import Hist
from include.histogramm.HistHelper import HistHelper
from include.widget.helper.ButtonCustomizeHelper import ButtomCustomizeHelper as BCH
from include.widget.helper.CorePlot import PlotInitWidget


def _own_stat_eval_func(measurement, background_simulation, signal_simulation, background_name="b", signal_name="s"):
    b_nll = sum(bac_s - m + m * np.log(m / bac_s) for (m, bac_s) in zip(measurement,
                                                                        background_simulation) if float(m) != 0.0)
    bs_nll = sum((bac_s + sig_s) - m + m * np.log(m / (bac_s + sig_s)) for (m, bac_s, sig_s) in
                 zip(measurement, background_simulation, signal_simulation) if float(m) != 0.0)
    
    nlr_ = 2 * (b_nll - bs_nll)
    q0_ = np.round(nlr_, 3) if nlr_ > 0 else 0
    
    bn_, sn_ = background_name, signal_name
    
    name_ = f"$ 2 \\ln \\left( \\dfrac{{ \\mathcal{{L}}_{{ {bn_} + {sn_} }} }}{{ \\mathcal{{L}}_{{ {bn_} }} }}  \\right)$"
    
    return name_, q0_


class PlotHiggs(PlotInitWidget):
    
    def __init__(self, *args, b_num=37, hist_range=(70, 181), info=None, stat_eval_func=None,
                 mc_dir_="../data/mc_aftH", mc_other_dir="../data/other_mc/dXXX/mc_aftH", on_pseudo_data=True,
                 **kwargs):
        self.on_pseudo_data = on_pseudo_data
        
        self.stat_eval_func = _own_stat_eval_func if stat_eval_func is None else stat_eval_func
        
        self.info = info if info is not None else [["2012"], ["A-D"]]
        
        self.mc_other_dir = mc_other_dir
        self.mc_dir_ = mc_dir_
        self.num_buttons, self.hist_range = b_num, hist_range
        
        self.other_mc_sig_num_list = [115, 120, 122, 124, 128, 130, 135, 140, 145, 150]
        self.mc_sig_name_list = [115, 120, 122, 124, 125, 128, 130, 135, 140, 145, 150]
        
        super(PlotHiggs, self).__init__(*args, b_num=self.num_buttons, draw_h_line=False, **kwargs)
        BCH.connect.button_and_plot(num=self.num_buttons, button_name=self.n_plus, func_name=self.plot)
        BCH.connect.button_and_plot(num=self.num_buttons, button_name=self.n_minus, func_name=self.plot)
        
        # check -> c_
        self._visible_other_mc = False
        
        self._visible_mu = False
        
        self._visible_stat_eval = False
        self.lower_border_, self.upper_border_ = self.hist_range
        
        self.used_Ui()
        
        self.histograms = {}
        self.load_hists()
    
    def used_Ui(self):
        self.HistUiCall()
        
        BCH.connect.submenu_and_function = partial(BCH.connect.submenu_and_function, instance_=self)
        BCH.setting.submenu_bullet = partial(BCH.setting.submenu_bullet, instance_=self, menu_bar_=self.menubar)
        
        self.menu_ansicht = BCH.setting.menu_bullet(instance_=self, name_="Ansicht", menu_bar_=self.menubar)
        self.menu_ansicht_draw_mc, self.menu_ansicht_stat_eval, self.menu_ansicht_mu = BCH.setting.submenu_bullet(
            name_=["MC Simulationen an", "Statistische Bewertung an", "Signal MC Skalierung an"], menu_bullet_=self.menu_ansicht)
        BCH.connect.submenu_and_function(object_=self.menu_ansicht_draw_mc, func_=self.change_other_mc_option)
        BCH.connect.submenu_and_function(object_=self.menu_ansicht_stat_eval, func_=self.change_stat_eval_option)
        BCH.connect.submenu_and_function(object_=self.menu_ansicht_mu, func_=self.change_mu_option)
        
        self.initHistButtonUi()
    
    def change_mu_option(self):
        self._visible_mu = not self._visible_mu
        self.change_mu_visibility()
        BCH.setting.text(self.menu_ansicht_draw_mc,
                         "Signal MC Skalierung an" if not self._visible_other_mc else "Signal MC Skalierung aus")
    
    def change_mu_visibility(self):
        for i, (spinbox, spinbox_text) in enumerate(zip(self.spinboxes, self.spinbox_text)):
            spinbox.setVisible(self._visible_mu and self.check_buttons[i].isChecked() and self._visible_other_mc)
            spinbox_text.setVisible(self._visible_mu and self.check_buttons[i].isChecked() and self._visible_other_mc)
    
    def change_other_mc_option(self):
        self._visible_other_mc = not self._visible_other_mc
        for button in self.check_buttons:
            button.setVisible(self._visible_other_mc)
        
        BCH.setting.text(self.menu_ansicht_draw_mc,
                         "MC Simulationen an" if not self._visible_other_mc else "MC Simulationen aus")
    
    def change_stat_eval_option(self):
        self._visible_stat_eval = not self._visible_stat_eval
        BCH.setting.text(self.menu_ansicht_stat_eval,
                         "Statistische Bewertung an" if not self._visible_stat_eval else "Statistische Bewertung aus")
        self._stat_eval_set_border_window() if self._visible_stat_eval else self.plot()
    
    def _stat_eval_set_border_window(self):
        
        def _on_click_set_border():
            self.lower_border_ = int(_win.set_txt_box_l_border.text())
            self.upper_border_ = int(_win.set_txt_box_u_border.text())
            _win.close()
            self.plot()
        
        _win = QDialog(self)
        BCH.setting.geometry(_win, 100, 100, 200, 140)
        
        _win.info_txt_box = QtWidgets.QTextBrowser(_win)
        BCH.setting.button(_win.info_txt_box, text_="Betrachtungsintervall setzen.",
                           x_=20, y_=10, w_=200, h_=30, transparent_=True)
        
        _win.info_txt_box_l_border = QtWidgets.QTextBrowser(_win)
        BCH.setting.button(_win.info_txt_box_l_border, text_="untere Grenze", x_=20, y_=45, w_=80, h_=30, transparent_=True)
        
        _win.info_txt_box_u_border = QtWidgets.QTextBrowser(_win)
        BCH.setting.button(_win.info_txt_box_u_border, text_="obere Grenze", x_=20, y_=90, w_=80, h_=30, transparent_=True)
        
        _win.set_txt_box_l_border = QLineEdit(_win)
        BCH.setting.geometry(_win.set_txt_box_l_border, x_=100, y_=45, w_=80, h_=20)
        BCH.setting.text(_win.set_txt_box_l_border, f"{self.lower_border_}")
        
        _win.set_txt_box_u_border = QLineEdit(_win)
        BCH.setting.geometry(_win.set_txt_box_u_border, x_=100, y_=90, w_=80, h_=20)
        BCH.setting.text(_win.set_txt_box_u_border, f"{self.upper_border_}")
        
        _win.set_button = QtWidgets.QPushButton("setzen", _win)
        BCH.setting.geometry(_win.set_button, x_=140, y_=115, w_=40, h_=20)
        _win.set_button.clicked.connect(_on_click_set_border)
        
        _win.exec_()
    
    def initHistButtonUi(self):
        self.check_buttons = [QtWidgets.QCheckBox(self) for _ in self.mc_sig_name_list]
        self.spinboxes = [QtWidgets.QDoubleSpinBox(self) for _ in self.mc_sig_name_list]
        self.spinbox_text = [QtWidgets.QTextBrowser(self) for _ in self.mc_sig_name_list]
        for i, (button, spinbox, spinbox_text, name) in enumerate(zip(self.check_buttons,
                                                                      self.spinboxes,
                                                                      self.spinbox_text,
                                                                      self.mc_sig_name_list)):
            BCH.setting.button(object_=button, x_=self.f_width + self.b_space, y_=self.f_start_y + 30 * i, w_=175, h_=30,
                               text_=f"Higgsmasse: {str(name)} GeV", func_=self.plot)
            button.setVisible(self._visible_other_mc)
            button.setChecked(False)
            BCH.setting.geometry(object_=spinbox, x_=button.x() + button.width() + self.b_space - 30,
                                 y_=self.f_start_y + 30 * i + 5, w_=50, h_=20)
            
            BCH.setting.button(object_=spinbox_text, x_=spinbox.x() - 30, y_=button.y() + 5,
                               w_=50, h_=30, transparent_=True, text_="mu: ")
            spinbox_text.setVisible(False)
            
            spinbox.setMinimum(0.0)
            spinbox.setValue(1.0)
            spinbox.setSingleStep(0.1)
            spinbox.setVisible(self._visible_mu)
            spinbox.valueChanged.connect(self.plot)
    
    def save_operation(self):
        self.figure.savefig(self.file_name)
        self.canvas.draw()
    
    def load_hists(self):
        
        h = Hist(bins=self.num_buttons, hist_range=self.hist_range)
        h.set_empty_bin(["data", "mc_bac"])
        h.fill_hist_from_dir(col_="mass_4l", dir_=self.mc_dir_, info=self.info)
        
        self.y_plot_limits = (0, math.ceil(np.amax(h.data["mc_bac"])))
        self.histograms["mc_bac"] = h
        
        h = Hist(bins=self.num_buttons, hist_range=self.hist_range)
        h.set_empty_bin(["data", "mc_bac", "mc_sig"])
        h.fill_hist_from_dir(col_="mass_4l", dir_=self.mc_dir_, info=self.info)
        
        self.histograms["mc_bac_sig_125"] = h
        
        for i, num in tqdm(enumerate(self.other_mc_sig_num_list), total=len(self.other_mc_sig_num_list)):
            file_list = TSH.mixed_file_list(TSH.get_other_mc_dir(num, dir_=self.mc_other_dir), main_dir_=self.mc_dir_)
            h = Hist(bins=self.num_buttons, hist_range=self.hist_range, signal_mass_mc=None if self.on_pseudo_data else num)
            h.set_empty_bin(["data", "mc_bac", "mc_sig"])
            h.fill_hist_from_dir(col_="mass_4l", dir_=None, info=self.info, file_list_=file_list)
            self.histograms[f"mc_bac_sig_{num}"] = h
    
    def _stat_eval_bac_sig(self, hist_object, num="", mu=""):
        
        _width = hist_object.bin_width
        filter_lower_ = (hist_object.x_range - _width / 2. > self.lower_border_)
        filter_upper_ = (hist_object.x_range + _width / 2. < self.upper_border_)
        filter_ = filter_lower_ & filter_upper_
        
        _background_simulation = hist_object.data["mc_bac"][filter_]
        _signal_simulation = hist_object.data["mc_sig"][filter_]
        
        _measurement = self.list_num[filter_]
        
        mu = f"{mu}" if mu != "" and float(mu) != 1.0 else ""
        
        name_, val_ = self.stat_eval_func(measurement=_measurement,
                                          background_simulation=_background_simulation, signal_simulation=_signal_simulation,
                                          background_name="b", signal_name=f"{mu}s_{{{num} \\ \\mathrm{{GeV}}}}")
        
        return f"{name_} = {round(val_, 3)}"
    
    def plot(self):
        
        _stat_eval_sig_bac_string = ""
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        h = Hist(bins=self.num_buttons, hist_range=self.hist_range)
        
        colors = ["green", "orange", "yellow", "cyan", "orangered",
                  "magenta", "red", "brown", "dodgerblue", "silver", "lawngreen"]
        
        for i, (num, color) in enumerate(zip(self.mc_sig_name_list, colors)):
            if self.check_buttons[i].isChecked():
                temp_hist = deepcopy(self.histograms[f"mc_bac_sig_{num}"])
                _mu = 1.0
                if self._visible_mu:
                    self.change_mu_visibility()
                    _mu = float(self.spinboxes[i].text().replace(",", "."))
                    temp_hist.data["mc_sig"] *= _mu
                
                temp_hist.draw(pass_name=["data", "mc_bac", f"mc_sig"],
                               color=["black", "royalblue", color],
                               label=["Messung", r"Untergrund: $ZZ, Z\gamma$",
                                      f"$m_H = ${num} GeV"],
                               alpha=[1, 1, 0.5],
                               figure=self.figure, ax=ax)
                _sig_bac_temp_str = self._stat_eval_bac_sig(temp_hist, num=num, mu=f"{_mu}")
                _stat_eval_sig_bac_string += f"{_sig_bac_temp_str},  " if self._visible_stat_eval else ""
                del temp_hist
        else:
            self.histograms["mc_bac"].draw(pass_name=["data", "mc_bac"],
                                           color=["black", "royalblue"],
                                           label=["Messung", r"Untergrund: $ZZ, Z\gamma$"],
                                           figure=self.figure, ax=ax)
        
        if (self.lower_border_, self.upper_border_) != self.hist_range and self._visible_stat_eval:
            ax.vlines([self.lower_border_, self.upper_border_], 0, 100, label="Begrenzung", ls="--")
        
        if self._visible_stat_eval:
            ax.text(0.5, 0.7, _stat_eval_sig_bac_string[:-3], size=14, bbox=dict(facecolor='white', alpha=0.75),
                    horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
        
        pass_x, pass_y = np.array([]), np.array([])
        y_err_calc_func = HistHelper.calc_errors_alternative_near_simplified
        for i in range(len(h.x_range)):
            if self.list_num[i] != 0:
                pass_x, pass_y = np.append(pass_x, h.x_range[i]), np.append(pass_y, self.list_num[i])
        
        label_name = "Messung" if np.sum(len(self.list_num)) > 0.0 else None
        
        ax.errorbar(pass_x, pass_y, xerr=0, yerr=y_err_calc_func(pass_y), fmt="o", marker="o",
                    color="black", label=label_name)
        
        TSH.legend_without_duplicate_labels(ax)
        
        ax.set_xticks([70 + (181 - 70) / self.num_buttons * i for i in range(self.num_buttons + 1)])
        ax.set_yticks([0 + 2 * i for i in range(self.y_plot_limits[1])])
        ax.set_xlim(*self.hist_range)
        ax.set_ylim(*self.y_plot_limits)
        ax.set_xlabel(r"$m_{4\ell}$ in GeV")
        ax.set_ylabel(r"Bineinträge")
        
        self.cursor = Cursor(ax, lw=2)
        
        def onclick(event):
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
            ax.hlines(12, 120, 130)
            self.canvas.draw()
        
        self.figure.canvas.mpl_connect('button_press_event', onclick)
        
        self.canvas.draw()


def WidgetHiggs(*args, **kwargs):
    # app = QtWidgets.QApplication.instance()
    # if app is None:
    #     app = QtWidgets.QApplication(sys.argv)
    # else:
    #     print('QApplication instance already exists: %s' % str(app))
    
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    
    window = PlotHiggs(*args, **kwargs)
    window.setWindowFlags(window.windowFlags() | QtCore.Qt.CustomizeWindowHint)
    window.setWindowFlags(window.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)
    window.show()
    app.exec_()


if __name__ == "__main__":
    WidgetHiggs(b_num=37, info=[["2012"], ["B", "C"]],
                mc_dir_="../../data/mc_aftH",
                mc_other_dir="../../data/other_mc/dXXX/mc_aftH",
                f_width=1500, f_height=700, f_bottom=0.09)
