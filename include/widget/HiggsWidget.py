import math
import os
import sys

import mplcursors
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QLineEdit
from matplotlib.widgets import Cursor
from tqdm import tqdm

from .helper.ButtonCustomizeHelper import ButtomCustomizeHelper as BCH
from .helper.CorePlot import PlotInitWidget
from ..RandomHelper import ToSortHelper as TSH
from ..histogramm.Hist import Hist
from ..histogramm.HistHelper import HistHelper


class PlotHiggs(PlotInitWidget):
    
    def __init__(self, *args,
                 b_num=37, hist_range=(70, 181), mc_dir_="./data/mc_aftH", ru_dir_="./data/ru_aftH",
                 mc_other_dir="./other_mc_sig_mc/dXXX/mc_aftH",
                 **kwargs):
        self.mc_other_dir = mc_other_dir
        self.ru_dir_, self.mc_dir_ = ru_dir_, mc_dir_
        self.num_buttons, self.hist_range = b_num, hist_range
        
        self.other_sig_num_list = [115, 120, 122, 124, 128, 130, 135, 140, 145, 150]
        self.name_list = [115, 120, 122, 124, 125, 128, 130, 135, 140, 145, 150]
        
        super(PlotHiggs, self).__init__(*args,
                                        b_num=self.num_buttons,
                                        draw_h_line=True,
                                        **kwargs)
        BCH.connect.button_and_plot(num=self.num_buttons, button_name=self.n_plus, func_name=self.plot)
        BCH.connect.button_and_plot(num=self.num_buttons, button_name=self.n_minus, func_name=self.plot)
        
        # check -> c_
        self.c_other_mc = False
        
        self.c_nll = False
        self.l_b_, self.h_b_ = self.hist_range
        
        self.used_Ui()
        
        self.histograms = {}
        self.load_hists()
    
    def used_Ui(self):
        self.HistUiCall()
        self.menu_ansicht = BCH.setting.menu_bullet(instance_=self,
                                                    name_="Ansicht",
                                                    menu_bar_=self.menubar)
        self.menu_ansicht_draw_mc, self.menu_ansicht_print_nll = BCH.setting.submenu_bullet(
            instance_=self, name_=["MC Simulationen anzeigen",
                                   "NLL Berechnung anzeigen"],
            menu_bullet_=self.menu_ansicht, menu_bar_=self.menubar)
        BCH.connect.submenu_and_function(instance_=self, object_=self.menu_ansicht_draw_mc,
                                         func_=self.change_other_mc_on)
        BCH.connect.submenu_and_function(instance_=self, object_=self.menu_ansicht_print_nll,
                                         func_=self.change_nll_option)
        
        self.initHistButtonUi()
    
    def change_other_mc_on(self):
        self.c_other_mc = not self.c_other_mc
        for button in self.check_buttons:
            button.setVisible(self.c_other_mc)
        
        BCH.setting.text(self.menu_ansicht_draw_mc,
                         "MC Simulationen anzeigen" if not self.c_other_mc else "MC Simulationen ausblenden")
    
    def change_nll_option(self):
        self.c_nll = not self.c_nll
        BCH.setting.text(self.menu_ansicht_print_nll,
                         "NLL Berechnung anzeigen" if not self.c_nll else "NLL Berechnung ausblenden")
        self.nll_set_border_window_() if self.c_nll else self.plot()
    
    def nll_set_border_window_(self):
        wgt_nll_border = QDialog(self)
        BCH.setting.geometry(wgt_nll_border, 100, 100, 200, 140)
        
        wgt_nll_border.info_txt_box_init = QtWidgets.QTextBrowser(wgt_nll_border)
        BCH.setting.button(wgt_nll_border.info_txt_box_init, text_="Betrachtungsintervall setzen.",
                           x_=20, y_=10, w_=200, h_=30, transparent_=True)
        
        wgt_nll_border.info_txt_box_1 = QtWidgets.QTextBrowser(wgt_nll_border)
        BCH.setting.button(wgt_nll_border.info_txt_box_1, text_="untere Grenze", x_=20, y_=45, w_=80, h_=30, transparent_=True)
        
        wgt_nll_border.info_txt_box_2 = QtWidgets.QTextBrowser(wgt_nll_border)
        BCH.setting.button(wgt_nll_border.info_txt_box_2, text_="obere Grenze", x_=20, y_=90, w_=80, h_=30, transparent_=True)
        
        wgt_nll_border.set_txt_box_1 = QLineEdit(wgt_nll_border)
        wgt_nll_border.set_txt_box_1.move(100, 45)
        wgt_nll_border.set_txt_box_1.resize(80, 20)
        BCH.setting.text(wgt_nll_border.set_txt_box_1, f"{self.l_b_}")
        
        wgt_nll_border.set_txt_box_2 = QLineEdit(wgt_nll_border)
        wgt_nll_border.set_txt_box_2.move(100, 90)
        wgt_nll_border.set_txt_box_2.resize(80, 20)
        BCH.setting.text(wgt_nll_border.set_txt_box_2, f"{self.h_b_}")
        
        def __on_click():
            self.l_b_ = int(wgt_nll_border.set_txt_box_1.text())
            self.h_b_ = int(wgt_nll_border.set_txt_box_2.text())
            wgt_nll_border.close()
            self.plot()
        
        wgt_nll_border.set_button = QtWidgets.QPushButton("setzen", wgt_nll_border)
        BCH.setting.geometry(wgt_nll_border.set_button, x_=140, y_=115, w_=40, h_=20)
        wgt_nll_border.set_button.clicked.connect(__on_click)
        
        wgt_nll_border.exec_()
    
    def initHistButtonUi(self):
        self.check_buttons = []
        for name in self.name_list:
            exec(f"self.check_button_{name} = QtWidgets.QCheckBox(self)")
            exec(f"self.check_buttons.append(self.check_button_{name})")
        for i, (button_, name) in enumerate(zip(self.check_buttons, self.name_list)):
            BCH.setting.button(object_=button_, x_=self.f_width + self.b_space, y_=self.f_start_y + 30 * i, w_=225, h_=30,
                               text_=f"Higgsmasse: {str(name)} GeV", func_=self.plot)
            button_.setVisible(self.c_other_mc)
    
    def save_operation(self):
        self.figure.savefig(self.file_name)
        self.canvas.draw()
    
    def load_hists(self):
        
        h = Hist(bins=self.num_buttons, hist_range=self.hist_range)
        h.set_empty_bin(["data", "mc_bac"])
        h.fill_hist_from_dir(col_="mass_4l", dir_=self.mc_dir_, info=[["2012"], ["A-D"]])
        self.histograms["mc_bac"] = h
        
        self.y_plot_limits = (0, math.ceil(np.amax(h.data["mc_bac"])))
        
        h = Hist(bins=self.num_buttons, hist_range=self.hist_range)
        h.set_empty_bin(["data", "mc_bac", "mc_sig_125"])
        h.fill_hist_from_dir(col_="mass_4l", dir_=self.mc_dir_, info=[["2012"], ["A-D"]],
                             name_sig_mc_=["mc_sig_125" for _ in os.listdir(self.mc_dir_)])
        self.histograms["mc_bac_sig_125"] = h
        # return None
        for i, num in tqdm(enumerate(self.other_sig_num_list), total=len(self.other_sig_num_list)):
            if f"mc_bac_sig_{num}" not in self.histograms.keys():
                h = Hist(bins=self.num_buttons, hist_range=self.hist_range, signal_mass_mc=num)
                file_list = TSH.mixed_file_list(TSH.get_other_mc_dir(num, dir_=self.mc_other_dir),
                                                main_dir_=self.mc_dir_)
                n_mc_sig = [f"mc_sig_{num}" for _ in file_list]
                h.set_empty_bin(["data", "mc_bac", f"mc_sig_{num}"])
                h.fill_hist_from_dir(col_="mass_4l", dir_=None,
                                     info=[["2012"], ["A-D"]], file_list_=file_list, name_sig_mc_=n_mc_sig)
                self.histograms[f"mc_bac_sig_{num}"] = h
                del h
    
    def f_hist_nll(self, hist_object, num=""):
        w_ = hist_object.bin_width
        filter_lower_ = (hist_object.x_range - w_ / 2. > self.l_b_)
        filter_higher_ = (hist_object.x_range + w_ / 2. < self.h_b_)
        filter_ = filter_lower_ & filter_higher_
        
        bac_ = hist_object.data[[item for item in hist_object.data if "mc_bac" in item][0]][filter_]
        sig_ = hist_object.data[[item for item in hist_object.data if "mc_sig" in item][0]][filter_]
        
        data_ = self.list_num[filter_]
        
        # TODO: in zwei Zeilen und eigene Funktion
        bac_nll_ = 0
        bac_sig_nll_ = 0
        for i, d_ in enumerate(data_):
            if d_ > 0:
                bac_nll_ += bac_[i] - d_ + d_ * np.log(d_ / bac_[i])
                bac_sig_nll_ += (bac_[i] + sig_[i]) - d_ + d_ * np.log(d_ / (bac_[i] + sig_[i]))
        
        nlr_ = 2 * (bac_nll_ - bac_sig_nll_)
        q0_ = np.round(nlr_, 3) if nlr_ > 0 else 0
        
        s_ = f"$ 2 \\ln \\left( \\dfrac{{ \\mathcal{{L}}_{{ B + S={num} }} }}{{ \\mathcal{{L}}_{{B}} }}  \\right) = $ {q0_}"
        
        return s_
    
    def plot(self):
        
        nll_string_ = ""
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        h = Hist(bins=self.num_buttons, hist_range=self.hist_range)
        
        colors = ["green", "orange", "yellow", "cyan", "orangered",
                  "magenta", "red", "brown", "dodgerblue", "silver", "lawngreen"]
        
        for i, (num, color) in enumerate(zip(self.name_list, colors)):
            if self.check_buttons[i].isChecked():
                self.histograms[f"mc_bac_sig_{num}"].draw(pass_name=["data", "mc_bac", f"mc_sig_{num}"],
                                                          color=["black", "royalblue", color],
                                                          label=["Messung", r"Untergrund: $ZZ, Z\gamma$",
                                                                 f"$m_H = ${num} GeV"],
                                                          alpha=[1, 1, 0.5],
                                                          figure=self.figure, ax=ax)
                nll_string_ += f"{self.f_hist_nll(self.histograms[f'mc_bac_sig_{num}'], num=num)},  " if self.c_nll else ""
        
        else:
            self.histograms["mc_bac"].draw(pass_name=["data", "mc_bac"],
                                           color=["black", "royalblue"],
                                           label=["Messung", r"Untergrund: $ZZ, Z\gamma$"],
                                           figure=self.figure, ax=ax)
        
        if (self.l_b_, self.h_b_) != self.hist_range and self.c_nll:
            ax.vlines([self.l_b_, self.h_b_], 0, 100, label="Begrenzung", ls="--")
        
        if self.c_nll:
            ax.text(0.5, 0.8,
                    r"q_0", size=14, bbox=dict(facecolor='white', alpha=0.75),
                    horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
            ax.text(0.5, 0.7,
                    nll_string_[:-3], size=14, bbox=dict(facecolor='white', alpha=0.75),
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
        
        mplcursors.cursor()
        
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
    WidgetHiggs(b_num=37,
                mc_dir_="../data/mc_aftH",
                ru_dir_="../data/ru_aftH",
                mc_other_dir="../other_mc_sig_mc/dXXX/mc_aftH",
                f_width=1500, f_height=700, f_bottom=0.09)
