import sys
from functools import partial

import numpy as np
import scipy.stats as scst
from PyQt5 import QtWidgets, QtCore, QtGui

from include.histogramm.HistHelper import HistHelper
from include.widget.helper.ButtonCustomizeHelper import ButtomCustomizeHelper as BCH
from include.widget.helper.CorePlot import PlotInitWidget


class Wuerfel(PlotInitWidget):
    
    def __init__(self, *args, b_num=6, calc_class=None, **kwargs):
        super(Wuerfel, self).__init__(b_num=b_num, *args, **kwargs)
        
        self.CalcCls = _FallBack(self) if calc_class is None else calc_class
        
        BCH.connect.button_and_plot(num=self.b_num, button_name=self.n_plus, func_name=self.plot)
        BCH.connect.button_and_plot(num=self.b_num, button_name=self.n_minus, func_name=self.plot)
        
        self.additional_menu_ui()
        self.additional_buttons()
        self.HistUiCall()
        
        BCH.setting.text(self.N_text, "N sim.")
        BCH.setting.text_alignment(self.N_text, "left")
        BCH.setting.geometry(self.N_text, x_=self.N_text.x() - int(self.b_space * 3.25),
                             y_=self.N_text.y(), w_=self.N_text.width(), h_=self.N_text.height())
    
    def additional_menu_ui(self):
        BCH.setting.menu_bullet = partial(BCH.setting.menu_bullet, instance_=self, menu_bar_=self.menubar)
        BCH.setting.submenu_bullet = partial(BCH.setting.submenu_bullet, instance_=self, menu_bar_=self.menubar)
        BCH.connect.submenu_and_function = partial(BCH.connect.submenu_and_function, instance_=self)
        
        self.menu_zusatz = BCH.setting.menu_bullet(name_="Zusatz")
        self.m_simulation, self.m_norm, self.m_measurement, self.m_stat_calc = BCH.setting.submenu_bullet(
            name_=["Simulation ein", "Simulation Normierung ein", "Messung ein", "Statistische Bewertung ein"],
            menu_bullet_=self.menu_zusatz)
        
        BCH.connect.submenu_and_function(object_=self.m_simulation, func_=self._get_simulation_button)
        BCH.connect.submenu_and_function(object_=self.m_norm, func_=self._get_draw_normed)
        BCH.connect.submenu_and_function(object_=self.m_measurement, func_=self._get_measurment_button)
        
        self._visible_simulation = False
        self._draw_normed = False
        
        self._visible_measurement = False
        self.measurement_array = np.zeros(len(self.list_num))
        
        BCH.connect.submenu_and_function(object_=self.m_stat_calc, func_=self._get_statistical_calculation)
        self._visible_stat_calc = False
    
    def save_operation(self):
        self.figure.savefig(self.file_name)
        self.canvas.draw()
    
    def additional_buttons(self):
        
        x_off = self.f_width + self.b_space
        y_off = self.f_start_y
        BCH.setting.button = partial(BCH.setting.button, x_=x_off, w_=180, h_=self.b_height)
        
        self.fill_simulation_one = QtWidgets.QPushButton(self)
        BCH.setting.button(self.fill_simulation_one, text_="simuliere 1", y_=y_off + 90, func_=[self.add_oc1r, self.plot])
        
        self.fill_simulation_n = QtWidgets.QPushButton(self)
        BCH.setting.button(self.fill_simulation_n, text_="simuliere 100", y_=y_off + 90 + self.b_height + self.b_space,
                           func_=[partial(self.add_ocnr, 100), self.plot])
        
        self.check_simulation_mean = QtWidgets.QCheckBox(self)
        self.check_simulation_sigma_all = QtWidgets.QCheckBox(self)
        self.check_simulation_sigma_individual = QtWidgets.QCheckBox(self)
        
        BCH.setting.button(self.check_simulation_mean, text_="Mittelwert",
                           y_=y_off + 90 + 2 * self.b_height + 2 * self.b_space, func_=self.plot)
        
        BCH.setting.button(self.check_simulation_sigma_all, text_="Standardabweichung (alle)",
                           y_=y_off + 90 + 3 * self.b_height + 2 * self.b_space, func_=self.plot)
        
        BCH.setting.button(self.check_simulation_sigma_individual, text_="Standardabweichung (einzeln)",
                           y_=y_off + 90 + 4 * self.b_height + 2 * self.b_space, func_=self.plot)
        
        self.get_measurment = QtWidgets.QPushButton(self)
        BCH.setting.button(self.get_measurment, text_="Messe: 10 mal", y_=y_off + 90 + 5 * self.b_height + 2 * self.b_space,
                           func_=[partial(self._get_measurment_data, num=10), self.plot,
                                  partial(self._get_measurment_table, init_call=False)])
        
        self.get_measurment.setVisible(False)
        
        self.check_simulation_mean.setVisible(False)
        self.check_simulation_sigma_all.setVisible(False)
        self.check_simulation_sigma_individual.setVisible(False)
        
        self.fill_simulation_one.setVisible(False)
        self.fill_simulation_n.setVisible(False)
        
        self._get_measurment_table(init_call=True)
        
        self._stat_text = QtWidgets.QLabel(self)
        BCH.setting.geometry(self._stat_text, x_=self._measurement_info_text_table_left_[-1].x(), w_=180, h_=30,
                             y_=self._measurement_info_text_table_left_[-1].y() + self.b_height + self.b_space)
        BCH.setting.text_style(self._stat_text)
        self._stat_text.setPixmap(QtGui.QPixmap(BCH.convert.latex_to_qtpixmap(self._get_rendered_stat())))
        self._stat_text.setVisible(False)
    
    def _get_rendered_stat(self):
        _measurement = self.measurement_array
        _expectation = self.norm_and_rescale_sim(self.list_num)
        name_, value_ = self.CalcCls.own_stat_evaluation(_measurement, _expectation)
        return f"${name_} = $ {value_:.2e}"
    
    def _get_statistical_calculation(self):
        self._visible_stat_calc = not self._visible_stat_calc
        self._stat_text.setVisible(self._visible_stat_calc)
        BCH.setting.text(self.m_stat_calc,
                         "Statistische Bewertung ein" if not self._visible_stat_calc else "Statistische Bewertung aus")
    
    def _get_measurment_button(self):
        
        self._visible_measurement = not self._visible_measurement
        
        self.get_measurment.setVisible(self._visible_measurement)
        BCH.setting.text(self.m_simulation, "Messung ein" if not self._visible_simulation else "Messung aus")
        
        for (table_left, table_right) in zip(self._measurement_info_text_table_left_, self._measurement_n_text_table_right_):
            table_left.setVisible(self._visible_measurement)
            table_right.setVisible(self._visible_measurement)
        self._measurement_info_text.setVisible(self._visible_measurement)
        
        if not self._visible_measurement:
            self.measurement_array = np.zeros(len(self.list_num))
            for table_right in self._measurement_n_text_table_right_:
                BCH.setting.text(table_right, "0")
                BCH.setting.text_style(table_right)
                BCH.setting.text_alignment(table_right, "left")
    
    def _get_measurment_table(self, init_call=True):
        x_start = self.f_width + self.b_space
        y_start = self.f_start_y + 90 + 6 * self.b_height + 2 * self.b_space
        
        BCH.setting.geometry = partial(BCH.setting.geometry, h_=30)
        
        if init_call:
            
            self._measurement_info_text_table_left_ = [QtWidgets.QTextBrowser(self) for _ in range(self.b_num)]
            self._measurement_n_text_table_right_ = [QtWidgets.QTextBrowser(self) for _ in range(self.b_num)]
            
            self._measurement_info_text = QtWidgets.QTextBrowser(self)
            BCH.setting.text(self._measurement_info_text, "Messungszusammenfassung:")
            BCH.setting.geometry(self._measurement_info_text, x_=x_start,
                                 y_=y_start + (0.125) * int(self.b_height), w_=200)
            BCH.setting.text_style(self._measurement_info_text)
            self._measurement_info_text.setVisible(False)
            BCH.setting.text_alignment(self._measurement_info_text, "left")
            
            for i, (n_text, num) in enumerate(zip(self._measurement_info_text_table_left_,
                                                  self._measurement_n_text_table_right_)):
                BCH.setting.text(n_text, f"{i + 1}:")
                BCH.setting.geometry(n_text, x_=x_start, y_=y_start + (i + 1) * int(self.b_height * 0.85), w_=30)
                BCH.setting.text(num, f"0")
                BCH.setting.geometry(num, x_=x_start + 32, y_=y_start + (i + 1) * int(self.b_height * 0.85), w_=90)
                
                BCH.setting.text_alignment(n_text, "right")
                BCH.setting.text_alignment(num, "left")
                BCH.setting.text_style(n_text)
                BCH.setting.text_style(num)
                
                n_text.setVisible(False)
                num.setVisible(False)
        
        if not init_call:
            for num_, text_ in zip(self.measurement_array, self._measurement_n_text_table_right_):
                text_.setText(f"{int(num_)}")
                BCH.setting.text_style(text_)
                BCH.setting.text_alignment(text_, "left")
    
    def _get_measurment_data(self, num=10):
        for _ in range(num):
            self.add_oc1r(measurment=True)
    
    def _get_simulation_button(self):
        
        self._visible_simulation = not self._visible_simulation
        
        self.fill_simulation_one.setVisible(self._visible_simulation)
        self.fill_simulation_n.setVisible(self._visible_simulation)
        self.check_simulation_sigma_all.setVisible(self._visible_simulation)
        self.check_simulation_mean.setVisible(self._visible_simulation)
        self.check_simulation_sigma_individual.setVisible(self._visible_simulation)
        
        BCH.setting.text(self.m_simulation, "Simulation ein" if not self._visible_simulation else "Simulation aus")
    
    def _get_draw_normed(self):
        self._draw_normed = not self._draw_normed
        BCH.setting.text(self.m_norm, "Simulation Normierung ein" if not self._draw_normed else "Simulation Normierung aus")
        self.plot()
    
    def add_oc1r(self, measurment=False):
        if not measurment:
            rand_ = self.CalcCls.own_pdf_simulation_one()
            self.n_text[rand_ - 1].setText(str(int(self.n_text[rand_ - 1].toPlainText()) + 1))
            self.list_num[rand_ - 1] += 1
        if measurment:
            rand_ = self.CalcCls.own_pdf_measurement_one()
            self.measurement_array[rand_ - 1] += 1
    
    def add_ocnr(self, num=100):
        for i in range(num):
            self.add_oc1r()
    
    def norm_and_rescale_sim(self, array):
        rescale_fac_ = self.CalcCls.own_measurement_scale(self.measurement_array) if self._visible_measurement else 1.0
        return rescale_fac_ * self.CalcCls.own_norm(array)
    
    def plot(self):
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_xlim(0.5, 0.5 + self.b_num)
        ax.set_xticks(np.arange(1, self.b_num + 1, 1))
        ax.set_ylabel("Bineinträge")
        ax.set_xlabel("n")
        
        self._stat_text.setPixmap(QtGui.QPixmap(BCH.convert.latex_to_qtpixmap(self._get_rendered_stat())))
        
        plot_y_array = self.list_num
        if self._draw_normed:
            plot_y_array = self.norm_and_rescale_sim(self.list_num)
        if not self._draw_normed and self._visible_measurement:
            plot_y_array = self.norm_and_rescale_sim(self.list_num)
        
        ax.bar(np.arange(1, self.b_num + 1, 1), plot_y_array,
               color="royalblue", alpha=0.5, label="Simulation")
        
        if self.check_simulation_mean.isChecked():
            ax.hlines(self.CalcCls.own_mean(plot_y_array), -1, self.b_num + 2, ls="--", label="Mittelwert")
        
        if self.check_simulation_sigma_all.isChecked():
            m_ = self.CalcCls.own_mean(plot_y_array)
            std_ = self.CalcCls.own_std(plot_y_array)
            ax.fill_between([-1, self.b_num + 2], [m_ + std_, m_ + std_], [m_ - std_, m_ - std_],
                            color="green", alpha=0.25, label="Standardabweichung (alle)")
        
        if self.check_simulation_sigma_individual.isChecked():
            y_err_ = self.CalcCls.own_individual_std(self.list_num)
            
            if self._draw_normed or self._visible_measurement:
                size_ = 1. if not self._visible_measurement else (
                    np.sum(self.measurement_array) if np.sum(self.measurement_array) != 0.0 else 1.0)
                y_err_ = (size_ / np.sum(self.list_num)) * np.array(y_err_)
            
            ax.errorbar(np.arange(1, self.b_num + 1, 1), plot_y_array, yerr=y_err_, fmt="bx", marker="",
                        ecolor="royalblue", alpha=1.0, capsize=int((0.125 / 2. * self.f_width) / self.b_num),
                        label="Standardabweichung (einzeln)")
        
        if self._visible_measurement:
            ax.errorbar(np.arange(1, self.b_num + 1, 1), self.measurement_array, xerr=0,
                        yerr=HistHelper.calc_errors_poisson_near_cont(self.measurement_array),
                        fmt="ko", lw=2, label="Messung")
        
        ax.legend()
        self.canvas.draw()


class _FallBack(object):
    
    def __init__(self, obj_: Wuerfel):
        self.w = obj_
    
    def own_pdf_simulation_one(self):
        return np.random.choice(np.arange(1, self.w.b_num + 1, 1), 1)[0]
    
    def own_pdf_measurement_one(self):
        return self.own_pdf_simulation_one()
    
    def own_norm(self, array):
        norm_ = np.sum(np.array(array)) if any(it > 0 for it in array) else 1.0
        return (1. / norm_) * np.array(array)
    
    def own_measurement_scale(self, measurement):
        return 1.0 if np.sum(measurement) == 0.0 else np.sum(measurement)
    
    def own_mean(self, array):
        return np.mean(array)
    
    def own_std(self, array):
        return np.std(array)
    
    def own_individual_std(self, array):
        return np.array(HistHelper.calc_errors_poisson_near_cont(array))
    
    def own_stat_evaluation(self, measurement, simulation, name="p_0"):
        if np.sum(measurement) == 0.0:
            return name, 9999.0
        chi2_ = sum((1.0 / s) * (m - s) ** 2 for m, s in zip(measurement, simulation) if float(s) != 0.0)
        p0_ = 1.0 - scst.chi2.cdf(chi2_, df=len(measurement) - 1)
        return name, p0_


def WidgetWuerfel(*args, **kwargs):
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    
    window = Wuerfel(*args, **kwargs)
    window.setWindowFlags(window.windowFlags() | QtCore.Qt.CustomizeWindowHint)
    window.setWindowFlags(window.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)
    window.show()
    app.exec_()
