import sys
from functools import partial

import numpy as np
from PyQt5 import QtWidgets, QtCore

from .helper.ButtonCustomizeHelper import ButtomCustomizeHelper as BCH
from .helper.CorePlot import PlotInitWidget
from ..histogramm.HistHelper import HistHelper


class Wuerfel(PlotInitWidget):
    _own_pdf_func = None
    _own_measurment_pdf_func = None
    
    def __init__(self, b_num=6, *args, own_func=None,
                 own_measurment_pdf_func=None, **kwargs):
        super(Wuerfel, self).__init__(b_num=b_num, *args, **kwargs)
        
        self._own_pdf_func = own_func
        self._own_measurment_pdf_func = own_measurment_pdf_func
        
        BCH.connect.button_and_plot(num=self.b_num,
                                    button_name=self.n_plus,
                                    func_name=self.plot)
        BCH.connect.button_and_plot(num=self.b_num,
                                    button_name=self.n_minus,
                                    func_name=self.plot)
        
        self.additional_menu_ui()
        self.additional_buttons()
        self.HistUiCall()
        self.N_text.x()
        BCH.setting.text(self.N_text, "N simuliert")
        BCH.setting.text_alignment(self.N_text, "left")
        BCH.setting.geometry(self.N_text,
                             x_=self.N_text.x() - int(self.b_space * 3.25),
                             y_=self.N_text.y(),
                             w_=self.N_text.width(),
                             h_=self.N_text.height())
    
    def additional_menu_ui(self):
        self.menu_zusatz = BCH.setting.menu_bullet(instance_=self,
                                                   name_="Zusatz",
                                                   menu_bar_=self.menubar)
        self.m_zusatz_1, self.m_zusatz_2, self.m_zusatz_3 = BCH.setting.submenu_bullet(
            instance_=self, name_=["one", "two", "three"],
            menu_bullet_=self.menu_zusatz,
            menu_bar_=self.menubar
        )
        
        BCH.connect.submenu_and_function(
            instance_=self, object_=self.m_zusatz_1,
            func_=self._get_simulation_button)
        
        BCH.connect.submenu_and_function(
            instance_=self, object_=self.m_zusatz_2,
            func_=self._get_draw_normed
        )
        
        BCH.connect.submenu_and_function(
            instance_=self, object_=self.m_zusatz_3,
            func_=self._get_measurment_button
        )
        
        self._visible_m_zusatz_1 = False
        self._draw_normed = False
        
        self._visible_measurment = False
        self.measurment_array = np.zeros(len(self.list_num))
    
    def save_operation(self):
        self.figure.savefig(self.file_name)
        self.canvas.draw()
    
    def additional_buttons(self):
        
        x_off = self.f_width + self.b_space
        y_off = self.f_start_y
        
        self.fill_simulation_one = QtWidgets.QPushButton(self)
        BCH.setting.button(self.fill_simulation_one, text_="simuliere 1",
                           x_=x_off, y_=y_off + 90, w_=180, h_=self.b_height,
                           func_=[self.add_oc1r, self.plot]
                           )
        self.fill_simulation_n = QtWidgets.QPushButton(self)
        BCH.setting.button(self.fill_simulation_n, text_="simuliere 100",
                           x_=x_off, h_=self.b_height, w_=180,
                           y_=y_off + 90 + self.b_height + self.b_space,
                           func_=[partial(self.add_ocnr, 100), self.plot]
                           )
        self.check_simulation_mean = QtWidgets.QCheckBox(self)
        self.check_simulation_sigma_all = QtWidgets.QCheckBox(self)
        self.check_simulation_sigma_individual = QtWidgets.QCheckBox(self)
        BCH.setting.button(self.check_simulation_mean, text_="Mittelwert",
                           x_=x_off, h_=self.b_height, w_=180,
                           y_=y_off + 90 + 2 * self.b_height + 2 * self.b_space,
                           func_=self.plot)
        
        BCH.setting.button(self.check_simulation_sigma_all, text_="Standardabweichung (alle)",
                           x_=x_off, h_=self.b_height, w_=180,
                           y_=y_off + 90 + 3 * self.b_height + 2 * self.b_space,
                           func_=self.plot)
        
        BCH.setting.button(self.check_simulation_sigma_individual, text_="Standardabweichung (einzeln)",
                           x_=x_off, h_=self.b_height, w_=180,
                           y_=y_off + 90 + 4 * self.b_height + 2 * self.b_space,
                           func_=self.plot)
        
        self.get_measurment = QtWidgets.QPushButton(self)
        BCH.setting.button(self.get_measurment, text_="Messe: 10 mal",
                           x_=x_off, h_=self.b_height, w_=150,
                           y_=y_off + 90 + 5 * self.b_height + 2 * self.b_space,
                           func_=[partial(self._get_measurment_data, num=10),
                                  self.plot,
                                  partial(self._get_measurment_table, init_call=False)]
                           )
        self.get_measurment.setVisible(False)
        
        self.check_simulation_mean.setVisible(False)
        self.check_simulation_sigma_all.setVisible(False)
        self.check_simulation_sigma_individual.setVisible(False)
        
        self.fill_simulation_one.setVisible(False)
        self.fill_simulation_n.setVisible(False)
        
        self._get_measurment_table(init_call=True)
    
    def _get_measurment_button(self):
        self._visible_measurment = not self._visible_measurment
        self.get_measurment.setVisible(self._visible_measurment)
        BCH.setting.text(self.m_zusatz_1, "on three" if not self._visible_m_zusatz_1 else "off three")
        
        for (table_left, table_right) in zip(self._measurment_n_text, self._got_measurment_n_text):
            table_left.setVisible(self._visible_measurment)
            table_right.setVisible(self._visible_measurment)
        self._measurment_info_text.setVisible(self._visible_measurment)
        
        if not self._visible_measurment:
            self.measurment_array = np.zeros(len(self.list_num))
            for table_right in self._got_measurment_n_text:
                BCH.setting.text(table_right, "0")
                BCH.setting.text_style(table_right)
                BCH.setting.text_alignment(table_right, "left")
    
    def _get_measurment_table(self, init_call=True):
        x_start = self.f_width + self.b_space
        y_start = self.f_start_y + 90 + 6 * self.b_height + 2 * self.b_space
        if init_call:
            self._measurment_n_text = [QtWidgets.QTextBrowser(self) for _ in range(self.b_num)]
            self._got_measurment_n_text = [QtWidgets.QTextBrowser(self) for _ in range(self.b_num)]
            
            self._measurment_info_text = QtWidgets.QTextBrowser(self)
            BCH.setting.text(self._measurment_info_text, "Messungszusammenfassung:")
            BCH.setting.geometry(self._measurment_info_text, x_=x_start,
                                 y_=y_start + (0.125) * int(self.b_height),
                                 w_=200, h_=30)
            BCH.setting.text_style(self._measurment_info_text)
            self._measurment_info_text.setVisible(False)
            
            BCH.setting.text_alignment(self._measurment_info_text, "left")
            
            for i, (n_text, num) in enumerate(zip(self._measurment_n_text, self._got_measurment_n_text)):
                BCH.setting.text(n_text, f"{i + 1}:")
                BCH.setting.geometry(n_text, x_=x_start,
                                     y_=y_start + (i + 1) * int(self.b_height * 0.85),
                                     w_=30, h_=30)
                BCH.setting.text(num, f"0")
                BCH.setting.geometry(num, x_=x_start + 32,
                                     y_=y_start + (i + 1) * int(self.b_height * 0.85),
                                     w_=90, h_=30)
                
                BCH.setting.text_alignment(n_text, "right")
                BCH.setting.text_alignment(num, "left")
                BCH.setting.text_style(n_text)
                BCH.setting.text_style(num)
                n_text.setVisible(False)
                num.setVisible(False)
        
        if not init_call:
            for num_, text_ in zip(self.measurment_array, self._got_measurment_n_text):
                text_.setText(f"{int(num_)}")
                BCH.setting.text_style(text_)
                BCH.setting.text_alignment(text_, "left")
    
    def _get_measurment_data(self, num=10):
        for _ in range(num):
            self.add_oc1r(measurment=True)
    
    def _get_simulation_button(self):
        self._visible_m_zusatz_1 = not self._visible_m_zusatz_1
        
        self.fill_simulation_one.setVisible(self._visible_m_zusatz_1)
        self.fill_simulation_n.setVisible(self._visible_m_zusatz_1)
        self.check_simulation_sigma_all.setVisible(self._visible_m_zusatz_1)
        self.check_simulation_mean.setVisible(self._visible_m_zusatz_1)
        self.check_simulation_sigma_individual.setVisible(self._visible_m_zusatz_1)
        
        BCH.setting.text(self.m_zusatz_1, "on one" if not self._visible_m_zusatz_1 else "off one")
    
    def _get_draw_normed(self):
        self._draw_normed = not self._draw_normed
        BCH.setting.text(self.m_zusatz_2, "on two" if not self._draw_normed else "off two")
        self.plot()
    
    def add_oc1r(self, measurment=False):
        if not measurment:
            try:
                rand_ = self._own_pdf_func()
            except:
                rand_ = np.random.choice(np.arange(1, self.b_num + 1, 1), 1)[0]
            
            self.n_text[rand_ - 1].setText(str(int(self.n_text[rand_ - 1].toPlainText()) + 1))
            self.list_num[rand_ - 1] += 1
        if measurment:
            try:
                rand_ = self._own_measurment_pdf_func()
            except:
                rand_ = np.random.choice(np.arange(1, self.b_num + 1, 1), 1)[0]
            self.measurment_array[rand_ - 1] += 1
    
    def add_ocnr(self, num=100):
        for i in range(num):
            self.add_oc1r()
    
    def plot(self):
        
        def norm(array):
            size_ = 1.0 if np.sum(self.measurment_array) == 0.0 else np.sum(self.measurment_array)
            norm_ = np.sum(np.array(array)) if any(it > 0 for it in array) else 1.0
            return (size_ / norm_) * np.array(array)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        plot_y_array = self.list_num
        if self._draw_normed:
            plot_y_array = norm(self.list_num)
        if not self._draw_normed and self._visible_measurment:
            plot_y_array = norm(self.list_num)
        
        # if not self.check_simulation_sigma_individual.isChecked():
        ax.bar(np.arange(1, self.b_num + 1, 1),
               plot_y_array,
               color="royalblue", alpha=0.5,
               label="Simulation")
        
        ax.set_xlim(0.5, 0.5 + self.b_num)
        ax.set_xticks(np.arange(1, self.b_num + 1, 1))
        
        if self.check_simulation_mean.isChecked():
            ax.hlines(np.sum(plot_y_array) / self.b_num, -1, self.b_num + 2,
                      ls="--", label="Mittelwert")
        
        if self.check_simulation_sigma_all.isChecked():
            m_ = np.sum(plot_y_array) / self.b_num
            std_ = np.sqrt((1 / (self.b_num - 1)) * np.sum((plot_y_array - m_) ** 2))
            
            ax.fill_between([-1, self.b_num + 2],
                            [m_ + std_, m_ + std_],
                            [m_ - std_, m_ - std_],
                            color="green", alpha=0.25,
                            label="Standardabweichung (alle)")
        if self.check_simulation_sigma_individual.isChecked():
            y_err_ = HistHelper.calc_errors_poisson_near_cont(self.list_num)
            if self._draw_normed or self._visible_measurment:
                size_ = 1. if not self._visible_measurment else np.sum(self.measurment_array)
                y_err_ = (size_ / np.sum(self.list_num)) * np.array(y_err_)
            ax.errorbar(np.arange(1, self.b_num + 1, 1),
                        plot_y_array,
                        fmt="bx", marker="",
                        ecolor="royalblue", alpha=1.0,
                        yerr=y_err_,
                        capsize=int((0.125 / 2. * self.f_width) / self.b_num),
                        # error_kw=dict(ecolor='royalblue', alpha=1.0),
                        label="Standardabweichung (einzeln)")
        
        if self._visible_measurment:
            ax.errorbar(np.arange(1, self.b_num + 1, 1),
                        self.measurment_array,
                        xerr=0,
                        yerr=HistHelper.calc_errors_poisson_near_cont(self.measurment_array),
                        fmt="ko", lw=2,
                        label="Messung")
        
        ax.set_ylabel("Bineinträge")
        ax.set_xlabel("n")
        ax.legend()
        
        self.canvas.draw()


def WidgetWuerfel(*args, **kwargs):
    # app = QtWidgets.QApplication.instance()
    # if app is None:
    #     app = QtWidgets.QApplication(sys.argv)
    # else:
    #     print('QApplication instance already exists: %s' % str(app))
    #
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    
    window = Wuerfel(*args, **kwargs)
    window.setWindowFlags(window.windowFlags() | QtCore.Qt.CustomizeWindowHint)
    window.setWindowFlags(window.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)
    window.show()
    app.exec_()


if __name__ == "__main__":
    WidgetWuerfel(f_width=700, f_height=700, f_bottom=0.09, f_left=0.09, b_num=6, own_func=None,
                  own_measurment_pdf_func=None, g_width=900)
