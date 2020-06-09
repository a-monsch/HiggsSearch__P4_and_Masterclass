import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow
import yaml

from include.widget.helper.ButtonCustomizeHelper import ButtomCustomizeHelper as BCH


class CoreWidget(QMainWindow):
    
    def __init__(self, b_num=30, b_coord=None, glob_geom=None, info_text_array=None, draw_h_line=True, language="EN"):
        super(CoreWidget, self).__init__()
        self.b_num = b_num
        self.list_num = np.zeros(self.b_num)
        self.info_text_array = info_text_array
        self.draw_h_line = draw_h_line
        self.global_geom = glob_geom
        self.b_coord = b_coord
        self.file_name = None
        self.la = language
        self.td = self.load_language_dict()
        self.menuBarInitUi()
    
    def load_language_dict(self):
        with open('./include/widget/lang/gui.yml', 'r') as outfile:
            my_dict = yaml.full_load(outfile)
        return my_dict
    
    def menuBarInitUi(self):
        self.menubar = QtWidgets.QMenuBar(self)
        
        self.menu_file = BCH.setting.menu_bullet(instance_=self,
                                                 name_=self.td["file"][self.la],
                                                 menu_bar_=self.menubar)
        self.menu_file_save_pic, self.menu_file_saveas_pic = BCH.setting.submenu_bullet(
            instance_=self,
            name_=[self.td["save plot"][self.la], self.td["save plot as"][self.la]],
            menu_bullet_=self.menu_file,
            menu_bar_=self.menubar)
        
        self.menu_file_open_meas, self.menu_file_save_meas, self.menu_file_saveas_meas = BCH.setting.submenu_bullet(
            instance_=self, menu_bullet_=self.menu_file, menu_bar_=self.menubar,
            name_=[self.td["load measurement"][self.la], self.td["save measurement"][self.la], self.td["save measurement as"][self.la]])
        
        BCH.connect.submenu_and_function(instance_=self, object_=self.menu_file_save_pic, func_=self.ActionSaveFile_pic)
        BCH.connect.submenu_and_function(instance_=self, object_=self.menu_file_saveas_pic, func_=self.ActionSaveFileAs_pic)
        
        BCH.connect.submenu_and_function(instance_=self, object_=self.menu_file_save_meas, func_=self.ActionSaveFile_meas)
        BCH.connect.submenu_and_function(instance_=self, object_=self.menu_file_saveas_meas, func_=self.ActionSaveFileAs_meas)
        BCH.connect.submenu_and_function(instance_=self, object_=self.menu_file_open_meas, func_=self.ActionOpenFile_meas)
    
    def ActionSaveFileAs_pic(self):
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "*.png",
                                                   "All Files (*);;*.png;;*.jpg;;*.pdf)", options=opt)
        if file_name:
            print(file_name)
            self.file_name_pic = file_name
            self.save_operation_pic()
    
    def ActionSaveFile_pic(self):
        self.ActionSaveFileAs_pic() if self.file_name is None else self.save_operation_pic()
    
    def ActionSaveFileAs_meas(self):
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "*.csv",
                                                   "All Files (*);;*.csv)", options=opt)
        if file_name:
            print(file_name)
            self.file_name_meas = file_name
            self.save_operation_meas()
    
    def ActionSaveFile_meas(self):
        self.ActionSaveFileAs_meas() if self.file_name is None else self.save_operation_meas()
    
    def ActionOpenFile_meas(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;CSV Files (*.csv)",
                                                   options=options)
        if file_name:
            self.file_name_meas = file_name
            self.open_operation_meas()
            del self.file_name_meas
    
    def save_operation_meas(self):
        pass
    
    def open_operation_meas(self):
        pass
    
    def save_operation_pic(self):
        pass
    
    def set_b_coord(self, item=None):
        if self.b_coord is None:
            self.b_coord = {"x_s": 100, "y_s": 320, "w": 30, "h": 30, "s": 10}
        if item is not None:
            self.b_coord = item
    
    def set_global_geom(self, item=None):
        if self.global_geom is None:
            self.global_geom = (100, 100, 1000, 600)
        if item is not None:
            self.global_geom = item
        
        self.setGeometry(*self.global_geom)
    
    def InitCoreUi(self):
        self.set_global_geom()
        self.set_b_coord()
        self.InitButtonsUi()
    
    def InitButtonsUi(self):
        self.n_text = self.create_text_blocks(num=self.b_num)
        BCH.setting.text(object_=self.n_text, text_="0")
        BCH.setting.text_alignment(self.n_text, "center")
        BCH.setting.geometry(object_=self.n_text, x_=self.b_coord["x_s"], w_=self.b_coord["w"],
                             y_=self.b_coord["y_s"])
        
        self.n_plus = self.create_buttons(num=self.b_num)
        BCH.setting.text(object_=self.n_plus, text_="+")
        BCH.setting.geometry(object_=self.n_plus, x_=self.b_coord["x_s"], w_=self.b_coord["w"],
                             y_=self.b_coord["y_s"] + self.b_coord["h"] + self.b_coord["s"])
        
        self.n_minus = self.create_buttons(num=self.b_num)
        BCH.setting.text(object_=self.n_minus, text_="-")
        BCH.setting.geometry(object_=self.n_minus, x_=self.b_coord["x_s"], w_=self.b_coord["w"],
                             y_=self.b_coord["y_s"] + 2 * (self.b_coord["h"] + self.b_coord["s"]))
        
        BCH.connect.button_and_text(num_=self.b_num, button_=self.n_plus, func_=self.add_on_click)
        BCH.connect.button_and_text(num_=self.b_num, button_=self.n_minus, func_=self.del_on_click)
        
        self.N_text = QtWidgets.QTextBrowser(self)
        BCH.setting.button(object_=self.N_text, text_="N", transparent_=True,
                           x_=self.b_coord["x_s"] - self.b_coord["w"] / 5. - self.b_coord["s"],
                           y_=self.b_coord["y_s"], w_=self.b_coord["w"], h_=self.b_coord["h"])
        
        self.draw_lines_for_table()
    
    def draw_lines_for_table(self):
        nb = self.b_num
        self.v_lines = []
        for i in range(nb):
            exec(f"self.v_line_{i} = QtWidgets.QFrame(self)")
            exec(f"self.v_lines.append(self.v_line_{i})")
        
        for i, v_line in enumerate(self.v_lines):
            left_x = self.n_minus[i].x()
            left_y = self.b_coord["y_s"] - 2 * self.n_plus[i].height() + self.b_coord["s"]
            width_ = 1
            height_ = 2 * self.n_plus[i].height() + 2 * self.b_coord["s"]
            v_line.setGeometry(QtCore.QRect(left_x, left_y, width_, height_))
            v_line.setFrameShape(QtWidgets.QFrame.VLine)
            v_line.setFrameShadow(False)
        
        if self.draw_h_line:
            left_x = self.n_minus[0].x() - self.b_coord["s"] - self.b_coord["w"] / 2.
            left_y = ((self.b_coord["y_s"]) + (self.b_coord["y_s"] - self.b_coord["h"] - self.b_coord["s"])) / 2.
            width_ = self.n_minus[-1].x() + 2 * self.b_coord["s"] - self.b_coord["x_s"] + self.b_coord["w"]
            height_ = self.b_coord["s"]
            
            exec(f"self.h_line = QtWidgets.QFrame(self)")
            self.h_line.setGeometry(QtCore.QRect(int(left_x), int(left_y), int(width_), int(height_)))
            self.h_line.setFrameShape(QtWidgets.QFrame.HLine)
            self.h_line.setFrameShadow(False)
            self.h_line.setObjectName("h_line")
    
    def create_buttons(self, num):
        return [QtWidgets.QPushButton(self) for _ in range(num)]
    
    def create_text_blocks(self, num, transparent=True):
        buttons = [QtWidgets.QTextBrowser(self) for _ in range(num)]
        if transparent:
            for button in buttons:
                BCH.setting.text_style(button)
        return buttons
    
    def add_on_click(self, num=0):
        self.n_text[num].setText(str(int(self.n_text[num].toPlainText()) + 1))
        self.list_num[num] += 1
        BCH.setting.text_alignment(self.n_text, "center")
    
    def del_on_click(self, num=0):
        if int(self.n_text[num].toPlainText()) - 1 >= 0:
            self.n_text[num].setText(str(int(self.n_text[num].toPlainText()) - 1))
            self.list_num[num] -= 1
            BCH.setting.text_alignment(self.n_text, "center")
