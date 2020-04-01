from functools import partial

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QShortcut
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class Connect(object):
    
    @staticmethod
    def button_and_text(num_, button_, func_, **kwargs):
        if not isinstance(num_, list) and not isinstance(button_, list):
            button_.clicked.connect(partial(func_, **kwargs))
        else:
            for i in range(num_):
                button_[i].clicked.connect(partial(func_, i))
    
    @staticmethod
    def button_and_plot(num, button_name, func_name):
        for i in range(num):
            button_name[i].clicked.connect(func_name)
    
    @staticmethod
    def button_and_function(object_, func_):
        object_.clicked.connect(func_)
    
    @staticmethod
    def submenu_and_function(instance_, object_, func_, shortcut_=None):
        if shortcut_ is not None:
            s_ = QShortcut(QtGui.QKeySequence(shortcut_), instance_)
            s_.activated.connect(func_)
        object_.triggered.connect(func_)


class Setting(object):
    _transparent_text_style_sheet = "background-color: transparent; " \
                                    "border: transparent;" \
                                    "text-align: center; "
    
    @staticmethod
    def __prebuild(object_, str_, operation_):
        
        if not isinstance(object_, list):
            getattr(object_, operation_)(str_)
        if isinstance(object_, list):
            str_ = str_ if isinstance(str_, list) else [str_ for _ in object_]
            for obj_, t_ in zip(object_, str_):
                getattr(obj_, operation_)(t_)
    
    @staticmethod
    def text_style(object_, stylesheet_=None):
        stylesheet_ = stylesheet_ if stylesheet_ is not None else Setting._transparent_text_style_sheet
        Setting.__prebuild(object_=object_, str_=stylesheet_, operation_="setStyleSheet")
    
    @staticmethod
    def text_alignment(object_, loc_):
        pos_ = {"center": QtCore.Qt.AlignCenter, "right": QtCore.Qt.AlignRight, "left": QtCore.Qt.AlignLeft}
        object_ = object_ if isinstance(object_, list) else [object_]
        for obj_ in object_:
            obj_.setAlignment(pos_[loc_])
    
    @staticmethod
    def text(object_, text_):
        Setting.__prebuild(object_=object_, str_=text_, operation_="setText")
    
    @staticmethod
    def name(object_, name_):
        Setting.__prebuild(object_=object_, str_=name_, operation_="setObjectName")
    
    @staticmethod
    def geometry(object_, x_, y_, w_=100, h_=30, s_=10):
        object_ = object_ if isinstance(object_, list) else [object_]
        for i, obj_ in enumerate(object_):
            obj_.setGeometry(QtCore.QRect(int(x_ + (s_ + w_) * i), int(y_), int(w_), int(h_)))
    
    @staticmethod
    def button(object_, text_, x_, y_, w_=100, h_=30, s_=10,
               name_=None, func_=None, transparent_=False):
        Setting.geometry(object_, x_=x_, y_=y_, w_=w_, h_=h_, s_=s_)
        Setting.text(object_=object_, text_=text_)
        Setting.name(object_=object_, name_=name_) if name_ is not None else None
        if func_ is not None:
            func_ = func_ if isinstance(func_, list) else [func_]
            for f_ in func_:
                Connect.button_and_function(object_, f_)
        if transparent_:
            option_ = 'background-color: transparent; border: transparent;'
            Setting.__prebuild(object_, option_, 'setStyleSheet')
    
    @staticmethod
    def menu_bullet(instance_, name_, menu_bar_):
        if not isinstance(name_, list):
            reg_ = QtWidgets.QMenu(name_, instance_)
            instance_.setMenuBar(menu_bar_)
            return reg_
        if isinstance(name_, list):
            return tuple([
                Setting.menu_bullet(instance_=instance_, name_=n_,
                                    menu_bar_=menu_bar_)
                for n_ in name_
            ])
    
    @staticmethod
    def submenu_bullet(instance_, name_, menu_bullet_, menu_bar_):
        if not isinstance(name_, list):
            temp_obj_ = QtWidgets.QAction(name_, instance_)
            menu_bullet_.addAction(temp_obj_)
            menu_bar_.addAction(menu_bullet_.menuAction())
            return temp_obj_
        if isinstance(name_, list):
            return tuple([
                Setting.submenu_bullet(instance_=instance_, name_=n_,
                                       menu_bullet_=menu_bullet_, menu_bar_=menu_bar_)
                for n_ in name_
            ])


class Convert(object):
    
    @staticmethod
    def latex_to_qtpixmap(str_, fontsize_=None):
        figure_ = Figure()
        figure_.set_canvas(FigureCanvasAgg(figure_))
        figure_.patch.set_facecolor("none")
        ax_ = figure_.add_axes([0, 0, 1, 1])
        ax_.axis('off')
        text_ = ax_.text(0, 0, str_, ha='left', va='bottom', fontsize=fontsize_)
        
        f_w_, f_h_ = figure_.get_size_inches()
        bbox_f_ = figure_.get_window_extent(figure_.canvas.get_renderer())
        bbox_t_ = text_.get_window_extent(figure_.canvas.get_renderer())
        
        figure_.set_size_inches((bbox_t_.width / bbox_f_.width) * f_w_,
                                (bbox_t_.height / bbox_f_.height) * f_h_)

        buf, size = figure_.canvas.print_to_buffer()
        return QtGui.QImage.rgbSwapped(QtGui.QImage(buf, *size, QtGui.QImage.Format_ARGB32))


class ButtomCustomizeHelper(object):
    connect = Connect
    setting = Setting
    convert = Convert
