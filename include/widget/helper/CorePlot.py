from PyQt5 import QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from include.widget.helper.CoreWidget import CoreWidget


class PlotInitWidget(CoreWidget):
    
    def __init__(self,
                 *args,
                 b_x_start=75, b_height=30, b_space=10, b_width=None,
                 f_left=0.04, f_top=0.98, f_bottom=0.17, f_height=350, f_width=1600, f_start_y=30,
                 g_width=1900, g_height=None,
                 canvas_geom=None,
                 draw_h_line=False,
                 **kwargs):
        
        super(PlotInitWidget, self).__init__(*args, **kwargs)
        
        self.draw_h_line = draw_h_line
        self.canvas_geom = canvas_geom
        self.f_start_y = f_start_y
        self.f_left, = f_left,
        self.f_right = 1 - self.f_left
        self.f_top, self.f_bottom, self.f_height, self.f_width = f_top, f_bottom, f_height, f_width
        self.b_x_start, self.b_height, self.b_space = b_x_start, b_height, b_space
        self.g_height, self.g_width = g_height, g_width
        
        if self.g_height is None and not self.draw_h_line:
            self.g_height = self.f_height + 3 * self.b_height + 4 * self.b_space
        
        if self.g_height is None and self.draw_h_line:
            self.g_height = self.f_height + 4 * self.b_height + 5 * self.b_space
        
        self.b_width = b_width
        if self.b_width is None:
            self.b_width = int((self.f_width - 2 * self.b_x_start - (self.b_num - 1) * self.b_space) / self.b_num)
        
        self.set_b_coord(item={"x_s": self.b_x_start, "y_s": self.f_height + self.b_space,
                               "w": self.b_width, "h": self.b_height, "s": self.b_space})
        self.set_global_geom(item=(0, 0, self.g_width, self.g_height))
        
        self.InitCoreUi()
    
    def HistUiCall(self):
        self.set_canvas_geom()
        self.init_HistUi()
        self.figure.subplots_adjust(left=self.f_left, right=self.f_right, top=self.f_top, bottom=self.f_bottom)
    
    def set_canvas_geom(self, item=None):
        if self.canvas_geom is None:
            self.canvas_geom = self.calc_canvas_geom()
        if item is not None:
            self.canvas_geom = item
    
    def calc_canvas_geom(self):
        x_axis_length = (self.b_num) * self.b_width + (self.b_num) * self.b_space
        tot_f_length = x_axis_length * 1 / (1 - (1 - self.f_right) - self.f_left)
        return (self.b_x_start - tot_f_length * self.f_left, self.f_start_y, tot_f_length, self.f_height - self.f_start_y)
    
    def init_HistUi(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.figure.set_facecolor("none")
        self.canvas.setParent(self)
        self.canvas_geom = tuple(int(it) for it in self.canvas_geom)
        self.canvas.setGeometry(QtCore.QRect(*self.canvas_geom))
