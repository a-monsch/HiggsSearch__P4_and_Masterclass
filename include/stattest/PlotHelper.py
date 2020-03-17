import matplotlib.pyplot as plt
import numpy as np


class PlotHelper(object):
    """
    Collection a functions that are useful for the plots of the StatTest class.
    """
    
    @staticmethod
    def get_error_points_from_iminuit_obj(m_obj_, var_name_, sigma_=(1.0,)):
        """
        Calculates the uncertainties using the provided iminuit object.

        :param m_obj_: iminuit object
        :param var_name_: "str"
        :param sigma_: tuple
        :return: ndarray
        """
        m_obj_temp = m_obj_
        opt_val_ = m_obj_temp.values[var_name_]
        temp_array_ = list()
        temp_array_.append(opt_val_)
        
        for s in sigma_:
            m_obj_temp.minos(sigma=s)
            temp_array_.append(opt_val_ + abs(m_obj_temp.merrors[(var_name_, 1.0)]))
            temp_array_.append(opt_val_ - abs(m_obj_temp.merrors[(var_name_, -1.0)]))
        
        del m_obj_temp
        return np.sort(np.array(temp_array_))
    
    @staticmethod
    def get_x_and_optional_nll(nll_vec_obj_=None, p_=(), np_=100, extend_=False):
        """
        Creates an array for a scan of the given size and optional nagative
        logarithm of likelihood.

        :param nll_vec_obj_: np.vectorized function
        :param p_: tuple
                   (mu - sigma_left, sigma, mu + sigma_right)
        :param np_: int
                    numper of points used
        :param extend_: bool
                        if p_ contains 3 points this option try to extend to next sigma values.
        :return: ndarray or tuple
                 1D array or tuple of 1D arrays containing data with "float" type.
        """
        f_ = 1.5 if extend_ else 0.5
        x_r_ = np.append(np.array(p_), np.linspace(p_[0] - f_ * (p_[1] - p_[0]), p_[-1] + f_ * (p_[-1] - p_[-2]), np_))
        x_ = np.sort(np.unique(x_r_))
        return x_ if nll_vec_obj_ is None else (x_, nll_vec_obj_(x_))
    
    @staticmethod
    def get_error_points_from_nll(x_, nll_, sigma_=(1.0,)):
        """
        Calculates the uncertainties using a given negative
        logarithm of likelihood object as an np.vectorized function.

        :param x_: ndarray
                   1D array containing data with "float" type.
        :param nll_: ndarray
                     1D array containing data with "float" type.
        :param sigma_: tuple
        :return: ndarray
                 1D array containing data with "float" type.
        """
        temp_array_ = list()
        temp_array_.append(x_[np.argmin(nll_)])
        
        l_p = nll_[:np.argmin(nll_)]
        r_p = nll_[np.argmin(nll_):]
        
        for s in sigma_:
            temp_array_.append(x_[np.argmin(abs(l_p - s ** 2))])
            temp_array_.append(x_[np.argmin(abs(r_p - s ** 2)) + len(l_p)])
        
        return np.sort(np.array(temp_array_))
    
    @staticmethod
    def contour_own(ax_obj_, m_obj_, nll_obj_, var1_, var2_,
                    x_label_=r"$m_{\rm{H}}$", y_label_=r"$\alpha_{\rm{s}}$", sigma_=(1.0, 2.0)):
        """
        Draws the contour of the two-dimensional scan using the
        given np.vectorizded negative logarithm of the likelihood function.

        :param ax_obj_: matplotlib.pyplot.axes
        :param m_obj_: iminuit object
        :param nll_obj_: vectorized function
        :param var1_: str
        :param var2_: str
        :param x_label_: str
        :param y_label_: str
        :param sigma_: tuple

        """
        init_px_ = PlotHelper.get_error_points_from_iminuit_obj(m_obj_=m_obj_, var_name_=var1_)
        init_py_ = PlotHelper.get_error_points_from_iminuit_obj(m_obj_=m_obj_, var_name_=var2_)
        err_bar_ = ax_obj_.errorbar(init_px_[1], init_py_[1],
                                    xerr=[[init_px_[1] - init_px_[0]], [init_px_[2] - init_px_[1]]],
                                    yerr=[[init_py_[1] - init_py_[0]], [init_py_[2] - init_py_[1]]],
                                    fmt="kx", capsize=5, elinewidth=0.5, markeredgewidth=0.5, label=r"$\rm{fit}$")
        
        @np.vectorize
        def nll_vec(alpha, mass): return nll_obj_(alpha=alpha, mass=mass)
        
        @np.vectorize
        def nll_alpha_only(alpha): return nll_obj_(alpha=alpha, mass=init_py_[1])
        
        @np.vectorize
        def nll_mass_only(mass): return nll_obj_(alpha=init_px_[1], mass=mass)
        
        t_a_, t_nll_a_ = PlotHelper.get_x_and_optional_nll(nll_vec_obj_=nll_alpha_only, p_=init_px_, extend_=True, np_=101)
        t_m_, t_nll_m_ = PlotHelper.get_x_and_optional_nll(nll_vec_obj_=nll_mass_only, p_=init_py_, extend_=True, np_=101)
        
        x_ax_set = PlotHelper.get_error_points_from_nll(x_=t_a_, nll_=t_nll_a_ - np.amin(t_nll_a_), sigma_=sigma_)
        y_ax_set = PlotHelper.get_error_points_from_nll(x_=t_m_, nll_=t_nll_m_ - np.amin(t_nll_m_), sigma_=sigma_)
        
        sigma_labels_ = [f"${i}\\sigma$" for i in range(-int(sigma_[-1]), int(sigma_[-1] + 1))]
        ax_obj_.set_xlim(x_ax_set[0] - 0.05, x_ax_set[-1] + 0.15)
        ax_obj_.set_ylim(y_ax_set[0] - 0.35, y_ax_set[-1] + 0.35)
        ax_obj_.set_xticks(x_ax_set)
        ax_obj_.set_xticklabels(sigma_labels_)
        ax_obj_.set_yticks(y_ax_set)
        ax_obj_.set_yticklabels(sigma_labels_)
        ax_obj_.set_ylabel(y_label_)
        ax_obj_.set_xlabel(x_label_)
        
        a_plot_ = PlotHelper.get_x_and_optional_nll(p_=x_ax_set, np_=25)
        m_plot_ = PlotHelper.get_x_and_optional_nll(p_=y_ax_set, np_=25)
        
        X, Y = np.meshgrid(np.array(a_plot_), np.array(m_plot_))
        Z = nll_vec(X, Y)
        Z = Z - np.amin(Z)
        
        cs = ax_obj_.contourf(X, Y, Z, levels=[0, 1, 4], colors=('red', 'blue', 'white'), alpha=0.125)
        
        proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in cs.collections]
        proxy.append(err_bar_)
        
        ax_obj_.legend(proxy, [r"$\sigma$", r"$2\sigma$", r"$\rm{fit}$", "t"])
    
    @staticmethod
    def plot_nll_scan_main_window(ax_obj_, x_, y_, z_,
                                  x_lim_=(0.0, 0.8), y_lim_=(0.0, 40.0),
                                  x_label=r"$\alpha_{\rm{s}}$"):
        """
        Draws the main window of the likelihood scan.
        Also draws the significances on the left of the y-axis.

        :param ax_obj_: matplotlib.pyplot.axes
        :param x_: ndarray
                   1D array containing data with "float" type.
        :param y_:ndarray
                  1D array containing data with "float" type.
        :param z_: float
        :param x_lim_: tuple
        :param y_lim_: tuple
        :param x_label: str
        """
        ax_obj_.plot(x_, y_, label=r"$- 2 \ln \left( \frac{\mathcal{L}}{\mathcal{L}_{\rm{min}}} \right)-\rm{profile}$")
        
        ax_obj_.set_xlabel(x_label)
        ax_obj_.set_ylabel(r"$- 2 \ln \left( \frac{\mathcal{L}}{\mathcal{L}_{\rm{min}}} \right)$")
        
        ax_obj_.legend(loc="upper left")
        
        ax_obj_.annotate("{}".format(round(z_, 3)) + r"$\sigma$",
                         xy=(0.0, z_ ** 2), xytext=(0.14, 14),
                         arrowprops=dict(width=1.0, color="black"))
        
        ax_obj_.hlines([4, 9, 16, 25], 0, 0.05, color="red")
        
        ax_obj_.text(0.0525, 3.7, r"$2 \sigma$", color="red")
        ax_obj_.text(0.0525, 8.7, r"$3 \sigma$", color="red")
        ax_obj_.text(0.0525, 15.7, r"$4 \sigma$", color="red")
        ax_obj_.text(0.0525, 24.7, r"$5 \sigma$", color="red")
        ax_obj_.set_xlim(*x_lim_)
        ax_obj_.set_ylim(*y_lim_)
    
    @staticmethod
    def plot_nll_scan_inside_window(ax_obj_, x_, y_, x_ticks_,
                                    label_=None, x_label=None, y_label=None):
        """
        Draws the main window of the likelihood scan.

        :param ax_obj_: matplotlib.pyplot.axes
        :param x_: ndarray
                   1D array containing data with "float" type.
        :param y_: ndarray
                   1D array containing data with "float" type.
        :param x_ticks_: tuple
        :param label_: str
        :param x_label: str
        :param y_label: str
        """
        ax_obj_.plot(x_, y_, label=label_)
        
        ax_obj_.grid(alpha=1)
        
        ax_obj_.set_xlabel(x_label)
        ax_obj_.set_ylabel(y_label)
        
        ax_r_t_ = abs(x_ticks_[0] - x_ticks_[-1]) * 0.1
        ax_range_ = (x_ticks_[0] - ax_r_t_, x_ticks_[-1] + ax_r_t_)
        ax_obj_.set_xlim(*ax_range_)
        ax_obj_.set_ylim(0.0, 1.5)
        
        ax_obj_.set_yticks([0.0, 1.0])
        ax_obj_.set_xticks(x_ticks_)
        
        plt.setp(ax_obj_.get_xticklabels(), visible=True)
        plt.setp(ax_obj_.get_yticklabels(), visible=True)
    
    @staticmethod
    def contour_iminuit(ax_obj_, m_obj_, var1_, var2_,
                        y_label=r"$m_{\rm{H}}$", x_label=r"$\alpha_{\rm{s}}$"):
        """
        Draws the contour of the two-dimensional scan using the
        given iminuit object

        :param ax_obj_: matplotlib.pyplot.axes
        :param m_obj_: iminuit object
        :param var1_: str
        :param var2_: str
        :param y_label: str
        :param x_label: str
        """
        x_ax_set = PlotHelper.get_error_points_from_iminuit_obj(m_obj_=m_obj_, var_name_=var1_, sigma_=(1.0, 2.0))
        y_ax_set = PlotHelper.get_error_points_from_iminuit_obj(m_obj_=m_obj_, var_name_=var2_, sigma_=(1.0, 2.0))
        
        px_ = x_ax_set[1:-1]
        py_ = y_ax_set[1:-1]
        err_bar_ = ax_obj_.errorbar(px_[1], py_[1], xerr=[[px_[1] - px_[0]], [px_[2] - px_[1]]],
                                    yerr=[[py_[1] - py_[0]], [py_[2] - py_[1]]],
                                    fmt="kx", capsize=5, elinewidth=0.5, markeredgewidth=0.5, label=r"$\rm{fit}$")
        
        ax_obj_.set_xlim(x_ax_set[0] - 0.05, x_ax_set[-1] + 0.15)
        ax_obj_.set_ylim(y_ax_set[0] - 0.35, y_ax_set[-1] + 0.35)
        ax_obj_.set_xticks(x_ax_set)
        ax_obj_.set_xticklabels([r"$-2\sigma$", r"$-\sigma$", r"$0\sigma$", r"$\sigma$", r"$2\sigma$"])
        ax_obj_.set_yticks(y_ax_set)
        ax_obj_.set_yticklabels([r"$-2\sigma$", r"$-\sigma$", r"$0\sigma$", r"$\sigma$", r"$2\sigma$"])
        ax_obj_.set_ylabel(y_label)
        ax_obj_.set_xlabel(x_label)
        
        x_, y_, val_ = m_obj_.contour(var1_, var2_, bins=25, bound=3, args=None, subtract_min=True)
        
        cs = ax_obj_.contourf(x_, y_, val_, levels=[0, 1, 4], colors=('red', 'blue', 'white'), alpha=0.125)
        
        proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in cs.collections]
        proxy.append(err_bar_)
        
        ax_obj_.legend(proxy, [r"$\sigma$", r"$2\sigma$", r"$\rm{fit}$", "t"])
    
    @staticmethod
    def get_p0_sigma_lines(x_, ax_obj_, max_sigma_=4):
        """
        Auxiliary function for the drawing of the significance of the p0 estimate.

        :param x_: ndarray
                   1D array containing data with "float" type.
        :param ax_obj_: matplotlib.pyplot.axes
        :param max_sigma_: int
        """
        s1, s2, s3 = 0.317310507863 / 2., 0.045500263896 / 2., 0.002699796063 / 2.
        s4, s5, s6 = 0.000063342484 / 2., 0.000000573303 / 2., 0.000000001973 / 2.
        s_ = [s1, s2, s3, s4, s5, s6]
        ax_obj_.hlines(s_[0:max_sigma_], x_[0], x_[-1], colors="red", alpha=0.75)
        for i, sigma in enumerate(s_[0:max_sigma_]):
            if i == 0:
                ax_obj_.text(x_[1], sigma * 1.1, f"$\\sigma$", color="red")
                continue
            ax_obj_.text(x_[1], sigma * 1.1, f"${i}\\sigma$", color="red")
        ax_obj_.set_yscale("log")
        ax_obj_.set_xlim(x_[0], x_[-1])
        ax_obj_.set_ylim(s_[max_sigma_ - 1] * 0.5, 1.25)
