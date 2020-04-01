import math

import numpy as np
import scipy.special as scsp
import scipy.stats as scst


class SignalFunction(object):
    """
    Class which implements possible functions for
    bell-shaped signal description and their string
    representation (LaTeX) in the software package kafe2.
    """
    _x_mean = 0.0
    _x_latex = "m_{{4\\ell}}"
    _x_mean_latex = " - \\bar{{m}}_{{4\\ell}}"
    _method_latex_names = {"gauss": "\\mathcal{{G}}", "cauchy": "\\mathcal{{C}}", "DSCB": "DSCB", "SSCB": "SSCB",
                           "voigt": "\\mathcal{{V}}"}
    _method_latex_param_names = {"sigma": "\\sigma", "beta": "\\beta", "mu": "\\mu", "gamma": "\\gamma",
                                 "x": _x_latex}
    _method_latex_func_expr = {"gauss": None, "cauchy": None,
                               "voigt": "\\mathcal{{G}}(" + ",".join(
                                   [_method_latex_param_names["x"],
                                    _method_latex_param_names["sigma"],
                                    _method_latex_param_names["mu"]]
                               ) + ")* \\mathcal{{C}}(" + ",".join(
                                   [_method_latex_param_names["x"],
                                    _method_latex_param_names["gamma"],
                                    _method_latex_param_names["mu"]]
                               ) + ")",
                               "DSCB": None, "SSCB": None}
    
    def __init__(self, x_mean=0.0):
        self._x_mean = x_mean
        self._x_mean_latex = self._x_mean_latex if x_mean != 0.0 else ""
        SignalFunction.set_global_x_mean(self._x_mean, self._x_latex, self._x_mean_latex)
    
    def _get_param_names(self, name, with_x=False):
        """
        Iterates through the function arguments and converts them to LaTeX format.

        :param name: str
        :param with_x: bool
        :return: dict
        """
        _m = self._method_latex_param_names
        if with_x:
            return {k: (_m[k] if k in _m.keys() else k)
                    for k in getattr(self, name).__code__.co_varnames}
        if not with_x:
            return {k: (_m[k] if k in _m.keys() else k)
                    for k in getattr(self, name).__code__.co_varnames if k != "x"}
    
    def pretty_dict(self, method_name, with_x=True):
        """
        Creates a dict from functions name, expression and
        parameters (LaTeX) that is used in kafe2.

        :param grade: int
        :return: dict
        """
        return {'x_name': self._x_latex,  # not required in newer kafe2 versions -> set in func_param
                'func_name': self._method_latex_names[method_name],
                'func_expr': self._method_latex_func_expr[method_name],
                'func_param': self._get_param_names(method_name, with_x=with_x)}
    
    def set_local_x_mean(self, value=None, x_latex=None, x_mean_latex=None):
        """
        Sets the mean value of x and its string representation localy.

        :param value: float
        :param x_latex: str
        :param x_mean_latex: str
        """
        self._x_mean = value if value is not None else self._x_mean
        self._x_latex = "m_{{4\\ell}}" if x_latex is None else x_latex
        self._x_mean_latex = " - \\bar{{m}}_{{4\\ell}}" if x_mean_latex is None else x_mean_latex
        
        SignalFunction.set_global_x_mean(self._x_mean, self._x_latex, self._x_mean_latex)
    
    @staticmethod
    def set_global_x_mean(value=None, x_latex=None, x_mean_latex=None):
        """
        Sets the mean value of x and its string representation globaly.

        :param value: float
        :param x_latex: str
        :param x_mean_latex: str
        """
        SignalFunction._x_mean = value if value is not None else SignalFunction._x_mean
        SignalFunction._x_latex = "{x}-\\bar{{m}}_{{4\\ell}}" if x_latex is None else x_latex
        SignalFunction._x_mean_latex = " - \\bar{{m}}_{{4\\ell}}" if x_mean_latex is None else x_mean_latex
    
    @staticmethod
    def gauss(x, sigma=2.027, mu=124.807):
        x = x - SignalFunction._x_mean
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)
    
    @staticmethod
    def cauchy(x, gamma=0.824, mu=124.484):
        x = x - SignalFunction._x_mean
        return (1. / np.pi) * (gamma / (gamma ** 2 + (x - mu) ** 2))
    
    @staticmethod
    def SSCB(x, sigma=1.2, mu=125.0, beta=1.1, m=1.2):
        x = x - SignalFunction._x_mean
        return scst.crystalball.pdf(x, beta=beta, m=m, loc=mu, scale=sigma)
    
    @staticmethod
    @np.vectorize
    def DSCB(x, sigma=1.0, mu=125.0, alpha_l=0.5, alpha_r=0.5, n_l=1.1, n_r=1.1):
        t = (x - mu) / sigma
        lf = ((alpha_l / n_l) * ((n_l / alpha_l) - alpha_l - t)) ** (-n_l)
        rf = ((alpha_r / n_r) * ((n_r / alpha_r) - alpha_r + t)) ** (-n_r)
        
        n1 = np.sqrt(np.pi / 2.) * (math.erf(alpha_r / np.sqrt(2.)) + math.erf(alpha_l / np.sqrt(2.)))
        n2 = np.exp(-0.5 * alpha_l ** 2) * ((alpha_l / n_l) ** (-n_l)) * ((n_l / alpha_l) ** (-n_l + 1)) * (
                1. / (n_l - 1))
        n3 = np.exp(-0.5 * alpha_r ** 2) * ((alpha_r / n_r) ** (-n_r)) * ((n_r / alpha_r) ** (-n_r + 1)) * (
                1. / (n_r - 1))
        
        norm = 1. / (n1 + n2 + n3)
        norm = norm / sigma
        
        if -alpha_l <= t <= alpha_r:
            return norm * np.exp(-0.5 * t ** 2)
        if t < -alpha_l:
            return norm * np.exp(-0.5 * alpha_l ** 2) * lf
        if t > alpha_r:
            return norm * np.exp(-0.5 * alpha_r ** 2) * rf
    
    @staticmethod
    @np.vectorize
    def voigt(x, sigma=1.0, mu=125.0, gamma=1.0, ):
        z = ((x - mu) + 1j * gamma) / (np.sqrt(2) * sigma)
        f1_ = np.real(scsp.wofz(z).real)
        f2_ = sigma * np.sqrt(2 * np.pi)
        return f1_ / f2_


SignalFunction.DSCB.__name__ = "DSCB"
SignalFunction.DSCB.__code__ = (lambda x, sigma, mu, alpha_l, alpha_r, n_l, n_r: None).__code__
SignalFunction.voigt.__name__ = "voigt"
SignalFunction.voigt.__code__ = (lambda x, sigma, mu, gamma: None).__code__
