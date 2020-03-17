import numpy as np


class LegendreLinearCombination(object):
    """
    The linear combination of legendrepolynomials and their formatting
    for the software package kafe2 is implemented in this class.
    """
    _x_mean = 0.0
    _x_latex = "m_{{4\\ell}}"
    _x_mean_latex = " - \\bar{{m}}_{{4\\ell}}"
    
    def __init__(self, x_mean=0.0):
        self._x_mean = x_mean
        self._x_mean_latex = self._x_mean_latex if x_mean != 0.0 else ""
        LegendreLinearCombination.set_global_x_mean(self._x_mean, self._x_latex, self._x_mean_latex)
    
    def _get_param_names(self, name, with_x=False):
        """
        Iterates through the function arguments and converts them to LaTeX format.
        
        :param name: str
        :param with_x: bool
        :return: dict
        """
        if with_x:
            return {k: (f"a_{i - 1}" if k != "x" else self._x_latex)
                    for i,k in enumerate(getattr(self, name).__code__.co_varnames)}
        if not with_x:
            return {k: (f"a_{i - 1}" if k != "x" else self._x_latex)
                    for i,k in enumerate(getattr(self, name).__code__.co_varnames) if k != "x"}
    
    def pretty_dict(self, grade, with_x=False):
        """
        Creates a dict from functions name, expression and
        parameters (LaTeX) that is used in kafe2.
        
        :param grade: int
        :return: dict
        """
        return {'x_name': self._x_latex,  # not required in newer kafe2 versions -> with_x=True
                'func_name': f"L_{grade}",
                'func_expr': '\sum_{{n=0}}^' + f"{grade} a_i " + f'({self._x_latex + self._x_mean_latex})^n',
                'func_param': self._get_param_names(f"grade_{grade}", with_x=with_x)}
    
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
        
        LegendreLinearCombination.set_global_x_mean(self._x_mean, self._x_latex, self._x_mean_latex)
    
    @staticmethod
    def set_global_x_mean(value=None, x_latex=None, x_mean_latex=None):
        """
        Sets the mean value of x and its string representation globaly.

        :param value: float
        :param x_latex: str
        :param x_mean_latex: str
        """
        LegendreLinearCombination._x_mean = value if value is not None else LegendreLinearCombination._x_mean
        LegendreLinearCombination._x_latex = "{x}-\\bar{{m}}_{{4\\ell}}" if x_latex is None else x_latex
        LegendreLinearCombination._x_mean_latex = " - \\bar{{m}}_{{4\\ell}}" if x_mean_latex is None else x_mean_latex
    
    @staticmethod
    def custom(x, *args, **kwargs):
        """
        Function which creates any linear combination of legend trepolynomials.
        Not usable in the kafe2 fits.
        Usage: specify either only args or only kwargs next to x.
        
        :param x: float or np.ndarray
        :param args: floats
        :param kwargs: floats
        :return: float
        """
        x = x - LegendreLinearCombination._x_mean
        
        if (kwargs and args) or (not kwargs and not args):
            raise NotImplementedError("Use only args or only kwargs here")
        
        if args:
            return np.polynomial.legendre.legval(x, [*args])
        if kwargs:
            return np.polynomial.legendre.legval(x, np.array(list(kwargs.values())))
    
    @staticmethod
    def grade_0(x, a):
        x = x - LegendreLinearCombination._x_mean
        return np.polynomial.legendre.legval(x, [a])
    
    @staticmethod
    def grade_1(x, a, b):
        x = x - LegendreLinearCombination._x_mean
        return np.polynomial.legendre.legval(x, [a, b])
    
    @staticmethod
    def grade_2(x, a=50. / 2500 + 0.05 / 2500 / 3, b=0, c=-0.05 / 2500 * 2 / 3):
        x = x - LegendreLinearCombination._x_mean
        return np.polynomial.legendre.legval(x, [a, b, c])
    
    @staticmethod
    def grade_3(x, a, b, c, d):
        x = x - LegendreLinearCombination._x_mean
        return np.polynomial.legendre.legval(x, [a, b, c, d])
    
    @staticmethod
    def grade_4(x, a, b, c, d, e):
        x = x - LegendreLinearCombination._x_mean
        return np.polynomial.legendre.legval(x, [a, b, c, d, e])
    
    @staticmethod
    def grade_5(x, a, b, c, d, e, f):
        x = x - LegendreLinearCombination._x_mean
        return np.polynomial.legendre.legval(x, [a, b, c, d, e, f])
    
    @staticmethod
    def grade_6(x, a, b, c, d, e, f, g):
        x = x - LegendreLinearCombination._x_mean
        return np.polynomial.legendre.legval(x, [a, b, c, d, e, f, g])
