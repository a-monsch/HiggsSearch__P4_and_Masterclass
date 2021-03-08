import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scst


class _AsymPoissonErr(object):
    _near_calc_lower_error = {0: 0,
                              1: 0.42421579360961914, 2: 0.7561395168304443, 3: 1.0175738334655762, 4: 1.24031600356102,
                              5: 1.437682181596756, 6: 1.6401071548461914, 7: 1.828809380531311, 8: 1.9980171918869019,
                              9: 2.1531015038490295, 10: 2.2973418831825256, 11: 2.4329039454460144, 12: 2.573564440011978,
                              13: 2.71704238653183, 14: 2.8509205877780914, 15: 2.9769710302352905, 16: 3.0964967012405396,
                              17: 3.2104830741882324, 18: 3.3196941614151, 19: 3.4247340261936188, 20: 3.5306936502456665,
                              21: 3.6448516845703125, 22: 3.753896266222, 23: 3.858468145132065, 24: 3.959094285964966,
                              25: 4.056209862232208, 26: 4.150178790092468, 27: 4.241308212280273, 28: 4.3298600018024445,
                              29: 4.416058778762817, 30: 4.500114470720291}
    
    _near_calc_upper_error = {0: 0,
                              1: 2.0667134523391724, 2: 2.405105322599411, 3: 2.745358496904373, 4: 3.0266018211841583,
                              5: 3.245048940181732, 6: 3.4299083948135376, 7: 3.6338911056518555, 8: 3.8340370655059814,
                              9: 4.008370667695999, 10: 4.1644439697265625, 11: 4.306962668895721, 12: 4.439002811908722,
                              13: 4.581984728574753, 14: 4.730859637260437, 15: 4.868239730596542, 16: 4.996275305747986,
                              17: 5.1165759563446045, 18: 5.230372965335846, 19: 5.338625460863113, 20: 5.442093729972839,
                              21: 5.551243782043457, 22: 5.66814911365509, 23: 5.779230922460556, 24: 5.885223418474197,
                              25: 5.9867331981658936, 26: 6.084259033203125, 27: 6.17824923992157, 28: 6.26904222369194,
                              29: 6.356954097747803, 30: 6.442250490188599}
    
    def __init__(self, array, kind="approx"):
        self.array = array
        
        if kind == "approx":
            self._approximative()
        if kind == "calc":
            self._near_calc()
    
    def _approximative(self):
        self.upper_error = lambda x: -0.5 + np.sqrt(x + 0.25)
        self.lower_error = lambda x: 0.5 + np.sqrt(x + 0.25)
    
    def _near_calc(self):
        
        intervall = 0.68
        
        @np.vectorize
        def upper_limit(mu, eps=1e-8):
            
            if mu in _AsymPoissonErr._near_calc_upper_error:
                return _AsymPoissonErr._near_calc_upper_error[mu]
            
            @np.vectorize
            def __pp_f(x):
                return scst.poisson.pmf(math.floor(x), mu=mu)
            
            upper_limit, x, start_block, int_border = 0.5 + intervall / 2., mu, 0.0, 0.0
            
            while int_border + __pp_f(start_block) <= upper_limit:
                int_border += __pp_f(start_block)
                start_block += 1
            
            x_, dx_, = start_block, 0.5
            while dx_ > eps:
                if int_border + __pp_f(x_) * dx_ <= upper_limit:
                    int_border += __pp_f(x_) * dx_
                    x = x_ - mu + 0.5
                    x_ += dx_
                else:
                    dx_ *= 0.5
            
            _AsymPoissonErr._near_calc_upper_error[mu] = x
            return _AsymPoissonErr._near_calc_upper_error[mu]
        
        @np.vectorize
        def lower_limit(mu, eps=1e-8):
            
            if mu in _AsymPoissonErr._near_calc_lower_error:
                return _AsymPoissonErr._near_calc_lower_error[mu]
            
            @np.vectorize
            def __pp_c(x):
                return scst.poisson.pmf(np.ceil(x), mu=mu)
            
            lower_limit, x, start_block, int_border = 0.5 - intervall / 2., mu, mu, 0.5
            
            while int_border - __pp_c(start_block) > lower_limit:
                int_border -= __pp_c(start_block)
                start_block -= 1
            
            x_, dx_ = start_block, 0.5
            while dx_ > eps:
                if int_border - __pp_c(start_block) * dx_ >= lower_limit:
                    int_border -= __pp_c(start_block) * dx_
                    x = mu - x_ - 0.5
                    x_ -= dx_
                else:
                    dx_ *= 0.5
            
            _AsymPoissonErr._near_calc_lower_error[mu] = x
            return _AsymPoissonErr._near_calc_lower_error[mu]
        
        # a little confusion, naming error: lower <-> upper
        self.upper_error = lambda x: lower_limit(x)
        self.lower_error = lambda x: upper_limit(x)
    
    @property
    def error(self):
        return np.array([self.upper_error(self.array),
                         self.lower_error(self.array)])


class Hist(object):
    
    def __init__(self, bins, hist_range):
        self.hist_range = hist_range
        self.bins = bins
        
        _h, self.be = np.histogram(np.zeros(1), bins=self.bins, range=self.hist_range)
        _h = np.array(_h, dtype=float)
        self.data = {}
        
        self.width = abs(self.be[0] - self.be[1])
        self.bc = self.be[1:] - self.width / 2
    
    def set_bins(self, bin_content, label):
        if len(bin_content) == self.bins:
            self.data[label] = bin_content
    
    def fill(self, data, label, global_scale=1, local_scale=None):
        _h, _ = np.histogram(data, bins=self.be, weights=local_scale)
        _h = np.array(_h, dtype=float)
        _h *= global_scale if global_scale else 1
        try:
            self.data[label] += _h
        except KeyError:
            self.data[label] = _h
    
    def draw(self, label_list, figure=None, ax=None, matplotlib_dicts=None):
        fig, ax = (figure, ax) if figure else plt.subplots(1, 1, figsize=(10, 6))
        
        if "mc_bac" in label_list:
            if not matplotlib_dicts or "mc_bac" not in matplotlib_dicts:
                _d = {"label": "Background", "color": "royalblue", "alpha": 1.0}
            else:
                _d = matplotlib_dicts["mc_bac"]
            ax.bar(self.bc, self.data["mc_bac"], width=self.width, **_d)
        
        if "mc_sig" in label_list:
            if not matplotlib_dicts or "mc_sig" not in matplotlib_dicts:
                _d = {"label": "Signal", "color": "orangered", "alpha": 0.75}
            else:
                _d = matplotlib_dicts["mc_sig"]
            
            ax.bar(self.bc, self.data["mc_sig"], width=self.width,
                   bottom=self.data["mc_bac"] if "mc_bac" in label_list else np.zeros_like(self.data["mc_sig"]),
                   **_d)
        if "data" in label_list:
            if not matplotlib_dicts or "data" not in matplotlib_dicts:
                _d = {"label": "Measurement", "color": "black", "marker": "o", "fmt": "o"}
            else:
                _d = matplotlib_dicts["data"]
            
            _x, _y = np.array([]), np.array([])
            for x_val, y_val in zip(self.bc, self.data["data"]):
                if y_val != 0:
                    _x, _y = np.append(_x, x_val), np.append(_y, y_val)
            ax.errorbar(_x, _y, xerr=0, yerr=_AsymPoissonErr(_y, "approx").error, **_d)
        
        ax.legend()
        return fig, ax
