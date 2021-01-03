import inspect
import os
import re
import zipfile


class AliasDict(dict):
    
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.aliases = {}
    
    def add_alias(self, key, alias_name):
        if isinstance(alias_name, list):
            for name in alias_name:
                self.add_alias(key=key, alias_name=name)
        else:
            self.aliases[alias_name] = key
    
    def del_alias(self, key, *, delete_all=False):
        self.del_alias(self.aliases_of(key)) if delete_all else None
        if isinstance(key, str) and key in self.aliases:
            del self.aliases[key]
        if isinstance(key, list):
            for item in key:
                self.del_alias(item)
    
    def aliases_of(self, key):
        _initial_key = self.initial_key(key)
        return [_initial_key] + [_a for _a, _n in self.aliases.items() if _n == _initial_key] if _initial_key else None
    
    def initial_key(self, key):
        for _a, _n in self.aliases.items():
            if _a == key or _n == key:
                return _n
        else:
            return None
    
    def change_initial_key(self, new_key, old_key=None):
        if new_key not in self and old_key is None:
            raise KeyError(new_key)
        
        if old_key:
            _old_aliases = self.aliases_of(old_key)
            _old_aliases.remove(new_key)
            dict.__setitem__(self, new_key, self[self.initial_key(old_key)])
            
            del self[new_key]
            self.add_alias(new_key, _old_aliases)
        if not old_key:
            if new_key != self.initial_key(new_key):
                self.change_initial_key(new_key=new_key, old_key=self.initial_key(new_key))
    
    def __getitem__(self, key):
        return dict.__getitem__(self, self.aliases.get(key, key))
    
    def __setitem__(self, key, value):
        return dict.__setitem__(self, self.aliases.get(key, key), value)
    
    def __contains__(self, key):
        if self.initial_key(key):
            if key == self.initial_key(key):
                return True
            if any(key == _a for _a in self.aliases_of(key)):
                return True
        else:
            return False
    
    def __delitem__(self, key):
        dict.__delitem__(self, self.aliases.get(key, key))
        self.aliases = {k: v for k, v in self.aliases.items() if v != self.initial_key(key)}


class ToSortHelper(object):
    """
    Class with randomly required functions for McFit or the widgets.
    """
    
    @staticmethod
    def legend_without_duplicate_labels(ax):
        """
        Throws all duplicate labels out of the ax object and creates the legend.

        :param ax: matplotlib.pyplot.axes
        """
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))
    
    @staticmethod
    def mixed_file_list(new_higgs_mass, main_dir_):
        """
        Function to obtain a mixture of the original MC simulations
        and MC simulations of the Higgs boson of other masses.
        
        :param new_higgs_mass: int or str
                               Higgs mass (num) or path of files
        :param main_dir_: str
                          path of main mc_dir
        :return: list
                 1D list of path strings
        """
        if isinstance(new_higgs_mass, str):
            files = [os.path.join(main_dir_, it) for it in os.listdir(main_dir_) if "_H_to_" not in it]
            for item in os.listdir(new_higgs_mass):
                files.append(os.path.join(new_higgs_mass, item))
            return files
        if isinstance(new_higgs_mass, int):
            ToSortHelper.mixed_file_list(new_higgs_mass=ToSortHelper.get_other_mc_dir(new_higgs_mass),
                                         main_dir_=main_dir_)
    
    @staticmethod
    def get_other_mc_dir(num, dir_="../other_mc_sig_mc/dXXX/mc_aftH"):
        """
        Changes the number in directory from XXX to num.
        
        :param num: int
        :param dir_: str
        :return: str
        """
        dir_path_split = os.path.normpath(dir_).split(os.path.sep)
        # noinspection PyTypeChecker
        dir_path_split[-2] = f"d{num}"
        return os.path.join(*tuple(dir_path_split))


def check_data_state(directory="../data", group="student"):
    _cont_dir = [item for item in os.listdir(directory) if ".zip" not in item]
    _cont_zip = [item for item in os.listdir(directory) if ".zip" in item]
    look_for = ["for_long_analysis", "for_event_display"]
    
    if "pupil" in group:
        look_for.append("for_widgets")
    
    if all(item in _cont_dir for item in look_for):
        print("Necessary directories present.")
        return None
    
    if any(item not in _cont_dir for item in look_for):
        if all(f"{item}.zip" in _cont_zip for item in look_for):
            print("All .zip files present. Unpacking ...", end="\r", flush=True)
            for item in _cont_zip:
                with zipfile.ZipFile(os.path.join(directory, item), 'r') as zip_ref:
                    zip_ref.extractall(directory)
                os.remove(os.path.join(directory, item))
            print("All .zip files implemented_method_present. Unpacking complete.")
        else:
            if "student" in group:
                print("Some files are missing. It may be necessary to download the data sets manually")
            if "pupil" in group:
                print("Some files are missing. Downloading ...", end="\r", flush=True)
                os.system("sh ../include/reset_data.sh")
                if all(f"{item}.zip" in _cont_zip for item in look_for):
                    print("Some files are missing. Downloading complete.")
                else:
                    print("Some files are missing. It may be necessary to download the data sets manually")


class BColors:
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PINK = '\033[95m'
    CYAN = '\033[96m'
    
    ENDC = '\033[0m'
    
    @staticmethod
    def wraps(msg, style):
        if isinstance(style, str):
            if "," in style:
                return BColors.wraps(msg, style.replace(" ", "").split(","))
            try:
                return f"{getattr(BColors, style)}{msg}{BColors.ENDC}"
            except AttributeError:
                return msg
        if isinstance(style, list):
            for s in style:
                msg = BColors.wraps(msg, s)
            return msg


def in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        _shell_list = ['ZMQInteractiveShell', 'TerminalInteractiveShell']
        return True if shell in _shell_list else False
    except NameError:
        return False


class ReformatText(object):
    
    @staticmethod
    def implemented_method_present(_cls, _method):
        try:
            _str = inspect.getsource(getattr(_cls, _method)).replace(":return:", "")
            return True if "return" in _str else False
        except AttributeError:
            return False
    
    @staticmethod
    def color_docs(msg, calc_instance=None, filter_instance=None, set_format=None):
        if set_format is None:
            set_format = {"workflow": "BOLD, UNDERLINE"}
            # methods
            for (_cls, _method) in re.findall(r"(?<=\()(.*?)\.(.*?)(?=\))", msg):
                if calc_instance is None and filter_instance is None:
                    set_format[f"{_cls}.{_method}"] = "UNDERLINE, BOLD, CYAN"
                    continue
                else:
                    _prim_cls = calc_instance if "Calc" in _cls else (filter_instance if "Filter" in _cls else None)
                    if ReformatText.implemented_method_present(_prim_cls, _method):
                        set_format[f"{_cls}.{_method}"] = "UNDERLINE, BOLD, GREEN"
                    if not ReformatText.implemented_method_present(_prim_cls, _method):
                        set_format[f"{_cls}.{_method}"] = "UNDERLINE, RED, BOLD"
        
        for key, value in set_format.items():
            msg = msg.replace(key, BColors.wraps(key, value))
        return msg
