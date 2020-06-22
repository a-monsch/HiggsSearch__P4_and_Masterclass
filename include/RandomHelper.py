import ast
import os


class ToSortHelper(object):
    """
    Class with randomly required functions for McFit or the widgets.
    """
    @staticmethod
    def dict_out_of_doc(func_):
        """
        Creates a dict for kafe2 plot if "Formatting dict:" exists within the docs.

        :param func_: function
        :return: dict
        """
        if func_.__doc__ is None or "Formatting dict:" not in func_.__doc__:  return None
        my_doc = func_.__doc__.split("Formatting dict:")[-1]
        used_dict_ = ast.literal_eval(my_doc.replace("\n", "").replace(" ", ""))
        used_dict_["func_param"] = ast.literal_eval(used_dict_["func_param"])
        return used_dict_

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
        Changes the number in dir_ from XXX to num.
        
        :param num: int
        :param dir_: str
        :return: str
        """
        dir_path_split = os.path.normpath(dir_).split(os.path.sep)
        # noinspection PyTypeChecker
        dir_path_split[-2] = f"d{num}"
        return os.path.join(*tuple(dir_path_split))


def check_data_state():
    d_cont_ = os.listdir("../data")
    look_for = ["for_long_analysis", "for_event_display", "for_widgets"]
    
    if any(item not in d_cont_ for item in look_for):
        print("Some Files not found, downloadng...")
        os.system("sh ../include/reset_data.sh")
        if any(item not in os.listdir("../data") for item in look_for):
            print ("An error has occurred. It may be necessary to download the data sets manually")
        else:
            print("...downloading complete")
    else:
        print("Necessary files present. Continue...")