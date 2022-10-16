"""
This file contains utilities used in the TP1 exercises.
"""

from itertools import product, cycle, combinations
import numpy as np
import pandas as pd
import vector as vec
import matplotlib.pyplot as plt


### Class to access particle four vectors
@pd.api.extensions.register_dataframe_accessor("v4")
class FourVecAccessor(object):
    def __init__(self, pandas_obj):
        # to distinguish between multiple particles or single particle
        # we only need to save the column information,
        self._obj_columns = pandas_obj.columns
        # to keep data consistent when appending columns unsing this accessor save indices to use them in returns
        self._obj_indices = pandas_obj.index
        # get the correct index level, 0 for single particle, 1 for multiple
        _vars = self._obj_columns.get_level_values(self._obj_columns.nlevels - 1)

        if "E" in _vars and "px" in _vars:
            kin_vars = ["E", "px", "py", "pz"]
        elif "E" in _vars and "pt" in _vars:
            kin_vars = ["E", "pt", "phi", "eta"]
        else:
            raise KeyError("No matching structure implemented for interpreting the data as a four "
                           "momentum!")

        # the following lines are where the magic happens

        # no multi-index, just on particle
        if self._obj_columns.nlevels == 1:
            # get the dtypes for the kinetic variables
            dtypes = pandas_obj.dtypes
            kin_view = list(map(lambda x: (x, dtypes[x]), kin_vars))

            # get the kinetic variables from the dataframe and convert it to a numpy array.
            # require it to be C_CONTIGUOUS, vector uses C-Style
            # This array can then be viewed as a vector object.
            # Every property not given is calculated on the fly by the vector object.
            # E.g. the mass is not stored but calculated when the energy is given and vice versa.
            self._v4 = np.require(pandas_obj[kin_vars].to_numpy(), requirements='C').view(
                kin_view).view(vec.MomentumNumpy4D)

        # multi-index, e.g. getting the four momentum for multiple particles
        elif self._obj_columns.nlevels == 2:
            # get the dtypes for the kinetic variables
            # assume the same dtypes for the other particles
            dtypes = pandas_obj[self._obj_columns.get_level_values(0).unique()[0]].dtypes
            kin_view = list(map(lambda x: (x, dtypes[x]), kin_vars))
            self._v4 = np.require(pandas_obj.loc[:, (self._obj_columns.get_level_values(0).unique(),
                                                     kin_vars)].to_numpy(),
                                  requirements='C').view(kin_view).view(vec.MomentumNumpy4D)

        else:
            raise IndexError("Expected a dataframe with a maximum of two multi-index levels.")

    def __getattribute__(self, item):
        """
        Attributes of this accessor are forwarded to the four vector.

        Returns either a pandas dataframe, if we have multiple particles
        or a pandas Series for a single particle.
        """
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            try:
                return pd.DataFrame(self._v4.__getattribute__(item),
                                    columns=pd.MultiIndex.from_product(
                                        [self._obj_columns.unique(0), [item]]),
                                    index=self._obj_indices)
            except ValueError:
                try:
                    return pd.Series(self._v4.__getattribute__(item).flatten(), name=item, index=self._obj_indices)
                except AttributeError as e:
                    if "'function' object has no attribute 'flatten'" in str(e):
                        raise AttributeError(
                            "Functions of the four vectors can NOT be called directly via the "
                            "accessor. Use the vector property instead! "
                            "Usage: 'df['particle'].v4.vector.desired_function()'")
                    raise e

    @property
    def vector(self):
        """The four vector object itself. It's required when using methods like boosting."""
        if self._obj_columns.nlevels == 1:
            return self._v4[:, 0]
        else:
            return self._v4


### Plotting function of physics object quantities in a grid plot given one or more MultiIndex pd.DataFrames
def plot_quantities(df, column, quantity=None,
                    bins=None, hist_range=None, density=False, weights=None,
                    label=None, unit=None, yscale="log", suptitle=None):
    """
    Plot the distributions of given quantities of given physics objects that are present in given DataFrame(s). For
    two or more DataFrames, the plotting is done object-quantity wise in the same subplot.

    :param df: pd.DataFrame or list of pd.DataFrames
    :param column: str or list of str of physics objects, i. e. "lepton_0" or ["jet_0", "jet_1"]
    :param quantity: str or list of str, i. e. "pt" or ["pt", "eta", "phi"] in case of a two level index, None
                     if only one index level exists
    :param label: str or list of str with the same length as df if df is a list
    :param unit: str or list of str with the same length as quantity for x-axis label or the same length as columns in case
                 quantity is None
    :param density: bool or list of bool with the same length as quantity
    :param hist_range: None or (float, float) or list of those with same length as quantity or column or the same
                       length as all created subplots:
                       a) column = ["jet_0", "jet_1", "jet_2"], quantity = ["pt"],
                          hist_range = [(0, 100), (0, 50), (0, 25)]
                            => creates subplots of (jet_0, pt), (jet_1, pt) and (jet_2, pt) in (0, 100), (0, 50) and (0, 25)
                               intervals.
                       b) column = ["jet_0", "jet_1", "jet_2"], quantity = ["pt", "eta"],
                          hist_range = [(0, 100), (-4, 4)]
                           => creates subplots where pt of all subplots ranges are set to (0, 100) and eta ranges to (-4, 4)
                       c) column = ["jet_0", "jet_1", "jet_2"], quantity = ["pt", "eta"],
                          hist_range = [(0, 100), (-4, 4), None, (-4, 4), None, (-4, 4)]
                           => creates six subplots where all eta ranges are set to (-4, 4) and (jet_0, pt) range is set
                              to (0, 100). All other ranges are set by default.
    :param bins: int or list of int, setting the number of used bins. In case of a list: the same rules applies for bins as for
                 hist_range
    :param yscale: str like as in matplotlib yscale or list of str, setting the used yscale. In case of a list: the same rules
                   applies for bins as for hist_range. Possible options are for example 'log', 'linear', 'logit', 'sumlog', ...
    :param weights: None, 1D np.ndarray or pd.Series  or a list of those with the same length as df
    :param suptitle: None or str containing the figure title
    :return: None
    """

    def to_list(it):  # helper for argument conversion
        return it if isinstance(it, list) else [it]

    with_quantity = True if quantity else False  # in case no quantity is provided
    df, label, unit, hist_range = to_list(df), to_list(label), to_list(unit), to_list(hist_range)
    column, quantity = to_list(column), to_list(quantity)
    bins, density, yscale, weights = to_list(bins), to_list(density), to_list(yscale), to_list(weights)

    # padding up to quantity dim if necessary
    unit = unit + [None] * (len(quantity) - len(unit))
    density = density + [True if len(density) == 1 and density[0] else False] * (len(quantity) - len(density))

    # padding up to df dim if necessary
    label = label + [None] * (len(df) - len(label))
    weights = weights + [None] * (len(df) - len(weights))

    def check_dim_of(it):  # checking for bins, hist_range, yscale length in case to prevent strange plots
        return (len(it) == 1 or len(it) == len(quantity)
                or (len(it) == len(column) and len(quantity) == 1)
                or len(it) == len(list(product(column, quantity))))

    assert check_dim_of(hist_range) and check_dim_of(bins) and check_dim_of(yscale), \
        "(hist_range/bins/yscale) have either to be None or a (two element tuple/int/str) or be a " \
        "list of those types and have the same length as column if single or None quantity is " \
        "given or have the same length as quantity or the length of column * quantity"
    bins, hist_range, yscale = cycle(bins), cycle(hist_range), cycle(yscale)

    # number of created plots and subdivision in columns and rows
    n_plots = len(list(product(column, quantity)))
    n_columns = len(quantity) if len(quantity) > 1 else (4 if n_plots > 4 else n_plots)
    n_rows = int(np.ceil(n_plots / n_columns))

    fig, ax = plt.subplots(n_rows, n_columns, figsize=(25, int(4 * n_rows)))
    ax = ax if isinstance(ax, np.ndarray) else np.array(ax)  # in case only a single column amd a single quantity is given
    fig.suptitle(suptitle)

    def get_quantity(dataframe, _column_name, _quantity_name):  # helper to get pd.Series...
        if _quantity_name is not None:  # in case of a two level index
            try:  # of a four vector quantity
                return getattr(dataframe[_column_name].v4, _quantity_name).to_numpy()
            except AttributeError:  # of other quantity
                return dataframe.loc[:, (_column_name, _quantity_name)].to_numpy()
        else:  # in case of a single level index
            return dataframe.loc[:, _column_name].to_numpy()

    def is_None(it):  # helper: checking for None __and__ avoiding a collision with np.ndarray like objects
        return isinstance(it, type(None))

    for i, (_ax, (_column, (_quantity, _unit, _density))) in enumerate(zip(ax.flatten(),
                                                                           product(column,
                                                                                   zip(quantity, unit, density)))):
        empty_subplot_content = 0  # counter for setting subplots with no content to invisible
        edges = None  # for auto setting of first histogram edges
        _bins, _hist_range, _yscale = next(bins), next(hist_range), next(yscale)  # bins, hist_range, yscale for (_column, _quantity)
        for j, (_df, _label, _weights) in enumerate(zip(df, label, weights)):  # in case multiple dataframes are given
            _plot_values = get_quantity(_df, _column, _quantity)
            if np.all(np.isnan(_plot_values)):
                empty_subplot_content += 1
                continue
            _, edges, _ = _ax.hist(_plot_values, histtype="step", range=_hist_range, density=_density,
                                   bins=100 if not _bins and is_None(edges) else _bins if is_None(edges) else edges,  # same edges
                                   label=_label if _label else f"Label_{j}",  # default or custom
                                   weights=_weights)
            _ax.set(yscale=_yscale, ylabel="Events",
                    title=f"{_column}{':' if _quantity else ''} {_quantity if _quantity else ''}",
                    xlabel=f"{_quantity if _quantity else _column} "
                           f"{'in ' if (unit[i % len(unit)]) or (_unit and with_quantity) else ''}"
                           f"{_unit if (_unit and with_quantity) else (unit[i % len(unit)] if unit[i % len(unit)] else '')}")
            # _unit if with_quantity else unit[i] if _unit or unit[i] else ''
            _ax.legend()
        if empty_subplot_content == len(df):
            _ax.set_visible(False)  # not showing MET eta or similar not defined or emtpy NaN like quantity lists
    plt.tight_layout()
    plt.show()


### Helper function to reindex a changed dataframe to avoid unexpected behaviour
### Can be used via df.pipe(reset_idx)
def reset_idx(df):
    return df.reset_index().drop(columns=["index"], axis=0, level=0)


### Basic Event and Lepton Filter, simplified for .pipe usage

class Filter(object):

    # Helper 1
    @staticmethod
    def get_empty_mask(df):
        """
        Creates a True-filled mask of top level objects of a given pd.DataFrame

        :param df: pd.DataFrame
        :return: pd.DataFrame (mask)
        """
        return pd.DataFrame(np.ones((df.shape[0], df.columns.levshape[0])), columns=df.columns.levels[0], dtype=bool)

    # Helper 2
    @staticmethod
    def get_particle_filter_mask(df, mask):
        """
        Creates a full mask of a pd.DataFrame out of a given mask of top level object DataFrame
        refilling np.nans with True if some are present.

        :param df: pd.DataFrame
        :param mask: pd.DataFrame of top levels
        :return: pd.DataFrame (mask)
        """
        return (mask).reindex(df.columns, level=0, axis=1).fillna(True)

    # Helper 3
    @staticmethod
    def is_flavour(df, obj, flavour):
        """
        Creates a boolean top level mask considering the lepton flavour

        :param df: pd.DataFrame
        :param obj: List of strings of top level objects, i.e. ["lepton_0", "lepton_1", ...]
        :param flavour: 0 or 1; (muon or electron)
        :return: pd.DataFrame (mask)
        """
        return (df.loc[:, (obj, "flavour")] == flavour).droplevel(1, axis=1)

    # Helper 4
    @staticmethod
    def leptons(df):
        return np.unique([it for it in df.columns.get_level_values(0) if "lepton" in it]).tolist()


class EventFilter(Filter):

    @staticmethod
    def get_min_lepton_number_event_mask(df):
        """
        Performs a check on the minimum number of leptons (with corresponding
        flavour) within an event and creates a pd.Series 1D mask

        :param df: pd.DataFrame
        :return: pd.Series (mask)
        """

        # small helper function that summarize three steps:
        # 1. Mask all leptons with the undesired falvour with False/0
        # 2. Counts the remaining number of leptons with the desired flavour (.sum(axis=1)) and compares
        #    it with a given minimum number of leptons (n)
        # 3. Checks if the event channel corresponds to a given channel

        leptons = Filter.leptons(df)

        def _get_submask(flavour, n, channel):
            _min_leps_mask = Filter.is_flavour(df, leptons, flavour).sum(axis=1) >= n
            return _min_leps_mask & (df.event_information.channel == channel)

        tmp_mask = pd.Series(np.zeros_like(df.event_information.channel), index=df.index).astype(bool)

        tmp_mask |= _get_submask(0, 4, 0)  # four muon channel
        tmp_mask |= _get_submask(1, 4, 1)  # four electron channel
        tmp_mask |= (_get_submask(0, 2, 2) & _get_submask(1, 2, 2))  # mixed channel

        return tmp_mask

    @staticmethod
    def get_neutral_charge_event_mask(df):
        """
        Checking all the possible combinations of two (four) leptons to see if there
        is a charge neutral combination in the events that could be used to
        reconstruct the Z boson(s).

        :param df: pd.DataFrame
        :return: pd.Series (mask)
        """

        is_channel = lambda channel: df.event_information.channel == channel
        get_charge = lambda combination: df.loc[:, (combination, "charge")].droplevel(1, axis=1)

        tmp_mask = pd.Series(np.zeros_like(df.event_information.channel), index=df.index).astype(bool)

        leptons = Filter.leptons(df)

        for leps in combinations(leptons, 4):
            leps = list(leps)  # converting from tuple
            charge = get_charge(leps)  # charge for this combination
            muons, electrons = Filter.is_flavour(df[leps], leps, 0), Filter.is_flavour(df[leps], leps, 1)  # flavour masks

            # eventwise summation of charges (axis=1) given a specific combination
            four_mu = (charge[muons].sum(axis=1) == 0) & is_channel(0)
            four_el = (charge[electrons].sum(axis=1) == 0) & is_channel(1)
            two_mu_two_el = (charge[muons].sum(axis=1) == 0) & (charge[electrons].sum(axis=1) == 0) & is_channel(2)

            tmp_mask |= (four_mu | four_el | two_mu_two_el)

        return tmp_mask

    @staticmethod
    def get_z_mass_event_mask(df, z1_mass_min, z1_mass_max, z2_mass_min, z2_mass_max):
        """
        Performs a check on the masses of calculated Z Bosons
        within an event and creates a pd.Series 1D mask

        :param df: pd.DataFrame
        :return: pd.Series (mask)
        """

        tmp_mask = pd.Series(np.ones_like(df.event_information.channel), index=df.index).astype(bool)
        tmp_mask &= ((df.Z1.v4.mass > z1_mass_min) & (df.Z1.v4.mass < z1_mass_max))
        tmp_mask &= ((df.Z2.v4.mass > z2_mass_min) & (df.Z2.v4.mass < z2_mass_max))

        return tmp_mask

    # for .pipe Method
    @staticmethod
    def z_masses(df, z1_mass_min, z1_mass_max, z2_mass_min, z2_mass_max):
        return df[EventFilter.get_z_mass_event_mask(df, z1_mass_min, z1_mass_max, z2_mass_min, z2_mass_max).to_numpy()].pipe(reset_idx)

    # for .pipe Method
    @staticmethod
    def min_lepton_number(df):
        return df[EventFilter.get_min_lepton_number_event_mask(df).to_numpy()].pipe(reset_idx)

    # for .pipe Method
    @staticmethod
    def neutral_charge(df):
        return df[EventFilter.get_neutral_charge_event_mask(df).to_numpy()].pipe(reset_idx)


class LeptonFilter(Filter):

    @staticmethod
    def get_min_pt_lepton_mask(df, min_pt_muon=5, min_pt_electron=7):
        leptons = Filter.leptons(df)
        pt = df[leptons].v4.pt.droplevel(1, axis=1)
        muons, electrons = Filter.is_flavour(df, leptons, 0), Filter.is_flavour(df, leptons, 1)

        tmp_mask = Filter.get_empty_mask(df)
        tmp_mask &= ((pt > min_pt_electron) & electrons) | ((pt > min_pt_muon) & muons)

        return Filter.get_particle_filter_mask(df, tmp_mask)

    @staticmethod
    def get_relative_isolation_lepton_mask(df, relative_isolation_value=0.4):
        leptons = Filter.leptons(df)
        relative_isolation = df.loc[:, (leptons, "relpfiso")].droplevel(1, axis=1)

        tmp_mask = Filter.get_empty_mask(df)
        tmp_mask &= (relative_isolation < relative_isolation_value)

        return Filter.get_particle_filter_mask(df, tmp_mask)

    # for .pipe Method
    def relative_isolation_lepton(df, relative_isolation_value):
        return df[LeptonFilter.get_relative_isolation_lepton_mask(df, relative_isolation_value)].pipe(reset_idx)

    # For .pipe Method
    @staticmethod
    def min_pt_lepton(df, min_pt_muon, min_pt_electron):
        return df[LeptonFilter.get_min_pt_lepton_mask(df, min_pt_muon, min_pt_electron)].pipe(reset_idx)



### Helper function for getting the muons and electrons in an event with a two muons and two electrons decay
def get_leptons_by_flavour(row, flavour=0):
    leptons = Filter.leptons(pd.DataFrame(row).T)
    _filter = np.array(leptons)[Filter.is_flavour(pd.DataFrame(row).T, leptons, flavour).values[0]].tolist()
    return row[_filter[0]], row[_filter[1]]


### Helper Function for plotting the comparison between Monte Carlo simulation and the
### actual measurement. A channel wise MC scaling is performed during the creation
def get_scaled_bins_mc_data_comparison(df_data, df_mc_sig, df_mc_bkg, obj, quantity, bins, hist_range):
    """
    Helper Function to create the scaled histogram bins from given dataframes containing MC simulation

    :param df_data: pd.DataFrame containing the measurement
    :param df_mc_sig: pd.DataFrame containing the simulated signal
    :param df_mc_bkg: pd.DataFrame containing the simulated background
    :param obj: str name of a given top-level physics object that is present in all
                dataframes, e.g. "Z1", "four_lep", "lepton_0"
    :return: tuple of created signal_bins, background_bins, measurement_bins,
             corresponding bin_edges and the middle coordinates of the bins (measurement_x)
    """

    ### Load Information for histogram scaling
    scale_df = pd.read_csv("data/histogram_mc_scaling.csv")

    ### create empty arrays for a histogram
    bins_sig, edges = np.histogram(np.array([]), bins=bins, range=hist_range, weights=np.array([]))
    bins_bkg = np.zeros_like(bins_sig)

    for _channel in [0, 1, 2]:  # creating the MC histograms channel wise
        # Helper for getting specific attribute and the histogram scale factor
        get_attr = lambda df: getattr(df[df.event_information.channel == _channel][obj].v4, quantity).to_numpy()
        get_factor = lambda process: scale_df[(scale_df.process == process) & (scale_df.channel == _channel)].f.to_numpy()

        # summation for the final histogram
        bins_sig += np.histogram(get_attr(df_mc_sig), bins=edges)[0].astype(float) * get_factor("signal")
        bins_bkg += np.histogram(get_attr(df_mc_bkg), bins=edges)[0].astype(float) * get_factor("background")

    bins_measurement, _ = np.histogram(getattr(df_data[[obj]].v4, quantity), bins=edges)

    bins_sig = np.pad(bins_sig, 1)[1:]  # padding for .fill_between plotting method
    bins_bkg =  np.pad(bins_bkg, 1)[1:]  # padding for .fill_between plotting method
    measurement_x = edges[1:] - abs(edges[1] - edges[0]) / 2  # for plt.errorbar method

    return bins_sig, bins_bkg, bins_measurement, edges, measurement_x
