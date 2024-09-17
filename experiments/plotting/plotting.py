# from seaborn.categorical import _PointPlotter
from qiskit import QuantumCircuit, transpile
# from qiskit.providers.aer import QasmSimulator
# from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit_aer import AerSimulator, Aer
from qiskit.providers.fake_provider import FakeProvider, FakeManila, FakeToronto, FakeJakartaV2
from qiskit_aer.noise import NoiseModel
import qiskit.quantum_info as qi
import qiskit

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from datetime import date
import pathlib
import inspect
import time
import glob
from itertools import chain

import matplotlib as mpl
from seaborn.relational import _RelationalPlotter
from seaborn import utils
from seaborn.utils import remove_na, _normal_quantile_func, _draw_figure, _default_color
from seaborn._statistics import EstimateAggregator
from seaborn.palettes import color_palette, husl_palette, light_palette, dark_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
from seaborn.categorical import _CategoricalStatPlotter
from seaborn.relational import _RelationalPlotter, _ScatterPlotter


module_path = os.path.abspath(os.path.join('..'))
module_path = os.path.abspath(os.path.join(module_path, '..')) # qrc_surrogate
sys.path.insert(1, module_path)


class _better_PointPlotter(_CategoricalStatPlotter):

    default_palette = "dark"

    def __init__(self, x, y, hue, data, order, hue_order,
                 estimator, errorbar, n_boot, units, seed,
                 markers, linestyles, dodge, join, scale,
                 orient, color, palette, errwidth, capsize, label,
                 xoffset, xpos, linewidth, markersize):
        """Initialize the plotter."""
        self.establish_variables(x, y, hue, data, orient,
                                 order, hue_order, units)
        self.establish_colors(color, palette, 1)
        self.estimate_statistic(estimator, errorbar, n_boot, seed)

        # Override the default palette for single-color plots
        if hue is None and color is None and palette is None:
            self.colors = [color_palette()[0]] * len(self.colors)

        # Don't join single-layer plots with different colors
        if hue is None and palette is not None:
            join = False

        # Use a good default for `dodge=True`
        if dodge is True and self.hue_names is not None:
            dodge = .025 * len(self.hue_names)

        # Make sure we have a marker for each hue level
        if isinstance(markers, str):
            markers = [markers] * len(self.colors)
        self.markers = markers

        # Make sure we have a line style for each hue level
        if isinstance(linestyles, str):
            linestyles = [linestyles] * len(self.colors)
        self.linestyles = linestyles

        # Set the other plot components
        self.dodge = dodge
        self.join = join
        self.scale = scale
        self.errwidth = errwidth
        self.capsize = capsize
        self.label = label
        self.xoffset = xoffset
        self.data = data
        self.xpos = xpos
        self.x = x
        if linewidth is None:
            linewidth = mpl.rcParams["lines.linewidth"]
        self.linewidth = linewidth
        self.markersize = markersize

    @property
    def hue_offsets(self):
        """Offsets relative to the center position for each hue level."""
        if self.dodge:
            offset = np.linspace(0, self.dodge, len(self.hue_names))
            offset -= offset.mean()
        else:
            offset = np.zeros(len(self.hue_names))
        return offset

    def draw_points(self, ax):
        """Draw the main data components of the plot."""
        if self.xpos == True:
            pointpos = self.data[self.x].unique()
        else:
            # Get the center positions on the categorical axis
            pointpos = np.arange(len(self.statistic))
            # self.statistic # numpy array with y posiions
        if self.xoffset != 0:
            self.xoffset = float(self.xoffset)
            try:
                pointpos = np.array(pointpos, dtype=float)
            except ValueError as e:
                print(e)
                print('-> xpos=False might help.')
            pointpos += self.xoffset

        # variables = _ScatterPlotter.get_semantics(locals()) # dictionary
        # p = _ScatterPlotter(data=self.data, variables=variables, legend='auto')
        # empty = np.full(len(self.data), np.nan)
        # x = self.data.get("x", empty)

        # Get the size of the plot elements
        lw = self.linewidth * 1.8 * self.scale
        mew = lw * .75
        markersize = np.pi * np.square(lw) * self.markersize

        if self.plot_hues is None:

            # Draw lines joining each estimate point
            if self.join:
                color = self.colors[0]
                ls = self.linestyles[0]
                if self.orient == "h":
                    ax.plot(self.statistic, pointpos,
                            color=color, ls=ls, lw=lw)
                else:
                    ax.plot(pointpos, self.statistic,
                            color=color, ls=ls, lw=lw)

            # Draw the confidence intervals
            self.draw_confints(ax, pointpos, self.confint, self.colors,
                               self.errwidth, self.capsize)

            # Draw the estimate points
            marker = self.markers[0]
            colors = [mpl.colors.colorConverter.to_rgb(c) for c in self.colors]
            if self.orient == "h":
                x, y = self.statistic, pointpos
            else:
                x, y = pointpos, self.statistic

            ax.scatter(x, y,
                       linewidth=mew, marker=marker, s=markersize,
                       facecolor=colors, edgecolor=colors, label=self.label)

        else:

            offsets = self.hue_offsets
            for j, hue_level in enumerate(self.hue_names):

                # Determine the values to plot for this level
                statistic = self.statistic[:, j]

                # Determine the position on the categorical and z axes
                offpos = pointpos + offsets[j]
                z = j + 1

                # Draw lines joining each estimate point
                if self.join:
                    color = self.colors[j]
                    ls = self.linestyles[j]
                    if self.orient == "h":
                        ax.plot(statistic+self.xoffset, offpos, color=color,
                                zorder=z, ls=ls, lw=lw)
                    else:
                        ax.plot(offpos+self.xoffset, statistic, color=color,
                                zorder=z, ls=ls, lw=lw)

                # Draw the confidence intervals
                if self.confint.size:
                    confint = self.confint[:, j]
                    errcolors = [self.colors[j]] * len(offpos)
                    self.draw_confints(ax, offpos+self.xoffset, confint, errcolors,
                                       self.errwidth, self.capsize,
                                       zorder=z)

                # Draw the estimate points
                n_points = len(remove_na(offpos))
                marker = self.markers[j]
                color = mpl.colors.colorConverter.to_rgb(self.colors[j])

                if self.orient == "h":
                    x, y = statistic, offpos
                else:
                    x, y = offpos, statistic

                if not len(remove_na(statistic)):
                    x = y = [np.nan] * n_points
                
                ax.scatter(x, y, label=hue_level,
                           facecolor=color, edgecolor=color,
                           linewidth=mew, marker=marker, s=markersize,
                           zorder=z)

    def plot(self, ax):
        """Make the plot."""
        self.draw_points(ax)
        self.annotate_axes(ax)
        if self.orient == "h":
            ax.invert_yaxis()


def better_pointplot(
    data=None, *, x=None, y=None, hue=None, order=None, hue_order=None,
    estimator="mean", errorbar=("ci", 95), n_boot=1000, units=None, seed=None,
    markers="o", linestyles="-", dodge=False, join=True, scale=1,
    orient=None, color=None, palette=None, errwidth=None, ci="deprecated",
    capsize=None, label=None, ax=None,
    xoffset=0, xpos=False, linewidth=None, markersize=2,
):
    """Use like sns.pointplot, but optionally adjust the x positions. 
    Args:
        xoffset (int): shift x position to the right.
        xpos (int): plot along x axis like scatterplot, instead of using regular spacing.
        linewidth (float): linewidth
    Returns:
        matplotlib.axes
    """


    # errorbar = utils._deprecate_ci(errorbar, ci)

    plotter = _better_PointPlotter(x, y, hue, data, order, hue_order,
                            estimator, errorbar, n_boot, units, seed,
                            markers, linestyles, dodge, join, scale,
                            orient, color, palette, errwidth, capsize, label,
                            xoffset, xpos, linewidth, markersize)

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax)
    return ax


def get_relevant_dataframe(
        f, 
        model_of_interest, 
        data_of_interest,
        xaxis_column, 
        change_moi={}, 
        hue_columns=[], 
        keep_columns=[],
        cutoff=None,
        average = True, # average metrics
        select_values = {},
        select_values_range = {},
        verbose=False,
) -> tuple[pd.DataFrame, str]:
    """Get experiments where model and data settings match what we are interested in.

    Each Experiment is a row in the dataframe. 
    Each column is a model setting, a data setting, or a results / metric of the experiment.

    Args:
        f (str):
            path to the dataframe with all the experiments.
            Usually f'{module_path}/experiments/results/{experiment.model_name}.parquet'
        model_of_interest, data_of_interest (dict(str: any)):
            Model and data settings required to match for experiment to be in the returned dataframe.
            selects rows in dataframe which match column: value.
        average (bool):
            if True, repetitions of the same experiment (same model and data settings) are averaged.
        hue_columns (list(str)):
            columns in dataframe that shall appear in sns.pointplot as different colors.
        change_moi (dict(str: any)):
            overwrites model_of_interest. 
            Usefull when one wants to pass small variations of model_of_interest without changing model_of_interest.
        cutoff (float):
            experiments if a MSE Validation higher than the value are deleted.
        select_values (dict(str: list(any))):
            col: list(values). only keep experiments where the col has one of the values.
        select_values_range (dict(str: [float, float])):
            col: [min, max]. only keep experiments where the xaxis_column is greater than the value.
    """
    if verbose: print('-'*20)

    metrics = ['corr_train', 'corr_val', 'mse_train', 'mse_val', 'mape_train', 'mape_val', 'nrmse_train', 'nrmse_val', 'traintime']
    extended_metrics = metrics + ['index', 'level_0', 'traintime', 'date']

    df_rel = pd.read_parquet(f)

    # data of interest
    # df_rel = df_rel[df_rel['xtype'] == data_of_interest['xtype']]

    do_not_delete_columns = hue_columns + keep_columns + metrics + [xaxis_column]
    
    _selection_params = {**model_of_interest, **data_of_interest}
    _selection_params = {**_selection_params, **change_moi}
    _selection_params.pop(xaxis_column, None)
    for c in hue_columns:
        _selection_params.pop(c, None)
    for c in keep_columns:
        _selection_params.pop(c, None)
    
    unique_values = df_rel[xaxis_column].unique()
    if verbose: print(f'unique values in {xaxis_column}:', unique_values)
    # unique_values_hue = [i for i in [df_rel[_c].unique() for _c in hue_columns]]
    unique_values_hue = list(chain(*[list(df_rel[_c].unique()) for _c in hue_columns]))
    if verbose: print(f'unique values in {hue_columns}:', unique_values_hue)
    
    # get entries which match model parameters
    for key, value in _selection_params.items():
        if len(df_rel[key].unique()) > 1: # make sure not to end up with an empty dataframe
            df_rel = df_rel[df_rel[key] == value]
            # if verbose: print(f'after {key}:', df_rel['restarting'].unique()) 
            # if verbose: print(f'dataframe after removing {key}:', len(df))
            if len(df_rel[xaxis_column].unique()) != len(unique_values):
                if verbose: print(f' unique values after {key} ({value}):', df_rel[xaxis_column].unique())
                unique_values = df_rel[xaxis_column].unique()
            new_unique_values_hue = list(chain(*[list(df_rel[_c].unique()) for _c in hue_columns]))
            if len(new_unique_values_hue) != len(unique_values_hue):
                if verbose: print(f' unique values after {key} ({value}):', new_unique_values_hue)
                unique_values_hue = new_unique_values_hue
        else:
            # if verbose: print(f'only one {key} found')
            pass
    # if verbose: print(f'Rows in df:', len(df_rel))

    # drop columns with just one value
    for c in df_rel.columns.to_list():
        try:
            # some data types like lists this doesnt work (not hashable)
            if (len(df_rel[c].unique()) <= 1) and (c not in do_not_delete_columns):
                df_rel = df_rel.drop(c, axis=1)
        except:
            if verbose: print(f'couldnt drop {c}')
            continue
    
    # remove outliers
    if cutoff:
        # df_rel = df_rel[df_rel['mse_val'] < cutoff]
        # instead of removing, set to cutoff value
        df_rel.loc[df_rel['mse_val'] > cutoff, 'mse_val'] = cutoff
    # keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
    # df[np.abs(df.Data-df.Data.mean()) <= (3*df.Data.std())] 
    # from scipy import stats
    # df[(np.abs(stats.zscore(df[0])) < 3)]

    # after masking out a part of the indices, need to reindex
    # df_rel = df_rel.reset_index()

    # model params which are still left
    # used_passable_model_params = [m for m in df_rel.columns.to_list() if m in passable_model_params] 
    if verbose: print(f'Rows left in df:', len(df_rel))
    if verbose: print('Columns left in df:', df_rel.shape[1])
    for p in [p for p in df_rel.columns.to_list() if p not in extended_metrics]:
        if verbose: print('', p, df_rel[p].unique())

    for col, values in select_values.items():
        df_rel = df_select_values(df_rel, col, values)

    for col, values in select_values_range.items():
        df_rel = df_select_values_range(df_rel, col, values[0], values[1])

    # average over columns
    if average:
        keep_columns = [_c for _c in df_rel.columns.to_list() if _c not in extended_metrics]
        df_rel = df_rel.groupby(keep_columns).mean().reset_index()

    # sort
    df_rel = df_rel.sort_values([xaxis_column] + hue_columns)

    # hue = df[['qctype', 'restarting']].apply(tuple, axis=1)
    # hue = df['qctype'].astype(str) + ', ' + df['ftype'].astype(str)
    if hue_columns:
        hue = df_rel[hue_columns[0]].astype(str)
        for i in range(1, len(hue_columns)):
            hue += ', ' + df_rel[hue_columns[i]].astype(str)
    else:
        hue = None

    if verbose: print('-'*20)
    return df_rel, hue

def show_average_lines(ax, df, show_columns, _metric) -> plt.axes:
    colors = sns.color_palette().as_hex()
    xlims = ax.get_xlim()
    if len(show_columns) == 1:
        df_avg = df.groupby(show_columns).mean().reset_index()
        order = ax.get_legend_handles_labels()[1]
        # order = [int(s) for s in order]
        df_avg[show_columns[0]] = df_avg[show_columns[0]].astype(str)
        # df_avg = df_avg.sort_values(by=show_columns, key=lambda col: order.index(col))
        df_avg[show_columns[0]] = pd.Categorical(df_avg[show_columns[0]], ordered=True, categories=order)
        df_avg = df_avg.sort_values(show_columns).reset_index()
        df_avg = df_avg.reset_index()
        for row in df_avg[show_columns + [_metric]].itertuples(): # index, nqubits, mse_val
            ax.plot(xlims, [row[2], row[2]], color = colors[row[0]], linewidth = 1)
    return ax

def which_moi_does_this_column_have(f, want_to_plot, moi_col) -> None:
    df_full = pd.read_parquet(f)
    for _c in df_full[moi_col].unique():
        df_c = df_full[df_full[moi_col] == _c]
        print(f'{moi_col}={_c}:', df_c[want_to_plot].unique())
    return

def df_select_values(df_sel, col, values):
    df_sel['keep_col'] = False
    for v in values:
        df_sel.loc[df_sel[col] == v, 'keep_col'] = True
    df_sel = df_sel[df_sel['keep_col'] == True]
    df_sel = df_sel.drop('keep_col', axis=1)
    return df_sel

def df_select_values_range(df_sel, col, low, high):
    assert high > low, 'high must be greater than low'
    # low, high are both inclusive
    df_sel['keep_col'] = False
    df_sel.loc[df_sel[col] >= low, 'keep_col'] = True
    df_sel.loc[df_sel[col] > high, 'keep_col'] = False
    df_sel = df_sel[df_sel['keep_col'] == True]
    df_sel = df_sel.drop('keep_col', axis=1)
    return df_sel

def sparse_ticks(_ticks, _labels, _num=20):
    # sort both lists by ticks low to high
    _ticks, _labels = zip(*sorted(zip(_ticks, _labels)))
    last_tick = _ticks[0]
    tick_diff = abs(_ticks[0] - _ticks[-1]) / _num
    new_ticks = [_ticks[0]]
    new_labels = [_labels[0]]
    for npos, pos in enumerate(_ticks):
        if abs(pos - last_tick) > tick_diff:
            new_ticks.append(pos)
            new_labels.append(_labels[npos])
            last_tick = pos
    return new_ticks, new_labels

def average_df(_df, _passable_params):
    cols_avg = [c for c in _df.columns.to_list() if c in _passable_params]
    return _df.groupby(cols_avg).mean().reset_index()