# -*- coding: utf-8 -*-
"""Parent classes for the quantum reservoir computing models.

None of the classes here work on their own.
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, Aer
from qiskit.providers.fake_provider import FakeProvider
from qiskit_aer.noise import NoiseModel
from qiskit.opflow import I, X, Y, Z
import qiskit_aer

#For running on IQM simulations/hardware.
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import time
import inspect
import itertools
import pandas as pd
import pathlib
import sys
import os

sys.path.insert(1, os.path.abspath('..')) # module: qrc_surrogate

from src.data import DataSource
from src.helpers import mse, mape, nrmse, correlation, unnormalize
from src.circuits import \
    random_circuit, random_clifford_circuit, random_ht_circuit, ising_circuit, ising_circuit_natural, ising_circuit_naturalH, \
    jiri_circuit, spread_circuit, hcnot_circuit, \
    circularcnot_circuit, efficientsu2_circuit, downup_circuit, fake_circuit, random_hpcnot_circuit
from src.noisemodel import thermal_model

module_path = pathlib.Path(__file__).parent.parent.resolve().__str__()

class PredictionModelBase:
    """Parent class.
    Does not work on its own.

    Args:
        log (bool) = True:
            Whether to log results and model parameters in `results/model.parquet`.
        rseed (int) = 0:
            Random seed for reproducibility.
        washout (int) = 0:
            Number of timesteps to discard at the beginning of the dataset.
        add_x_as_feature (bool) = True:
            Whether to add the raw input x(t) as a feature to the fitter.
        regression_model (str) = 'ridge':
            Which regression model to use. Options: 'ridge', 'lasso', 'elasticnet', 'linear'.
        regression_alpha (float) = 0.1:
            Regularization parameter for ridge, lasso, elasticnet.
        regression_l1 (float) = 0.5:
            L1 ratio for elasticnet.
        fitter (str) = 'sklearn':
            Which fitter to use. Options: 'sklearn', 'sklearn_poly'.
        poly_degree (int) = 3:
            Degree of polynomial features to add to the fitter (only for fitter='sklearn_poly').
        nyfuture (int) = 1:
            | How many future steps to predict.
            | Only the next predicted step is used, but it can make predictions better.
            | During training the fitter uses all next step predictions, but the reported loss (e.g. self.mse_train) is only with respect to the next step.
            | E.g. 2 means predicting {y(t), y(t+1)} at each step t.
        delete_future_y (bool) = True:
            Only has an impact if nyfuture > 1. 
            Only affects the fitting during the training phase. Predictions are unaffected.
            At the last few timesteps we will predict future steps beyond the dataset.
            Solution 1 (True): delete the last few timeseps, at the cost of having a few less training samples. Works better in practice.
            Solution 2 (False): make up the values beyond the dataset, e.g. by assuming the dataset stays constant.
    """
    model_name = 'prediction_model_base'

    def __init__(
        self,
        log,
        rseed,
        washout,
        add_x_as_feature,
        # predicting multiple steps forward
        nyfuture, 
        delete_future_y,
        # fitter
        fitter,
        regression_model,
        regression_alpha,
        regression_l1,
        poly_degree,
    ) -> None:
        self.rseed = rseed
        self.log = log
        self.washout = washout
        self.washout_eff = washout
        self.add_x_as_feature = add_x_as_feature
        # predicting multiple steps forward
        self.nyfuture = max(1, nyfuture)
        self.delete_future_y = delete_future_y
        # fitter 
        self.fitter = fitter
        self.regression_model = regression_model
        self.regression_alpha = regression_alpha
        self.regression_l1 = regression_l1
        self.poly_degree = poly_degree
        # --
        # results
        self.mse_val = None
        self.mse_train = None
        self.mape_val = None
        self.mape_train = None
        self.nrmse_val = None
        self.nrmse_train = None 
        self.traintime = None
        # --
        # Data
        self.data = None
        self.dimx = None
        self.dimy = None
        self.xmax = None # used for angle encoding
        self.xmin = None
        self.nepisodes = None
        # --
        # Stored calculations
        self.dimf = None
        self.weights = None
        # input
        self.xtrain = None 
        self.xval = None
        # features 
        self.ftrain = None
        self.fval = None 
        # predictions based on features, normalized if data was normalized
        self.ytrain = None
        self.yval = None 
        # unnormalized to original data range, if data was normalized
        self.ytrain_nonorm = None
        self.yval_nonorm = None
        # 
        # Only has an effect if data was normed. 
        # If True, data is unnormalized for e.g. self.mse_train
        self.use_unnormed_error_metrics = True 
    
    def run(self, dataobj):
        """Setup, train, val, and evaluate.

        Args:
            dataobj (DataSource): Instance of DataSource class.
        """
        if not isinstance(dataobj, DataSource):
            raise Warning('Please pass instance of DataSource class to <class>.run().')
        self.start_time = time.time()
        if self.log:
            self._init_logging()
        self.data = dataobj
        self._set_fitter()
        self._set_data_dims(self.data)
        self._set_unitary_and_meas()
        self.train()
        self.traintime = int(time.time() - self.start_time)
        self.val()
        if self.log:
            self._log_results()
        return self
    
    def _set_unitary_and_meas(self):
        """Define quantum circuit at every timestep.

        Returns:
            self.unistep (qiskit.QuantumCircuit): unitary at every timestep.
        """
        raise Warning('Please overwrite set_circuit method in child class.')
    
    def _init_logging(self):
        """Set filename and check if we can write to it."""
        self.savefile = f'{module_path}/experiments/results/{self.model_name}.parquet'
        # make sure we can write to file
        if not os.path.exists(self.savefile):
            df = pd.DataFrame()
            # df.to_hdf(self.savefile, key='df', mode='a')
            try:
                df.to_parquet(self.savefile)
            except:
                raise FileNotFoundError(f'| file: {self.savefile} \n| module: {module_path}')
        if not os.path.exists(self.savefile):
            raise FileNotFoundError(f'{self.savefile} doesnt exist')
    
    def _log_results(self):
        """Write results, model settings, and data settings to file."""
        result = self.get_results_and_parameters()
        # add data info
        result.update(self.data.get_hyperparameters())
        # add date 
        result['date'] = int(time.strftime("%Y%m%d"))
        # df["date"] = pd.to_numeric(df["date"])
        # result['sim'] = result['sim'].name()
        # load
        df = pd.read_parquet(self.savefile)
        rows = df.shape[0]
        # save
        # df = df.append(result, ignore_index=True)
        df = pd.concat(
            [df, pd.DataFrame.from_records([result])], 
            ignore_index=True
        )
        assert df.shape[0] == rows+1, f'df{df.shape[0]}, r{rows}'
        # df.to_hdf(self.savefile, key='df', mode='a')
        try:
            df.to_parquet(self.savefile)
        except:
            print('df')
            print(df)
            print('result')
            print(result)
            raise Warning('''
                Couldnt save logging dataframe. Probably because a new result or class argument was added.
                This causes a bunch of NaNs to appear in the dataframe. 
                Fix by opening a juptyer notebook and running:
                module_path = os.path.abspath(os.path.join('..')) 
                savefile = module_path + f"/experiments/results/<classname>.parquet"
                df = pd.read_parquet(savefile)
                df.fillna({"<newcolumn>": <somevalue>}, inplace=True)]
                df.to_parquet(savefile)
            ''')
        return
    
    def _set_data_dims(self, data_obj):
        """Deduce dimensions of x and y data from Data object.
        Args:
            data_obj (DataSource): Instance of DataSource class.
        """
        self.dimx = data_obj.dimx 
        self.dimy = data_obj.dimy 
        self.xmin = data_obj.xmin
        self.xmax = data_obj.xmax
        self.ymin = data_obj.ymin
        self.ymax = data_obj.ymax
        self.nepisodes = len(data_obj.xtrain)
        return
    
    def _policy(self, y, e=None, step=None, train=False, offset=0):
        """A placeholder policy which returns the action x(t) given the observation y(t-1).

        Actions are taken from the training or testing data.
        
        Args:
            y (np.ndarray): Observation y(t-1). Not used, since we are using data.
            e (int): Episode number. Used to pull data from self.data.
            step (int): Step number. Used to pull data from self.data.
            offset (int): xyoffset. If offset=1, dont do the last prediction.

        Returns:
            np.ndarray: Action x(t).
            Or False if step is out of bounds.
        """
        if train:
            xdata = self.data.xtrain[e]
        else: 
            xdata = self.data.xval[e]
        if np.max(step) >= (np.shape(xdata)[0] - offset):
            return False
        else:
            x = xdata[step]
            return x.reshape(1, -1)
        
    def _judge_train(self, ypred, ytrue, ypred_nonorm, ytrue_nonorm):
        """Calculate error metrics between prediction and true.
        Write results to self.

        Args:
            ytrue, ypred (np.ndarray): Prediction and true values.
            ytrue_nonorm, ypred_nonorm (np.ndarray): Prediction and true values, unnormalized.
        """
        if self.use_unnormed_error_metrics:
            ypred = ypred_nonorm
            ytrue = ytrue_nonorm
        self.corr_train = correlation(ypred, ytrue)
        self.mse_train = mse(ypred, ytrue)
        # self.mse_train = mse(ypred_nonorm, ytrue_nonorm)
        self.mape_train = mape(ypred, ytrue)
        self.nrmse_train = nrmse(ypred, ytrue)
        return

    def _judge_val(self, ypred, ytrue, ypred_nonorm, ytrue_nonorm):
        """Calculate error metrics between prediction and true.
        Write results to self.

        Args:
            ytrue, ypred (np.ndarray): Prediction and true values.
            ytrue_nonorm, ypred_nonorm (np.ndarray): Prediction and true values, unnormalized.
        """
        if self.use_unnormed_error_metrics:
            ypred = ypred_nonorm
            ytrue = ytrue_nonorm
        self.corr_val = correlation(ypred, ytrue)
        self.mse_val = mse(ypred, ytrue)
        # self.mse_train = mse(ypred_nonorm, ytrue_nonorm)
        self.mape_val = mape(ypred, ytrue)
        self.nrmse_val = nrmse(ypred, ytrue)
        return
    
    def show_worst_val_episodes(self, plot=True, nepisodes=5):
        """Plot and return the validation episodes with the worst (highest) MSE.

        Args:
            plot (bool): If True, plot the episodes.
            nepisodes (int): Number of episodes to plot.

        Returns:
            train_errors, val_errors (np.ndarray): episode number and MSE. shape(episodes, 2).
        """
        # train
        train_errs = np.zeros(shape=(len(self.data.xtrain), 2))
        for e in range(len(self.data.xtrain)):
            train_errs[e, 0] = e
            train_errs[e, 1] = mse(
                a=self.ytrain[e][self.washout_eff:], 
                b=self.data.ytrain_nonorm[e][self.washout_eff:]
            )
        train_errs = train_errs[train_errs[:,1].argsort()[::-1]]
        # val
        val_errs = np.zeros(shape=(len(self.data.xval), 2))
        for e in range(len(self.data.xval)):
            val_errs[e, 0] = e
            val_errs[e, 1] = mse(
                a=self.yval[e][self.washout_eff:], 
                b=self.data.yval_nonorm[e][self.washout_eff:]
            )
        val_errs = val_errs[val_errs[:,1].argsort()[::-1]]
        if plot:
            worst_episodes = np.asarray(val_errs[:nepisodes, 0], dtype=int)
            worst_in = [self.data.xval_nonorm[int(e)] for e in worst_episodes]
            worst_y = [self.data.yval_nonorm[int(e)] for e in worst_episodes]
            worst_predictions = [self.yval[int(e)] for e in worst_episodes]
            self._plot_prediction(
                ypred=worst_predictions,
                ytrue=worst_y,
                x=worst_in, 
                title='Worst prediction error (val)', 
                nepisodes=nepisodes,
            )
        return train_errs, val_errs
    
    def print_results(self, results=None, full=False):
        """Print prediction metrics to console.
        
        Args:
            results (dict): Dictionary of results. If None, use self.get_results().
            full (bool): If True, print all metrics. If False, print only nrmse and corr.
        """
        if results is None:
            results = self.get_results()
        if full:
            for key, value in results.items():
                if value is not None:
                    print(f'{key}:' + (' '*(15-len(key))) + f'{value:.2e}')
        else:
            for key in ['nrmse_train', 'nrmse_val', 'corr_train', 'corr_val']:
                print(f'{key}:' + (' '*(15-len(key))) + f'{results[key]:.2e}')
        return

    def get_results(self):
        """Dictionary of evaluation metrics of fit and prediction.

        Includes Squared Pearson correlation coefficient, Mean Squared Error,
        Mean Absolute Percentage Error, Normalized Root Mean Squared Error.
        """
        return {
            'corr_train': self.corr_train, 
            'corr_val': self.corr_val,
            # mean squared error
            'mse_train': self.mse_train, 
            'mse_val': self.mse_val,
            'mape_train': self.mape_train,
            'mape_val': self.mape_val,
            'nrmse_train': self.nrmse_train,
            'nrmse_val': self.nrmse_val,
            'traintime': self.traintime,
            # 'omrsquared': self.omrsquared, # only for statsmodels fitter
            # 'pvalue': self.pvalue,
        }
    
    def get_hyperparameters(self):
        """Return a dictionary of passable parameters and their current values.
        Use to recreate instance:
        
        .. code-block:: python

            params = model.get_hyperparameters()
            model_copy = Model(**params)
        """
        params = inspect.getfullargspec(self.__init__).args[1:] 
        params = set(params) # only keep each item once
        dictstring = ''.join(f'\'{p}\': self.{p}, ' if p!='sim' else f'\'{p}\': str(self.{p}), ' for p in params)
        dictstring = '{' + dictstring + '}'
        paramsdict = eval(dictstring)
        # paramsdict.update(self.kwargs)
        return paramsdict
    
    def get_results_and_parameters(self):
        """Return a dictionary of results and parameters."""
        res = self.get_results()
        paras = self.get_hyperparameters()
        res.update(paras)
        return res
    
    def _plot_prediction(self, ypred, ytrue, x=None, title=None, nepisodes=5):
        """Plot predictions and true values for a handful of episodes.
        Args:
            ypred, ytrue, x: list[np.array]: (episode, steps, dimy)
            nepisodes (int): maximum number of episodes to be plotted
        """
        sns.set_style("whitegrid") # "darkgrid" whitegrid
        cs = sns.color_palette()
        # add step number
        gap = 5 # gap between episodes
        steps = [np.arange(0, np.shape(ep)[0], 1) for ep in ypred] # (steps, 1)
        for s in range(1, len(steps)):
            steps[s] += steps[s-1][-1] + 1 + gap
        # remove washout from predictions
        steps_wo_washout = steps.copy() # without washout
        ypred_wo_washout = ypred.copy() # without washout
        for ep in range(len(ypred)):
            steps_wo_washout[ep] = steps_wo_washout[ep][self.washout_eff:] 
            ypred_wo_washout[ep] = ypred_wo_washout[ep][self.washout_eff:, :]
        assert np.shape(ypred_wo_washout[0])[1] == self.dimy
        fig = plt.figure()
        for e in range(min(len(ypred_wo_washout), nepisodes)):
            if x is not None:
                lx = plt.plot(steps[e], x[e], c=cs[0], ls='--')
            ly = plt.plot(steps[e], ytrue[e], c=cs[1])
            lpred = plt.plot(steps_wo_washout[e], ypred_wo_washout[e], c=cs[2])
        # plt.plot(np.vstack(data.ytrain), c=cs[2], label='y true')
        plt.title(title)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        labels = [r'$a$', r'$s^{true}$', r'$s^{pred}$'] if x is not None else ['y true', 'y predicted']
        plt.legend(labels=labels)
        plt.xlabel('Timesteps')
        plt.ylabel('')
        # plt.legend()
        plt.show()
        return fig
    
    def plot_train(self, title=None, nepisodes=5, px=True, use_normed_values=False):
        """Calls :func:`._plot_prediction` with training data.

        Args:
            title (str): title of plot. If None, use 'Train'.
            nepisodes (int): number of episodes to be plotted.
            px: bool: if to plot x (input)
        """
        x = self.data.xtrain if px else None
        title = 'Train  (randomly selected episodes)' if title is None else title
        if use_normed_values:
            x = self.data.xtrain if px else None
            fig = self._plot_prediction(ypred=self.ytrain, ytrue=self.data.ytrain, x=x, title=title, nepisodes=nepisodes)
        else:
            x = self.data.xtrain_nonorm if px else None
            fig = self._plot_prediction(ypred=self.ytrain_nonorm, ytrue=self.data.ytrain_nonorm, x=x, title=title, nepisodes=nepisodes)
        return fig

    def plot_val(self, title=None, nepisodes=5, px=True, use_normed_values=False):
        """Calls :func:`._plot_prediction` with validation data.

        Args:
            title (str): title of plot. If None, use 'Validation'.
            nepisodes (int): number of episodes to be plotted.
            px: bool: if to plot x (input)
        """
        title = 'Validation (randomly selected episodes)' if title is None else title
        if use_normed_values:
            x = self.data.xval if px else None
            fig = self._plot_prediction(ypred=self.yval, ytrue=self.data.yval, x=x, title=title, nepisodes=nepisodes)
        else:
            x = self.data.xval_nonorm if px else None
            fig = self._plot_prediction(ypred=self.yval_nonorm, ytrue=self.data.yval_nonorm, x=x, title=title, nepisodes=nepisodes)
        return fig
    
    def plot_worst_val_episodes(self, title=None, nepisodes=5, px=True, use_normed_values=False):
        """Calls :func:`._plot_prediction` with the episodes in the validation data that have the largest MSE.

        Args:
            title (str): title of plot. If None, use 'Validation'.
            nepisodes (int): number of episodes to be plotted.
            px: bool: if to plot x (input)
        """
        # calculate MSE for each episode
        mse_episodes = np.zeros((len(self.yval_nonorm), 2))
        for ne, ye_pred in enumerate(self.yval_nonorm):
            mse_episodes[ne, 0] = ne
            mse_episodes[ne, 1] = mse(ye_pred, self.data.yval_nonorm[ne])
        # sort: largest MSE at the top
        # mse_episodes = mse_episodes[mse_episodes[:, 1].argsort()] 
        mse_episodes = mse_episodes[mse_episodes[:, 1].argsort()[::-1]] 
        indices = mse_episodes[:nepisodes, 0].astype(int)
        print('MSE per episode\n', mse_episodes)
        title = 'Validation (worst episodes)' if title is None else title
        if use_normed_values:
            x = self.data.xval if px else None
            fig = self._plot_prediction(
                ypred=[self.yval[i] for i in indices], 
                ytrue=[self.data.yval[i] for i in indices], 
                x=None if x is None else [x[i] for i in indices], 
                title=title, nepisodes=nepisodes
            )
        else:
            x = self.data.xval_nonorm if px else None
            fig = self._plot_prediction(
                ypred=[self.yval_nonorm[i] for i in indices], 
                ytrue=[self.data.yval_nonorm[i] for i in indices], 
                x=None if x is None else [x[i] for i in indices], 
                title=title, nepisodes=nepisodes
            )
        return fig
    
    def plot_best_val_episodes(self, title=None, nepisodes=5, px=True, use_normed_values=False):
        """Calls :func:`._plot_prediction` with the episodes in the validation data that have the smallest MSE.

        Args:
            title (str): title of plot. If None, use 'Validation'.
            nepisodes (int): number of episodes to be plotted.
            px: bool: if to plot x (input)
        """
        # calculate MSE for each episode
        mse_episodes = np.zeros((len(self.yval_nonorm), 2))
        for ne, ye_pred in enumerate(self.yval_nonorm):
            mse_episodes[ne, 0] = ne
            mse_episodes[ne, 1] = mse(ye_pred, self.data.yval_nonorm[ne])
        # sort: largest MSE at the top
        mse_episodes = mse_episodes[mse_episodes[:, 1].argsort()] 
        # mse_episodes = mse_episodes[mse_episodes[:, 1].argsort()[::-1]] 
        indices = mse_episodes[:nepisodes, 0]
        indices = indices.astype(int)
        print('MSE per episode\n', mse_episodes)
        title = 'Validation (best episodes)' if title is None else title
        if use_normed_values:
            x = self.data.xval if px else None
            fig = self._plot_prediction(
                ypred=[self.yval[i] for i in indices], 
                ytrue=[self.data.yval[i] for i in indices], 
                x=None if x is None else [x[i] for i in indices], 
                title=title, nepisodes=nepisodes
            )
        else:
            x = self.data.xval_nonorm if px else None
            fig = self._plot_prediction(
                ypred=[self.yval_nonorm[i] for i in indices], 
                ytrue=[self.data.yval_nonorm[i] for i in indices], 
                x=None if x is None else [x[i] for i in indices], 
                title=title, nepisodes=nepisodes
            )
        return fig

    def _plot_features(self, features, x, nepisodes=5, title=''):
        """Plot features and input data for a handful of episodes.

        Args:
            features, x: list[np.array]: (episode, steps, dim)
            nepisodes (int): maximum number of episodes to be plotted
        """
        cs = list(sns.color_palette())
        steps = [np.arange(0, np.shape(ep)[0], 1) for ep in features]
        for s in range(1, len(steps)):
            steps[s] += steps[s-1][-1] + 1
        fig, ax = plt.subplots()
        # ax2 = ax.secondary_yaxis('right')
        ax2 = ax.twinx()
        xdims = np.shape(x[0])[1]
        fdims = np.shape(features[0])[1]
        nepisodes_shown = min(len(features), nepisodes)
        for e in range(nepisodes_shown):
            for d in range(xdims):
                lx = ax2.plot(steps[e], x[e][:, d], c=cs[(0+d) % len(cs)], ls='--')
            for d in range(fdims):
                ly = ax.plot(steps[e], features[e][:, d], c=cs[(xdims+d) % len(cs)])
        plt.title(title)
        plt.legend(
            [ax2.get_lines()[(d*nepisodes_shown)] for d in range(xdims)] \
            + [ax.get_lines()[(d*nepisodes_shown)] for d in range(fdims)], 
            [f'x{d}' for d in range(xdims)] + [f'f{d}' for d in range(fdims)])
        # plt.legend()
        plt.show
        return
    
    def plot_train_features(self, title=None, nepisodes=1, px=True):
        """Calls :func:`._plot_features` with train data.

        Args:
            title (str): title of plot. If None, use 'Training features'.
            nepisodes (int): number of episodes to be plotted.
            px: bool: if to plot x (input)
        """
        x = self.data.xtrain if px else None
        title = 'Training features' if title is None else title
        self._plot_features(features=self.ftrain, x=x, title=title, nepisodes=nepisodes)
        return

    def plot_val_features(self, title=None, nepisodes=1, px=True):
        """Calls :func:`._plot_features` with val data.
        Args:
            title (str): title of plot. If None, use 'Training features'.
            nepisodes (int): number of episodes to be plotted.
            px: bool: if to plot x (input)
        """
        x = self.data.xval if px else None
        title = 'validating features' if title is None else title
        self._plot_features(features=self.fval, x=x, title=title, nepisodes=nepisodes)
        return
    
    def _set_fitter(self):
        """Set regression model and polynomial features."""
        match self.regression_model:
            # fit_intercept adds constant
            case'regression':
                self.weights = LinearRegression(fit_intercept=True)
            case 'ridge':
                self.weights = Ridge(alpha=self.regression_alpha, fit_intercept=True)
            case 'lasso':
                self.weights = Lasso(alpha=self.regression_alpha, fit_intercept=True)
            case 'elastic':
                # alpha = 1, l1 = .5
                self.weights = ElasticNet(alpha=self.regression_alpha, l1_ratio=self.regression_l1)
            case _:
                raise Warning(f'Invalid model {self.regression_model}')
        self.poly_feat = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
        return
    
    def _predict(self, out):
        """Predicts y from features (one episode)."""
        # prediction (out & weights -> y)
        match self.fitter:
            case 'statsmodels':
                out = sm.add_constant(out)
                y_predicted = np.dot(out, self.weights)
                return y_predicted.reshape((np.shape(y_predicted)[0], self.dimy))
            case 'sklearn':
                y_predicted = self.weights.predict(out)
                return y_predicted.reshape((np.shape(y_predicted)[0], self.dimy))
            case 'sklearn_poly':
                poly_features = self.poly_feat.fit_transform(out)
                y_predicted = self.weights.predict(poly_features)
                return y_predicted.reshape((np.shape(y_predicted)[0], self.dimy))
            case _:
                raise Warning(f'Invalid fitter {self.fitter}')
    
    def _fit(self, features, ys) -> None: 
        r"""Fit features to desired output using Ordinary least squares (OLS).

        OLS is linear regression with a quadratic loss (mean squared error),
        and can be optimized in a single step by solving a linear system of equations.

        Problem is convex -> the optimum lies at gradient zero.

        https://en.wikipedia.org/wiki/Linear_least_squares

        .. math::

            s^{pred} = f  w

        .. math::

            L_{MSE} &= \vert s^{pred} - s^{true} \vert^2 \\
            &= \left( f  w - s^{true} \right)^T \left( f  w - s^{true} \right) \\
            &= (s^{true})^T s^{true} - (s^{true})^T f w - w^T f^T s^{true} + w^T f^T f w\\
        
        Minimizing the loss function with respect to the weights:
        
        .. math::

            dL_{MSE}/dw = 2 (-f^T s^{true} + f^T f w)

        Optimal weights:

        .. math::

            w_{opt} = (f^T f)^{-1} f^T s^{true}

        Arguments:
            features: list of reservoir outputs for each episode. (episode, steps, dimf)
            ys: list of true values trying to fit to for each episode. (episode, steps, dim_y)
        """
        # remove washout periods
        features_train = [out[self.washout_eff:, :] for out in features]
        ys_train = [y[self.washout_eff:, :] for y in ys]
        match self.fitter:
            case 'statsmodels':
                # combine episodes into one big episode
                ytrain_all = np.vstack(ys_train)
                ftrain_all = np.vstack(features_train)
                # print('with washout_eff', np.shape(np.vstack(features)), '. without', np.shape(out_train))
                ftrain_all = sm.add_constant(ftrain_all) # (episodes*(steps-washout_eff), dimy+1)
                # fit
                model = sm.OLS(ytrain_all, ftrain_all)
                fitting = model.fit()
                self.weights = fitting.params
                # evaluate fit
                self.omrsquared = 1 - fitting.rsquared
                self.pvalue = fitting.f_pvalue
            case 'sklearn':
                ytrain_all = np.vstack(ys_train)
                ftrain_all = np.vstack(features_train)
                # linear regression 
                # fit_intercept adds constant
                self.weights.fit(ftrain_all, ytrain_all)
            case 'sklearn_poly':
                ytrain_all = np.vstack(ys_train)
                ftrain_all = np.vstack(features_train)
                # linear regression with polynomial features
                poly_features = self.poly_feat.fit_transform(ftrain_all)
                self.weights.fit(poly_features, ytrain_all)
            case _:
                raise Warning(f'Invalid fitter {self.fitter}')
        return self


class QuantumBase:
    r"""Base class for quantum circuits.

    Args:
        nqubits (int): 
            number of qubits
        qinit (str): 
            initialize qubit states before first step.
            'h': Hadamard
        nlayers (int): 
            number of unitaries at each step.
        ftype (int):
            | which expectation values from circuit measurements to use as features.
            | -1: all of the below.
            | 0: probability vector of possible bitstrings 
            |   (dimf = 2^n, n = num measured qubits).
            | 1: single qubit expectations <z> 
            |   (dimf = n). 
            | 2: two qubit expectations <zz> 
            |   (dimf = (n choose 2)). also includes 1.
            | 3: three qubit expectations <zzz> 
            |   (dimf = (n choose 3)). also includes 1 and 2.
            | 4: four qubit expectations <zzzz> 
            |   (dimf = (n choose 4)). also includes 1, 2, and 3.
        qctype (str): 
            | ising, ising_nn, ising_ladder, ising_natural, ising_natural_nn, ising_natural_ladder, ising_hadamard, ising_hadamard_nn, ising_hadamard_ladder random, random_hpcnot, random_clifford, random_ht, jiri, spread_input, hcnot, circularcnot, downup, efficientsu2, swapstack, empty, mockqc.
            | ising: fully connected transverse field ising model from :func:`circuits.ising_circuit()`.
            | ising_nn: nearest neighbour connection
            | ising_ladder: ladder up and down
            | ising_natural (prefix): use zz two qubit gates and x as single qubit rotations.
            | ising_hadamard (prefix): append hadamards to the start and end of the zz / x gates, so that it reproduces the original ising model.
            |   ising_t (float): time in e^-iHt
            |   ising_jmax (float): J in (-jmax, jmax) (spin-spin couplings)
            |   ising_wmax (float): D in (-wmax, wmax) (disorder strength)
            |   ising_h (float): h (transverse field)
            |   ising_random (bool): if True, random J, D
            | See :mod:`circuits` for details.
        shots (int) = 2^13 (8192):
            number of shots for measurement.
        sim (str) = 'aer_simulator':
            simulator for quantum circuit. 
            Can be an "IQMBackend" object for either simulation or actual experiment.
            'aer_simulator' for a noiseless simulation.
            Can be anything in
            :code:`available_backends = {b.name():b.configuration().n_qubits for b in FakeProvider().backends()}` for a noisy circuit. 
            'thermal' for the noisy :func:`noisemodel.thermal_model` (faster than the backends from FakeProvider).
        t1 (float) = None:
            if sim=thermal, mean T1 value in microseconds. mean T2 = 7/5 mean T1. Default: 50.
        sim_precision (str) = 'single':
            precision of simulator.
        enctype (str) = 'angle':
            | 'angle': :math:`Rx(x) |0> = cos(x/2) |0> + sin(x/2) |1>`, x in (0, Pi). 
            | 'ryrz': :math:`Rz(x^2 + Pi/4) Ry(x + Pi/4) |+>`, x in (0, Pi).
        encangle (float) = 1:
            if=1, x is in (0, Pi).
        nenccopies (int) = 1:
            number of copies of the input to be input into the circuit in parallel at each step.
        encaxes (str) = 'xyz':
            axes for encoding. If nenccopies=1, only the first character is used.
        measaxes (str) = 'xyz':
            axes for measurement. If three characters, circuit is run 3*shots times.
    """

    def __init__(
        self,
        nqubits,
        qctype,
        qinit ,
        nlayers,
        ftype,
        enctype,
        encangle,
        nenccopies,
        encaxes, 
        measaxes, 
        shots,
        sim,
        t1,
        sim_precision,
        sim_method,
        # ising circuit
        ising_t,
        ising_jmax,
        ising_h,
        ising_wmax,
        ising_random,
        ising_jpositive,
        ising_wpositive,
    ) -> None:
        self.qc = None
        self.dimxqc = None
        self.dimx_wo_copies = None
        self.qctype = qctype
        self.qinit = qinit
        self.enctype = enctype
        self.encangle = encangle
        self.ftype = ftype
        self.nqubits = nqubits # total qubits in the circuit
        self.nenccopies = nenccopies
        self.measaxes = measaxes 
        self.encaxes = encaxes 
        self.shots = shots
        self.nlayers = nlayers

        if isinstance(sim,IQMBackendBase):
            self.backend=sim
            print('IQM')
            print(self.backend)

        # --
        elif type(sim) == qiskit_aer.backends.aer_simulator.AerSimulator or sim == 'aer_simulator':
            # noiseless
            t1 = float('inf')
            sim_method = 'statevector'
            self.backend = AerSimulator(method = sim_method)
            print('Aer')
            print(self.backend)
        elif sim == 'thermal':
            sim_method = 'density_matrix'
            noise_model = thermal_model(
                nqubits=self.nqubits, 
                t1_mean=t1,
                t2_mean=t1*7/5,
            )
            self.backend = AerSimulator(method = sim_method, noise_model = noise_model)
        elif type(sim) == str:
            t1 = -1
            sim_method = 'density_matrix'
            provider = FakeProvider()
            backend = provider.get_backend(sim)
            noise_model = NoiseModel.from_backend(backend)
            self.backend = AerSimulator(method = sim_method, noise_model = noise_model)
        else:
            print('else')
            t1 = -1
            sim_method = 'density_matrix'
            noise_model = NoiseModel.from_backend(sim)
            self.backend = AerSimulator(method = sim_method, noise_model = noise_model)
        try:
            self.t1 = t1
        except:
            None
        try:
            self.sim_method = sim_method
        except:
            None
        try:
            self.sim = sim
        except:
            None
        try:
            self.sim_precision = sim_precision
            self.backend.set_options(precision=sim_precision)
        except:
            None
        
        

        # set during calculation
        self.quni = None # list of qubits to apply unitary to
        self.qin = None # list of qubits which encode x
        self.qmeas = None # list of qubits which are measured
        self.ngates = None
        self.unistep = None
        # --
        # Ising circuit parameters
        self.ising_t = ising_t 
        self.ising_jmax = ising_jmax
        self.ising_h = ising_h
        self.ising_wmax = ising_wmax
        self.ising_random = ising_random
        self.ising_jpositive = ising_jpositive
        self.ising_wpositive = ising_wpositive
        #
        self.preloading = 0
        # measaxes
        if type(measaxes) == int:
            self.measaxes = 'zyx'[:measaxes]
        if type(encaxes) == int:
            self.encaxes = 'xyz'[:encaxes]

    def _angle_encoding(self, episode, dmin=None, dmax=None):
        """Encode episode into range (0, pi).
        If using multiple encoding copies nenccopies > 1, only feed in the first copy.

        Args:
            episode (ndarray(steps, dimx)): input data
            dmin (list or ndarray(dimx)): min value in episode. defaults to xmin
            dmax (list or ndarray(dimx)): max value in episode. defaults to xmax

        Returns:
            encoded_input (ndarray(steps, dimx)): Same shape as episode
        """
        # rescale (min, max) -> (0, 2pi)
        new_min = 0
        new_max = np.pi * self.encangle
        # current min, max of the data
        dmin = self.xmin if dmin is None else dmin
        dmax = self.xmax if dmax is None else dmax
        # assert np.shape(dmin)[0] == self.dimxqc, f'{np.shape(dmin)} {self.dimxqc}'
        # output array
        encoding = np.zeros(np.shape(episode))
        assert np.shape(episode)[1] == int(self.dimxqc / self.nenccopies), \
            f'e{np.shape(episode)} == {int(self.dimxqc / self.nenccopies)}. dx{self.dimx} dxqc{self.dimxqc} enc{self.nenccopies}'
        for dim in range(np.shape(episode)[1]):
            # [steps, dim]
            encoding[:,dim] = ((episode[:,dim] - dmin[dim]) * (new_max - new_min)) / (dmax[dim] - dmin[dim]) + new_min
            # enc_seq = MinMaxScaler(feature_range=(new_min, new_max))
        # where data was initialized to outside of min max range (e.g. initialized to 0)
        # this would lead to very large angles
        # instead set angles to new_min
        dminarr = np.asarray(dmin).reshape(1, -1)
        encoding[episode < dminarr] = new_min
        return encoding

    def _binary_encoding(self, episode):
        """Encode x sequence into binary {0, pi}.
        
        Args:
            episode (ndarray(steps, dimx)): input data

        Returns:
            binary (ndarray(steps, dimx)): Same shape as episode
        """
        binary = np.where(episode > (np.max(episode) / 2), 1, 0) 
        return binary * np.pi

    def _set_unitary(self):
        """Set the unitary for each timestep (self.unistep).
        Just calls the appropriate function from :mod:`circuits` based on self.qctype.
        """
        match self.qctype:
            case 'random':
                self.unistep = random_circuit(len(self.quni), len(self.quni)*3, self.rseed)
            case 'random_hpcnot':
                self.unistep = random_hpcnot_circuit(len(self.quni), len(self.quni)*3, self.rseed)
            case 'random_clifford':
                self.unistep = random_clifford_circuit(len(self.quni), len(self.quni)*3, self.rseed)
            case 'random_ht':
                self.unistep = random_ht_circuit(len(self.quni), len(self.quni)*3, self.rseed)
            case 'jiri':
                assert len(self.quni) >= 3
                qc = QuantumCircuit(len(self.quni))
                for q in range (0, len(self.quni)-2):
                    qc.append(jiri_circuit(), qargs=self.quni[q:q+3])
                self.unistep = qc
            case 'spread_input':
                self.unistep = spread_circuit(len(self.quni), sources=self.qin, rseed=self.rseed)
            case 'hcnot':
                self.unistep = hcnot_circuit(len(self.quni), rseed=self.rseed)
            case 'circularcnot':
                if self.nlayers is None:
                    self.nlayers = 2
                self.unistep = circularcnot_circuit(len(self.quni), rseed=self.rseed)
            case 'downup':
                self.unistep = downup_circuit(nqubits=len(self.quni), rseed=self.rseed)
            case 'efficientsu2':
                self.unistep = efficientsu2_circuit(len(self.quni), rseed=self.rseed)
            case 'swapstack':
                # assumption: first qubit is input qubit
                qc = QuantumCircuit(len(self.quni))
                qc.swap(qc.num_qubits-1, 0)
                for q in reversed(range(1, qc.num_qubits-1)):
                    qc.swap(q, q+1)
                self.unistep = qc
            case 'ising':
                self.unistep = ising_circuit(
                    nqubits = len(self.quni), 
                    t = self.ising_t, 
                    jmax = self.ising_jmax, 
                    h = self.ising_h, 
                    wmax = self.ising_wmax, 
                    random = self.ising_random,
                    jpositive = self.ising_jpositive,
                    wpositive = self.ising_wpositive,
                    rseed = self.rseed, 
                )
            case 'ising_nn':
                self.unistep = ising_circuit(
                    nqubits = len(self.quni), 
                    t = self.ising_t, 
                    jmax = self.ising_jmax, 
                    h = self.ising_h, 
                    mode = 'nn',
                    wmax = self.ising_wmax, 
                    random = self.ising_random,
                    jpositive = self.ising_jpositive,
                    wpositive = self.ising_wpositive,
                    rseed = self.rseed, 
                )
            case 'ising_ladder':
                self.unistep = ising_circuit(
                    nqubits = len(self.quni), 
                    t = self.ising_t, 
                    jmax = self.ising_jmax, 
                    h = self.ising_h, 
                    wmax = self.ising_wmax, 
                    mode = 'ladder',
                    random = self.ising_random,
                    jpositive = self.ising_jpositive,
                    wpositive = self.ising_wpositive,
                    rseed = self.rseed, 
                )
            case 'ising_natural':
                self.unistep = ising_circuit_natural(
                    nqubits = len(self.quni), 
                    t = self.ising_t, 
                    jmax = self.ising_jmax, 
                    h = self.ising_h, 
                    wmax = self.ising_wmax, 
                    random = self.ising_random,
                    jpositive = self.ising_jpositive,
                    wpositive = self.ising_wpositive,
                    rseed = self.rseed, 
                )
            case 'ising_natural_nn':
                self.unistep = ising_circuit_natural(
                    nqubits = len(self.quni), 
                    t = self.ising_t, 
                    jmax = self.ising_jmax, 
                    h = self.ising_h, 
                    mode = 'nn',
                    wmax = self.ising_wmax, 
                    random = self.ising_random,
                    jpositive = self.ising_jpositive,
                    wpositive = self.ising_wpositive,
                    rseed = self.rseed, 
                )
            case 'ising_natural_ladder':
                self.unistep = ising_circuit_natural(
                    nqubits = len(self.quni), 
                    t = self.ising_t, 
                    jmax = self.ising_jmax, 
                    h = self.ising_h, 
                    wmax = self.ising_wmax, 
                    mode = 'ladder',
                    random = self.ising_random,
                    jpositive = self.ising_jpositive,
                    wpositive = self.ising_wpositive,
                    rseed = self.rseed, 
                )
            case 'ising_hadamard':
                self.unistep = ising_circuit_naturalH(
                    nqubits = len(self.quni), 
                    t = self.ising_t, 
                    jmax = self.ising_jmax, 
                    h = self.ising_h, 
                    wmax = self.ising_wmax, 
                    random = self.ising_random,
                    jpositive = self.ising_jpositive,
                    wpositive = self.ising_wpositive,
                    rseed = self.rseed, 
                )
            case 'ising_hadamard_nn':
                self.unistep = ising_circuit_naturalH(
                    nqubits = len(self.quni), 
                    t = self.ising_t, 
                    jmax = self.ising_jmax, 
                    h = self.ising_h, 
                    mode = 'nn',
                    wmax = self.ising_wmax, 
                    random = self.ising_random,
                    jpositive = self.ising_jpositive,
                    wpositive = self.ising_wpositive,
                    rseed = self.rseed, 
                )
            case 'ising_hadamard_ladder':
                self.unistep = ising_circuit_naturalH(
                    nqubits = len(self.quni), 
                    t = self.ising_t, 
                    jmax = self.ising_jmax, 
                    h = self.ising_h, 
                    wmax = self.ising_wmax, 
                    mode = 'ladder',
                    random = self.ising_random,
                    jpositive = self.ising_jpositive,
                    wpositive = self.ising_wpositive,
                    rseed = self.rseed, 
                )
            case 'empty':
                self.unistep = QuantumCircuit(len(self.quni))
            case 'mockqc': # for circuit illustration
                self.unistep = fake_circuit(len(self.quni))
            case _:
                raise Warning('Invalid qc_type')
        return
    
    def _add_input_to_qc(self, qc, angles, step) -> None:
        """Encode input onto circuit.
        
        Args:
            qc (QuantumCircuit): circuit to add input to
            angles (ndarray(steps, dimx)): encoded input data
            step (int): current step. Used to index angles, 0 indexed.
        """
        qn = 0 # qubit counter for loop
        for c in range(self.nenccopies):
            for d in range(int(self.dimxqc / self.nenccopies)): # dimx
                if self.enctype == 'angle':
                    match self.encaxes[c % len(self.encaxes)]:
                        case 'x':
                            qc.rx(theta=angles[step, d], qubit=self.qin[qn])
                        case 'y':
                            qc.ry(theta=angles[step, d], qubit=self.qin[qn])
                        case 'z':
                            qc.rz(phi=angles[step, d]/2, qubit=self.qin[qn])
                        case _:
                            raise Warning(f'Invalid encaxes={self.encaxes}')
                elif self.enctype == 'ryrz': 
                    # self.encaxes = ['ry', 'rz']
                    qc.h(qubit=self.qin[qn])
                    qc.ry(np.arctan(angles[step, d]) + np.pi/4, self.qin[qn])
                    qc.rz(np.arctan(angles[step, d]**2) + np.pi/4, self.qin[qn])
                else:
                    raise Warning(f'Invalid enctype {self.enctype}')
                qn += 1
        return

    def _get_circuit_measurements(self, qc) -> dict:
        """Transpile and run circuit on simulator.
        Args:
            qc (QuantumCircuit): circuit to measure
        Return:
            Counts of measurements (dict)
        """
        compiled_qc = transpile(qc, self.backend) #currently uses the default Qiskit transpilation.
        job = self.backend.run(compiled_qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        return counts
    
    def _step_meas_to_step_features(self, counts_step, features_step,Ex=False):
        """Expectation values of measurements are appended to features_step.

        Args:
            counts_step (dict): counts which are shall be processed to features
            features_step (list): features are appended to this list
            Ex: if True, gives expectation values; if False, they are linear shifts.
        Returns:
            features_step
        """
        shots_step = np.sum(c for _, c in counts_step.items())
        nmeas = len(list(counts_step)[0])
        if self.ftype in [0, -1] and nmeas < 15:
            # probability vector of possible bitstrings
            # bitstrings = [bin(n)[2:].zfill(nmeas) for n in range(2**nmeas)]
            prob_vec = np.zeros([1, 2**nmeas])
            for bitstring, count in counts_step.items():
                pos = int(bitstring, 2) # convert to integer
                prob = count / shots_step
                prob_vec[0, pos] += prob
            features_step.append(prob_vec[self.preloading:])
        if self.ftype >= 1 or self.ftype == -1:
            
            # single qubit expectations <z>
            # len(bitstring) = dimx * steps
            meas = np.array([
                [
                    int(bit) * count 
                    for bit in bitstring
                ] 
                for bitstring, count in counts_step.items()
            ])
            mean = np.sum(meas, axis=0) / shots_step # (steps*nmeas, )
            if nmeas==2: #Special thing...
                mean = np.flip(mean) # counter reversed qiskit ordering
            # mean = s1d1, s1d2, ..., s2d1, s2d2, ...
            mean = mean.reshape(1, nmeas) # (steps, nmeas)
            if Ex:
                mean=1-2*mean
            
            # outm = mean.reshape(nmeas, np.shape(angles)[0]).T
            features_step.append(mean[self.preloading:])
        if (self.ftype >= 2 or self.ftype == -1) and nmeas >= 2:
            
            # two qubit expectations <zz>
            pairs = [p for p in itertools.combinations(range(nmeas), 2)]
            meas_vec = np.zeros([1, len(pairs)])
            for bitstring, count in counts_step.items():
                for cp, p in enumerate(pairs):
                    if bitstring[p[0]] == bitstring[p[1]]:
                        # + if same sign
                        meas_vec[0, cp] += count / shots_step
                    else:
                        # - if opposite sign
                        if Ex:
                            meas_vec[0, cp] -= count / shots_step
            features_step.append(meas_vec[self.preloading:])
        if (self.ftype >= 3 or self.ftype == -1) and nmeas >= 3:
            
            # three qubit expectations <zzz>
            triples = [p for p in itertools.combinations(range(nmeas), 3)]
            meas_vec = np.zeros([1, len(triples)])
            even = ['011', '110', '101', '000']
            odd = ['001', '010', '100', '111']
            for bitstring, count in counts_step.items():
                for ct, triple in enumerate(triples):
                    bit_triple = ''.join(bitstring[c] for c in triple)
                    if bit_triple in even:
                        # + if same sign
                        meas_vec[0, ct] += count / shots_step
                    elif bit_triple in odd:
                        # - if opposite sign
                        if Ex:
                            meas_vec[0, ct] -= count / shots_step
                        pass
                    else:
                        raise Warning(f'Unknown bit_triple {bit_triple} {triple}')
            features_step.append(meas_vec[self.preloading:])
        if (self.ftype >= 4 or self.ftype == -1) and nmeas >= 4:
            # four qubit expectations <zzzz>
            quads = [p for p in itertools.combinations(range(nmeas), 4)]
            meas_vec = np.zeros([1, len(quads)])
            even = ['0000', '0011', '0101', '0110', '1001', '1010', '1100', '1111']
            odd = ['0001', '0010', '0100', '0111', '1000', '1011', '1101', '1110']
            for bitstring, count in counts_step.items():
                for cqu, quad in enumerate(quads):
                    bit_quad = ''.join(bitstring[c] for c in quad)
                    if bit_quad in even:
                        # + if same sign
                        meas_vec[0, cqu] += count / shots_step
                    elif bit_quad in odd:
                        # - if opposite sign
                        # meas_vec[0, cqu] -= count / shots_step
                        pass
                    else:
                        raise Warning(f'Unknown bit_quad {bit_quad}')
            features_step.append(meas_vec[self.preloading:])
        return features_step
    
    def _step_meas_to_step_features_sv(self, sv, features_step):
        """Expectation values of a statevector are appended to features_step.

        Args:
            sv (qi.Statevector): statevector which are shall be processed to features
            features_step (list): features are appended to this list

        Returns:
            features_step
        """
        nmeas = sv.num_qubits
        indices = list(range(nmeas))
        if self.ftype in [0, -1] and nmeas < 15:
            ev_vec = np.real_if_close(sv.probabilities()).reshape(1, -1)
            features_step.append(ev_vec)
        if self.ftype >= 1 or self.ftype == -1:
            # single qubit expectations <z>
            op = Z
            ev_vec = np.zeros(nmeas)
            for nq in range(nmeas):
                ev_vec[nq] = np.real_if_close(sv.expectation_value(op, qargs=[nq]))
            # real and rescale to (0, 1)
            ev_vec = ((ev_vec + 1) / 2).reshape(1, -1)
            features_step.append(ev_vec)
        if (self.ftype >= 2 or self.ftype == -1) and nmeas >= 2:
            # two qubit expectations <zz>
            op = Z^Z
            # ev_vec = np.zeros(int(nmeas*(nmeas-1)/2)) # (nmeas choose 2)
            # for nq in range(nmeas-1):
            #     for nq2 in range(nq+1, nmeas):
            #         ev_vec[nq*(nmeas-1) + nq2] = np.real_if_close(sv.expectation_value(op, qargs=[nq, nq2]))
            pair_indices = [list(i) for i in itertools.combinations(indices, 2)]
            ev_vec = np.zeros(len(pair_indices))
            for ntq, pair in enumerate(pair_indices):
                ev_vec[ntq] = np.real_if_close(sv.expectation_value(op, qargs=pair))
            # real and rescale to (0, 1)
            ev_vec = ((ev_vec + 1) / 2).reshape(1, -1)
            features_step.append(ev_vec)
        if (self.ftype >= 3 or self.ftype == -1) and nmeas >= 3:
            # three qubit expectations <zzz>
            op = Z^Z^Z
            # ev_vec = np.zeros(int(nmeas*(nmeas-1)*(nmeas-2)/6)) # (nmeas choose 3)
            # for nq in range(nmeas-2):
            #     for nq2 in range(nq+1, nmeas-1):
            #         for nq3 in range(nq2+1, nmeas):
            #             ev_vec[nq*(nmeas-2)*(nmeas-1) + nq2*(nmeas-2) + nq3] = np.real_if_close(sv.expectation_value(op, qargs=[nq, nq2, nq3]))
            triple_indices = [list(i) for i in itertools.combinations(indices, 3)]
            ev_vec = np.zeros(len(triple_indices))
            for ntq, triple in enumerate(triple_indices):
                ev_vec[ntq] = np.real_if_close(sv.expectation_value(op, qargs=triple))
            # real and rescale to (0, 1)
            ev_vec = ((ev_vec + 1) / 2).reshape(1, -1)
            features_step.append(ev_vec)
        if (self.ftype >= 4 or self.ftype == -1) and nmeas >= 4:
            # four qubit expectations <zzzz>
            op = Z^Z^Z^Z
            # ev_vec = np.zeros(int(nmeas*(nmeas-1)*(nmeas-2)*(nmeas-3)/24)) # (nmeas choose 4)
            # for nq in range(nmeas-3):
            #     for nq2 in range(nq+1, nmeas-2):
            #         for nq3 in range(nq2+1, nmeas-1):
            #             for nq4 in range(nq3+1, nmeas):
            #                 ev_vec[nq*(nmeas-3)*(nmeas-2)*(nmeas-1) + nq2*(nmeas-3)*(nmeas-2) + nq3*(nmeas-3) + nq4] = np.real_if_close(sv.expectation_value(op, qargs=[nq, nq2, nq3, nq4]))
            quad_indices = [list(i) for i in itertools.combinations(indices, 4)]
            ev_vec = np.zeros(len(quad_indices))
            for ntq, quad in enumerate(quad_indices):
                ev_vec[ntq] = np.real_if_close(sv.expectation_value(op, qargs=quad))
            # real and rescale to (0, 1)
            ev_vec = ((ev_vec + 1) / 2).reshape(1, -1)
            features_step.append(ev_vec)
        return features_step
    



class StepwiseModelBase:
    """Provides training loop for :class:`src.feedforward.CPolynomialFeedforward` 
    and :class:`src.feedforward.QExtremeLearningMachine`.

    Args:
        xyoffset (int 0 or 1) = 1: 
            | Number of steps to offset x and y.
            | Set to 1 for NARMA and reactor data.
            | xyoffset=0: :math:`y_t = f(x_t, y_{t-1})`
            | xyoffset=1: :math:`y_{t+1} = f(x_t, y_t)`
        set_past_y_to_0 (bool) = True: 
            When using y(t<0) values as input for the first few steps, set y(t<0)=0 or y(t<0)=y(0).
        use_true_y_in_val (bool) = False: 
            | Option for debugging or evaluating the model.
            | If the model takes in previously predicited (possible wrong) y as input, 
            | error can accumulate over time (drift).
            | True: use true previous y values in validation. 
            | Should lead to mse_train=mse_val if combined with DataSource(train_equal_val=True).
    """

    def __init__(
        self,
        # StepwiseModelBase
        xyoffset,
        set_past_y_to_0,
        use_true_y_in_val,
        lookback_max,
    ) -> None:
        self.xyoffset = xyoffset
        self.washout_eff = self.washout + self.xyoffset
        self.set_past_y_to_0 = set_past_y_to_0
        self.use_true_y_in_val = use_true_y_in_val
        self.lookback_max = lookback_max
        return
    
    def _init_t_inputs(self, x0, y0, steps_max):
        """Lookup table of past x and y values to repeat the past steps.

        It's own class in order to assure that train and val are doing the same thing.

        Args:
            x0, y0: first steps in this episode
            steps_max: maximum number of steps in this episode
        
        Returns:
            xe_lookback, ye_lookback (np.array(steps_max+lookback, dim))
        """
        # initial condition of the system
        if self.lookback_max:
            ylookback = 0
            xlookback = 0
        else:
            ylookback = self.ylookback
            xlookback = self.xlookback
        if self.set_past_y_to_0:
            # past y
            ye_lookback = np.zeros((ylookback+steps_max, self.dimy))
            # past x
            xe_lookback = np.zeros((xlookback+steps_max, self.dimx))
        else:
            # past y
            ye_lookback = np.vstack([np.full((ylookback, self.dimy), y0), np.zeros((steps_max, self.dimy))])
            # past x
            xe_lookback = np.vstack([np.full((xlookback, self.dimx), x0), np.zeros((steps_max, self.dimy))])
        if self.xyoffset == 1:
            if self.lookback_max:
                ye_lookback[0] = y0
            else:
                ye_lookback[self.ylookback-1] = y0
        return xe_lookback, ye_lookback
    
    def train(self):
        """Train the model on the training data for all episodes.

        Run model for all steps and collect features.
        If model takes in y as input, then use true y.
        After all steps and episodes, train linear fitter.
        """
        if self.lookback_max:
            ylookback = 1
            xlookback = 1
        else:
            ylookback = self.ylookback
            xlookback = self.xlookback
        self.xtrain = [] # (episodes, steps, dimx)
        self.ftrain = [] # (episodes, steps, dimf)
        for e, xe in enumerate(self.data.xtrain):
            # steps = np.shape(xe)[0]
            steps_max = 200
            xe_lookback, ye_lookback = self._init_t_inputs(
                x0=self.data.xtrain[e][0], y0=self.data.ytrain[e][0], 
                steps_max=steps_max
            )
            # save for evaluation
            xe = [] # actions in this episode
            fe = [] # features in this episode
            t_in = 0 # counter for step in loop (t-self.offsetxy) 
            for t_pred in range(self.xyoffset, steps_max):
                # get x(t-1) (action) 
                # x0 = self.data.xtrain[e][step-1].reshape(1, -1) 
                x0 = self._policy(0, e=e, step=t_in, train=True, offset=self.xyoffset)
                # save current input
                xe_lookback[t_in+xlookback-1] = x0
                if x0 == False:
                    break
                input_t = self._get_input_t(xe_lookback=xe_lookback, ye_lookback=ye_lookback, t_pred=t_pred, t_in=t_in)
                f1 = self._t_input_to_t_features(input_t=input_t, x0=x0, t_pred=t_pred)
                # get true output
                y1 = self.data.ytrain[e][t_pred]
                # save current ouput
                ye_lookback[t_in+ylookback] = y1 
                # save 
                xe.append(x0)
                fe.append(f1) # (steps, 1, dimf)
                t_in += 1
            # episode over
            if self.xyoffset == 1:
                xe += [self.data.xtrain[e][-1]] # add last step
            self.xtrain.append(np.vstack(xe))
            self.ftrain.append(np.vstack(fe)) # (episodes, steps-xyoffset, dimf)
            assert np.allclose(self.xtrain[e], self.data.xtrain[e]), f'{self.xtrain[e] - self.data.xtrain[e]}'
        # all episodes over
        self.dimf = np.shape(f1)[1]
        # train the fitter with all features and targets (y true)
        # all episodes stacked into one big episode
        self.delete_last_steps = steps_max
        # features_all = np.vstack(self.ftrain) # ((steps-1)*episodes, dimf)
        ytrain_all = np.vstack([ye[self.xyoffset:] for ye in self.data.ytrain]) # ((steps-xyoffset)*episodes, dimy)
        if self.nyfuture > 1:
            # at the last tmax-(nyfuture-1) timesteps we will predict ys that are beyond the data set
            # solution 1: delete tmax-(nyfuture-1) timesteps
            # solution 2: set ytrue to some default value (e.g. the last known value)
            if self.delete_future_y == True:
                # for fitting, remove the last step. for prediction, all steps are used
                self.delete_last_steps = 1 - self.nyfuture
                ytrain_all = []
                for ye in self.data.ytrain:
                    ye_extended = ye.copy()
                    for future in range(1, self.nyfuture):
                        # ytrain_all = [1, 2, 3] -> yfuture = [2, 3, 1]
                        yefuture = np.roll(ye, shift=-future, axis=0) 
                        # set future steps to last known value
                        yefuture[-future:] = yefuture[-future-1]
                        # -> yfuture = [2, 3, 3]
                        ye_extended = np.hstack([ye_extended, yefuture])
                    ytrain_all.append(ye_extended[self.xyoffset:self.delete_last_steps]) # remove first and last step
                ytrain_all = np.vstack(ytrain_all)
            else:
                # for fitting, set the last step to the last step in the data
                ytrain_all = np.vstack([ye[self.xyoffset:] for ye in self.data.ytrain]) # ((steps-1)*episodes, dimy)
                ytrain_all_extended = ytrain_all.copy()
                for future in range(1, self.nyfuture):
                    # ytrain_all = [1, 2, 3] -> yfuture = [2, 3, 1]
                    yfuture = np.roll(ytrain_all, shift=-future, axis=0) 
                    # set future steps to last known value
                    yfuture[-future:] = yfuture[-future-1]
                    # -> yfuture = [2, 3, 3]
                    ytrain_all_extended = np.hstack([ytrain_all, yfuture])
                ytrain_all = ytrain_all_extended
        features_all = np.vstack([f[:self.delete_last_steps] for f in self.ftrain]) # ((steps-nyfuture))*episodes, dimf)
        # fit all features to all ys
        self.weights.fit(features_all, ytrain_all)
        # make predictions
        # (episodes, steps, dimy)
        self.ytrain = [
            self.weights.predict(fe)[:, :self.dimy].reshape(np.shape(fe)[0], self.dimy) # step 1, ..., n
            for fe in self.ftrain
        ]
        if self.xyoffset > 0: # add step 0
            self.ytrain = [
                np.vstack([
                    self.data.ytrain[e][0:self.xyoffset], # add step 0
                    ye,
                ])
                for e, ye in enumerate(self.ytrain)
            ]
        # (episodes, steps-1, dimf) -> (episodes, steps, dimf)
        # self.ftrain = [np.vstack([fe[0], fe]) for fe in self.ftrain]
        # assert that all have the same amount of steps
        assert np.shape(self.ytrain[-1])[0] == np.shape(self.xtrain[-1])[0], f'y{np.shape(self.ytrain[-1])} x{np.shape(self.xtrain[-1])} != data{np.shape(self.data.xtrain[-1])}'
        # unnormalize
        if self.data.ynorm == 'norm':
            self.ytrain_nonorm = unnormalize(data=self.ytrain, dmin=self.data.ymin_nonorm, dmax=self.data.ymax_nonorm)
        else:
            self.ytrain_nonorm = self.ytrain
        # remove washout period
        self._judge_train(
            ypred = np.vstack([y[self.washout_eff:] for y in self.ytrain]),
            ytrue = np.vstack([y[self.washout_eff:] for y in self.data.ytrain]),
            ypred_nonorm = np.vstack([y[self.washout_eff:] for y in self.ytrain_nonorm]),
            ytrue_nonorm = np.vstack([y[self.washout_eff:] for y in self.data.ytrain_nonorm]),
        )
        return
    
    def val(self, infmode='data', nepisodes=None):
        """Train for all episodes.

        Run model for one step, predict y(t+1) from features.
        If model takes in y as input, then use predicted (possibly wrong) y.
        """
        if self.lookback_max:
            ylookback = 1
            xlookback = 1
        else:
            ylookback = self.ylookback
            xlookback = self.xlookback
        self.xval = [] # (episodes, steps, dimx)
        self.fval = [] # (episodes, steps, dimf)
        self.yval = [] # (episodes, steps, dimy)
        if infmode == 'data':
            nepisodes = len(self.data.xval)
        for e in range(nepisodes):
            # steps = np.shape(xe)[0]
            steps_max = 200
            xe_lookback, ye_lookback = self._init_t_inputs(
                x0=self.data.xval[e][0], y0=self.data.yval[e][0], 
                steps_max=steps_max
            )
            # save for evaluation
            xe = [] # actions in this episode
            fe = [] # features in this episode
            ye = []
            t_in = 0 # counter for step in loop (step-self.offsetxy) 
            for t_pred in range(self.xyoffset, steps_max):
                # get x(t-1) (current action)
                x0 = self._policy(0, e=e, step=t_in, train=False, offset=self.xyoffset)
                # save current input
                xe_lookback[t_in+xlookback-1] = x0
                if x0 == False:
                    break
                input_t = self._get_input_t(xe_lookback=xe_lookback, ye_lookback=ye_lookback, t_pred=t_pred, t_in=t_in)
                # make features out of x(t) and previous {y(t-lookback)}
                f1 = self._t_input_to_t_features(input_t=input_t, x0=x0, t_pred=t_pred)
                # predict output
                y1 = max(
                    min(
                        self.weights.predict(f1)[:, :self.dimy], 
                        np.asarray(self.ymax)
                    ), 
                    np.asarray(self.ymin)
                ).reshape(1, -1)
                if self.use_true_y_in_val: # debugging only
                    # get true output
                    ye_lookback[t_in+ylookback] = self.data.yval[e][t_pred]
                else:
                    # save current ouput
                    ye_lookback[t_in+ylookback] = y1 
                # save 
                xe.append(x0)
                fe.append(f1) # (steps, 1, dimf)
                ye.append(y1)
                t_in += 1
            # episode over
            if self.xyoffset > 0:
                xe += [self.data.xval[e][-1]] # add last step
                fe = [fe[0]] + fe # add first step
                ye = [self.data.yval[e][0]] + ye # add first step
            self.xval.append(np.vstack(xe))
            # add first step
            self.fval.append(np.vstack(fe)) # (episodes, steps, dimf)
            self.yval.append(np.vstack(ye)) # (episodes, steps, dimy) 
            assert np.allclose(self.xval[e], self.data.xval[e]), f'{self.xval[e] - self.data.xval[e]}'
            # assert np.shape(self.xval[-1])[0] == np.shape(self.fval[-1])[0], f'{np.shape(self.xval[-1])} {np.shape(self.fval[-1])} : {np.shape(self.data.xval[e])}'
            assert np.shape(self.xval[-1])[0] == np.shape(self.yval[-1])[0], f'{np.shape(self.xval[-1])} {np.shape(self.yval[-1])} : {np.shape(self.data.xval[e])}'
        # all episodes over
        # unnormalize
        if self.data.ynorm == 'norm':
            self.yval_nonorm = unnormalize(data=self.yval, dmin=self.data.ymin_nonorm, dmax=self.data.ymax_nonorm)
        else:
            self.yval_nonorm = self.yval
        # remove washout period
        self._judge_val(
            ypred = np.vstack([y[self.washout_eff:] for y in self.yval]), 
            ytrue = np.vstack([y[self.washout_eff:] for y in self.data.yval]),
            ypred_nonorm = np.vstack([y[self.washout_eff:] for y in self.yval_nonorm]),
            ytrue_nonorm = np.vstack([y[self.washout_eff:] for y in self.data.yval_nonorm]),
        )
        if self.use_true_y_in_val:
            print(f'Validation error == Train error: {np.allclose(self.mse_train, self.mse_val)}. Relative difference: {np.abs((self.mse_train - self.mse_val)/self.mse_train)}')
        return
    
    def _t_input_to_t_features(self, input_t, x0, t_pred=None):
        """Features for this steps, given some input.

        Args:
            input_t (nd.array): inputs for this step. shape(1, any).
            x0 (nd.array). current x. gets added as feature. shape(1, dimx).
            t_pred (int) = None: optional. in some model used to save the circuit at a specific step.
        """
        raise Warning('Dont use the StepwiseModelBase class directly. Create a child class and overwrite the t_input_to_t_features method')
    
    def _get_input_t(self, xe_lookback, ye_lookback, t_pred, t_in):
        """Get input for this step from lookup tables.

        Indices:
            x[t_in:t_in+self.xlooback]
            y[t_in:t_in+self.xlooback]

        Arguments:
            xe_lookback (nd.array): lookup table for past actions in this episode. shape(steps_max, dimx).
            ye_lookback (nd.array): lookup table for past outputs in this episode. shape(steps_max, dimy).
            t_pred (int): timestep to be predicted.
            t_in (int): step in loop.
        """
        raise Warning('Dont use the StepwiseModelBase class directly. Create a child class and overwrite the get_input_t method')

