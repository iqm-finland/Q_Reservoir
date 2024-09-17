"""Contains some loops to uun simulations for all the plots in the report, but for more random seeds.

Nothing special about it. You can use your own loops to run simulations.
All runs are saved automatically in a DataFrame in experiments/results.

Instructions:
    1. Select the model_type (class) to use.
    2. Select the model (dict) and data (dict) parameters to use.
    3. Select the rseeds_data (list) to use.
    4. Select the column (str) to vary.
    5. Select the values (list or str) to try.
    6. Run the script.
"""
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from datetime import date
import pathlib
import time
import datetime
import copy

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

module_path = pathlib.Path(__file__).parent.parent.parent.resolve().__str__() # qrc_surrogate
sys.path.append(module_path) # module: qrc_surrogate

from src.feedforward import QExtremeLearningMachine, CPolynomialFeedforward
from src.rewinding import QRewindingRC, QRewindingStatevectorRC
from src.data import DataSource
import sweeps_paras


def calc_error_source(model_type, data_params, model_params, rseeds_data):
    print(f'calc_error_source', datetime.datetime.now())
    data_copy = copy.deepcopy(data_params)
    model_copy = copy.deepcopy(model_params)
    total = len(rseeds_data) * 2 * 2
    cnt = 0
    t0 = time.time()
    for rseed_data in rseeds_data:
        for use_true_y_in_val in [True, False]:
            for train_equal_val in [True, False]:
                model_copy['use_true_y_in_val'] = use_true_y_in_val
                data_copy['train_equal_val'] = train_equal_val
                data_copy['rseed_data'] = rseed_data
                data = DataSource(**data_copy)
                rnn = model_type(**model_copy)
                rnn.run(data)
                now = datetime.datetime.now()
                print(f'{cnt}/{total} ({int((time.time() - t0) / 60)}min): rs{rseed_data}: MAPE={rnn.mape_val:.2e} MSE={rnn.mse_val:.2f} s{rnn.traintime:.2f} ({now.hour}:{now.minute})')
                cnt += 1
    return

def calc_noise(model_type, values, data_params, model_params, rseeds_data):
    print('calc_noise', datetime.datetime.now())
    data_copy = copy.deepcopy(data_params)
    model_copy = copy.deepcopy(model_params)
    data_copy.pop('rseed_data')
    # t1_list = np.arange(5, 10, .2)
    if values in ['stored', 'all']:
        f = f'{module_path}/experiments/results/{model_type().model_name}.parquet'
        df = pd.read_parquet(f)
        values = df['t1'].unique().tolist()
    total = len(rseeds_data) * len(values)
    cnt = 0
    t0 = time.time()
    for rseed_data in rseeds_data:
        for t1 in values:
            model_copy['sim'] = 'thermal'
            model_copy['t1'] = t1
            data = DataSource(**data_copy, rseed_data=rseed_data)
            rnn = model_type(**model_copy)
            rnn.run(data)
            now = datetime.datetime.now()
            print(f'{cnt}/{total} ({int((time.time() - t0) / 60)}min): rs{rseed_data} n{t1:.2f}: MAPE={rnn.mape_val:.2e} MSE={rnn.mse_val:.2f} s{rnn.traintime:.2f} ({now.hour}:{now.minute})')
            cnt += 1
    return

def calc_data_column(model_type, col, values, data_params, model_params, rseeds_data):
    print(f'Calc {col}', datetime.datetime.now())
    data_copy = copy.deepcopy(data_params)
    model_copy = copy.deepcopy(model_params)
    if values in ['stored', 'all']:
        f = f'{module_path}/experiments/results/{model_type().model_name}.parquet'
        df = pd.read_parquet(f)
        values = df[col].unique().tolist()
    total = len(rseeds_data) * len(values)
    cnt = 0
    t0 = time.time()
    for rseed_data in rseeds_data:
        for v in values:
            data_copy[col] = v
            data_copy['rseed_data'] = rseed_data
            data = DataSource(**data_copy)
            rnn = model_type(**model_copy)
            rnn.run(data)
            try:
                v_str = f'{v:.2f}'
            except:
                v_str = str(v)
            now = datetime.datetime.now()
            print(f'{cnt}/{total} ({int((time.time() - t0) / 60)}min): rs{rseed_data} v{v_str}: MAPE={rnn.mape_val:.2e} MSE={rnn.mse_val:.2f} s{rnn.traintime:.2f} ({now.hour}:{now.minute})')
            cnt += 1
    return

def calc_column(model_type, col, values, data_params, model_params, rseeds_data):
    """Vary the value of a parameter and calculate the error for each value.
    
    Args:
        model_type (class):
            The model class to use. 
            model_type = QRewindingStatevectorRC, QRewindingRC, QExtremeLearningMachine, CPolynomialFeedforward.
        col (str): 
            The name of the parameter to vary. 
            Options are all parameters in data_params and model_params, 
            i.e. parameters of DataSource and model_type, 
            i.e. columns in the results DataFrame.
        values (list or str):
            The values to try.
            If 'stored' or 'all', use the values stored in the results DataFrame.
        data_params (dict):
            The parameters of DataSource (other settings which are not varied).
        model_params (dict):
            The parameters of model_type (other settings which are not varied).
        rseeds_data (list):
            The seeds of the data to use.
    """
    print(f'Calc {col}', datetime.datetime.now())
    data_copy = copy.deepcopy(data_params)
    model_copy = copy.deepcopy(model_params)
    if values in ['stored', 'all']:
        f = f'{module_path}/experiments/results/{model_type().model_name}.parquet'
        df = pd.read_parquet(f)
        values = df[col].unique().tolist()
    total = len(rseeds_data) * len(values)
    cnt = 0
    t0 = time.time()
    for rseed_data in rseeds_data:
        for v in values:
            model_copy[col] = v
            data_copy['rseed_data'] = rseed_data
            data = DataSource(**data_copy)
            rnn = model_type(**model_copy)
            rnn.run(data)
            try:
                v_str = f'{v:.2f}'
            except:
                v_str = str(v)
            now = datetime.datetime.now()
            print(f'{cnt}/{total} ({int((time.time() - t0) / 60)}min): rs{rseed_data} v{v_str}: MAPE={rnn.mape_val:.2e} MSE={rnn.mse_val:.2f} s{rnn.traintime:.2f} ({now.hour}:{now.minute})')
            cnt += 1
    return




#####################################################

if __name__ == '__main__':
    # select model 
    model_type = QRewindingStatevectorRC
    model = model_type()
    f = f'{module_path}/experiments/results/{model.model_name}.parquet'
    df = pd.read_parquet(f)

    # The 'short list' used for most plots in the report 
    # is from sweeps_paras and includes 6 seeds.
    # The 'long list' includes 77 seeds
    # a.k.a. all seeds if every tried in any context
    # rseeds_of_interest = sweeps_paras.rseed_data_list # short list
    rseeds_of_interest = df['rseed_data'].unique().tolist() # long list

    # get base parameters for the model and data
    model_of_interest, data_of_interest = sweeps_paras.get_paras(model)

    # sweep over some parameters
    calc_error_source(model_type, data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    calc_column(model_type, col='shots', values='all', data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    calc_column(model_type, col='ftype', values='all', data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    calc_column(model_type, col='measaxes', values='all', data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    calc_column(model_type, col='nyfuture', values='all', data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    calc_column(model_type, col='qctype', values=['ising', 'ising_nn', 'empty'], data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    # nqubits roughly takes 24h on a M2 Macbook Pro
    calc_column(model_type, col='nqubits', values='all', data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)

    # # I do not think it is very necessary to get more statistics for these plots
    # # but feel free to uncomment the following lines if you have time
    # calc_data_column(col='rseed', values=rseeds_circ, data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    # calc_data_column(col='nepisodes', values=range(10, 90, 10), data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    # calc_column(col='ising_wmax', values=np.arange(0, 15, .1), data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    # calc_column(col='ising_h', values=np.arange(-1, 10, .1), data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    # calc_column(col='ising_jmax', values=np.arange(0, 5, .1), data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    # # noise can take multiple days
    # calc_noise(values='stored', data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)

    # how many steps should we repeat, if we do not want to restart from the beginning?
    moi = copy.deepcopy(model_of_interest)
    moi['restarting'] = False
    for nqubits in [3, 4, 5, 6, 7, 8, 9]:
        moi['nqubits'] = nqubits
        calc_column(model_type, col='lookback', values='all', data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)

    # when increasing ftype (feature type = expectation values) or nqubits,
    # we have the problem, that we estimate an increasing number of features
    # from a fixed number of shots = counts
    # as a comparison we can look at calculating the expectation values exactly
    # direclty from the statevector
    moi = copy.deepcopy(model_of_interest)
    moi['sim_sampling'] = 'exact'
    calc_column(model_type, col='ftype', values='all', data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
    calc_column(model_type, col='nqubits', values='all', data_params=data_of_interest, model_params=model_of_interest, rseeds_data=rseeds_of_interest)
