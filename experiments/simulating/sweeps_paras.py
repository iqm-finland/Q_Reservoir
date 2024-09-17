from qiskit import QuantumCircuit, transpile
# from qiskit.providers.aer import QasmSimulator
# from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit_aer import AerSimulator, Aer
from qiskit.providers.fake_provider import FakeProvider, FakeManila, FakeToronto, FakeJakartaV2
from qiskit_aer.noise import NoiseModel
import qiskit.quantum_info as qi

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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

module_path = pathlib.Path(__file__).parent.parent.parent.resolve().__str__() # qrc_surrogate
sys.path.append(module_path) # module: qrc_surrogate

from src.feedforward import QExtremeLearningMachine
from src.rewinding import QRewindingRC, QRewindingStatevectorRC
from src.data import DataSource


def print_best_paras(experiment):
    """Prints out the parameters of the experiments with the lowest validation MSE.
    
    Args:
        experiment (QRewindingRC or QExtremeLearningMachine):
            instance of model class
    
    Returns:
        None

    Usage:
        .. code-block:: python

            model = QRewindingRC # or QExtremeLearningMachine
            experiment = model()
            print_best_paras(experiment)   
    """
    f = f'{module_path}/experiments/results/{experiment.model_name}.parquet'
    df = pd.read_parquet(f)

    df = df[df['xtype'] == 'tsetpoint']

    # drop columns with just one value
    for c in df.columns.to_list():
        try:
            # some data types like lists this doesnt work (not hashable)
            if len(df[c].unique()) <= 1:
                df = df.drop(c, axis=1)
        except:
            print(f'couldnt drop {c}')
            continue

    passable_model_params = inspect.getfullargspec(experiment.__init__).args[1:]
    passable_data_params = inspect.getfullargspec(DataSource().__init__).args[1:]

    # model params which are still left
    used_passable_model_params = [m for m in df.columns.to_list() if m in passable_model_params] 
    used_passable_data_params = [m for m in df.columns.to_list() if m in passable_data_params] 
    # print('model params left:')
    # for p in model_params_red:
    #     print(p, df[p].unique())

    df = df.reset_index()
    ibest_list = df.nsmallest(5, ['mse_val']).index.to_list()

    for ibest in ibest_list:
        best_params = df.iloc[ibest].to_dict()
        best_model_params = {p:v for p,v in best_params.items() if p in used_passable_model_params}
        best_data_params = {p:v for p,v in best_params.items() if p in used_passable_data_params}
        best_data_params.pop('rseed_data')
        print('best_model_params:\n', f"{best_params['mse_val']:.2f}", best_model_params)
    return


# choose parameters we want to investigate for the rest of the file
def get_paras(experiment):
    """
    Usage:
        model = QRewindingRC # or QExtremeLearningMachine
        experiment = model()
        model_of_interest, data_of_interest = get_paras(experiment)
    """
    data_of_interest = {
        'xtype': 'tsetpoint', 'ytype': 'treactor', 'memory': 0,
        'xnorm': 'norm', 'ynorm': 'norm', 
        'rseed_data': 9369, 'nepisodes': 15
    }
    if type(experiment) == QRewindingStatevectorRC:
        model_of_interest = {
            'qctype': 'ising', 'nyfuture': 1, 
            'lookback': 1,
            'lookback_max': True,
            'add_y_to_input': True, 'use_true_y_in_val': False,
            'use_partial_meas': False, 
            'reset_instead_meas': False, 
            'reseti': True, 'resetm': True,
            'measaxes': 'zyx',
            'ising_t': 1.0, 'ising_h': 0.1, 'ising_jmax': 1, 'ising_wmax': 10,
            'rseed': 9369, 'ftype': 3, 
            'nqubits': 5, 'restarting': True, 'regression_model': 'ridge',
            'sim_precision': 'double', 'add_x_as_feature': True,
            'sim_sampling': 'multinomial',
            'sim': 'aer_simulator',
            'shots': 8192,
        }
    elif type(experiment) == QExtremeLearningMachine:
        model_of_interest = {
            'qctype': 'ising', 'nyfuture': 1, 'measaxes': 'zyx',
            'ising_t': 1.0, 'ising_h': 0.1, 'ising_jmax': 1, 'ising_wmax': 10,
            'rseed': 9369, 'ftype': 4, 
            'nqubits': 5, 'regression_model': 'ridge', 'use_true_y_in_val': False,
            'sim_precision': 'double', 'add_x_as_feature': True,
            'sim': 'aer_simulator',
            'shots': 8192,
        }
    else:
        raise TypeError()
    return model_of_interest, data_of_interest


rseed_data_list = [9369, 9808, 9809, 9810, 9811, 9812]
rseeds_circ = [2682, 2683, 2684, 2685, 2686]
qctype_list = ['ising', 'ising_nn', 'empty']
