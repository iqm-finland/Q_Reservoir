# -*- coding: utf-8 -*-
"""Contains :class:`QRewindingRC`, :class:`QRewindingStatevectorRC` and :class:`QRewindingRC_Neat` classes.

For simulation ONLY:
:class:`QRewindingStatevectorRC` is the quantum reservoir model performing the best.
In theory :class:`QRewindingRC` should perform the same, but a lot slower.
In practice :class:`QRewindingStatevectorRC` is has been much more tested and is highly recommended in all situations.
For RUNNING on REAL HARDWARE:
:class:`QRewindingRC_Neat` allows this functionality; along with much developed features such as saving and reloading.
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeManila, FakeToronto, FakeJakartaV2, FakeProvider
from qiskit_aer.noise import NoiseModel
import qiskit.quantum_info as qi
import qiskit_aer
import qiskit
import warnings
import shutil
from qiskit import qpy
from qiskit.transpiler import CouplingMap
from qiskit.circuit import Parameter
import json
from numpy.random import Generator, PCG64
from qiskit.quantum_info.operators.symplectic import Pauli
from itertools import product,permutations,combinations
from operator import itemgetter

def citemgetter(pos,string):
    item_executable=itemgetter(*pos)
    return ''.join(item_executable(string))

def replacer(s, newstring, indices, nofail=False):
    # raise an error if index is outside of the string


    for i in range(len(indices)):
        s = s[:indices[i]] + newstring[i] + s[indices[i] + 1:]

    # insert the new string between "slices" of the original
    return s

from qiskit.quantum_info.operators.predicates import is_hermitian_matrix, is_positive_semidefinite_matrix
from iqm.qiskit_iqm.iqm_transpilation import optimize_single_qubit_gates

import numpy as np
import sys
import os
import time

sys.path.insert(1, os.path.abspath('..')) # module: qrc_surrogate

from src.continuous import QContinuousRC
from src.basemodel import PredictionModelBase, QuantumBase, StepwiseModelBase
from src.data import DataSource
from src.helpers import mse, mape, nrmse, moving_average, weighted_gate_count, unnormalize,unstandardize

#have to remove if not available...
from src.mapomatic_functions import *
from src.utils import *
from src.RM_utils import *

from src.executor import batched_executor
from src.executor import Resonance_batched_executor
import pickle as pkl

from src.circuits import \
    random_circuit, random_clifford_circuit, random_ht_circuit, ising_circuit, ising_circuit_natural, ising_circuit_naturalH, \
    jiri_circuit, spread_circuit, hcnot_circuit, \
    circularcnot_circuit, efficientsu2_circuit, downup_circuit, fake_circuit, random_hpcnot_circuit

# np.set_printoptions(precision=2, suppress=True)

step_to_save_qc = 10

class QRewindingRC(StepwiseModelBase, QContinuousRC):
    """
    Takes x(t) (action) and previous ouput y(t-1) as input.
    Repeats past n steps and the current step, then measures all qubits.

    During training the model takes all x(t) as given and uses the correct y(t-1) from the data.
    During inference (validating) uses the previous predicted output ~ y(t-1) and x(t) can depend on the previous prediction.
    
    Args:
        nmeas (int): 
            Number of qubits to be measured at every rewinded (non-final) step.
            A final step all qubits are measured.
        reset_instead_meas (bool):
            Reset qubits instead of measuring at every rewinded (non-final) step.
        resetm (bool):
            Reset qubits after measurement at every rewinded (non-final) step.
        use_partial_meas (bool):
            If partial measurements at every rewinded (non-final) step are to be used for features.
            Set to 'False' if 'reset_instead_meas' is 'True'.
        lookback (int):
            Number of steps to be rewinded at every step.
        restarting (bool):
            If to restart at every step (rewind to the beginning).
        lookback_max (bool):
            If restarting is True, this is ignored (and set to True).
            If True, the number of steps to be rewinded is the maximum possible (all previous steps),
            but never goes further back than step 0.
            If False, steps before t=0 are possibly repeated as well, with input values depending on :code:`set_past_y_to_0`.
        add_y_to_input (bool):
            If to use previous y(t) in addition to x(t) as input.
    
    Additional arguments are passed to :class:`src.basemodel.StepwiseModelBase`
    and :class:`src.continuous.QOnlineReservoir`.
    """
    model_name = 'rewinding_rc'

    def __init__(
        self,
        # QRewindingRC
        use_partial_meas = False,
        reset_instead_meas = True,
        lookback = 3,
        lookback_max = True,
        restarting = False,
        mend = False,
        add_y_to_input = True,
        # -
        # QOnlineReservoir
        washout = 0, # number of first steps of episode ignored by fitting/error metrics
        preloading = 0, # number of times the first step is repeated before episode actually starts
        mtype = 'projection',
        minc = True, 
        nmeas = 1, # number of measured qubits
        reseti = True, # reset before input
        resetm = False, # reset after measurements
        nenccopies = 1,
        # -
        # StepwiseModelBase
        xyoffset = 1,
        set_past_y_to_0 = True,
        use_true_y_in_val = False,
        # -
        # PredictionModelBase
        rseed = 0,
        log = True,
        add_x_as_feature = True,
        # predicting multiple steps forward
        nyfuture = 1, 
        delete_future_y = True,
        # fitter
        fitter = 'sklearn',
        regression_model = 'regression',
        regression_alpha = 0.1,
        regression_l1 = 0.1,
        poly_degree = 3,
        # -
        # QuantumBase
        nqubits = 5,
        qctype = 'ising',
        qinit = 'none',
        nlayers = 1, # unitaries per timestep
        ftype = 0,
        enctype = 'angle',
        encaxes = 1, # number of axis for encoding
        measaxes = 3,
        encangle = 1, # if = 1, encoding is single qubit rotation from 0 to 1*Pi
        shots = 2**13, # 8192
        # ising
        ising_t = 1,
        ising_jmax = 1,
        ising_h = .1,
        ising_wmax = 10,
        ising_random = True,
        ising_jpositive = False,
        ising_wpositive = False,
        # sim
        sim = 'aer_simulator',
        t1 = 50,
        sim_method = 'statevector',
        sim_precision = 'single',
    ) -> None:
        QContinuousRC.__init__(
            self,
            washout = washout,
            preloading = preloading,
            mtype = mtype,
            minc = minc, 
            mend = mend,
            nmeas = nmeas,
            reseti = reseti, 
            resetm = resetm, 
            nenccopies = nenccopies,
            # -
            # PredictionModelBase
            rseed = rseed,
            log = log,
            add_x_as_feature = add_x_as_feature,
            # predicting multiple steps forward
            nyfuture = nyfuture, 
            delete_future_y = delete_future_y,
            # fitter
            fitter = fitter,
            regression_model = regression_model,
            regression_alpha = regression_alpha,
            regression_l1 = regression_l1,
            poly_degree = poly_degree,
            # -
            # QuantumBase
            nqubits = nqubits,
            qctype = qctype,
            qinit = qinit,
            nlayers = nlayers, 
            ftype = ftype,
            enctype = enctype,
            measaxes = measaxes,
            encaxes = encaxes,
            encangle = encangle,
            shots = shots, # 8192
            # ising
            ising_t = ising_t,
            ising_jmax = ising_jmax,
            ising_h = ising_h,
            ising_wmax = ising_wmax,
            ising_random = ising_random,
            ising_jpositive = ising_jpositive,
            ising_wpositive = ising_wpositive,
            # sim
            sim = sim,
            t1 = t1,
            sim_method = sim_method,
            sim_precision = sim_precision,
        )
        StepwiseModelBase.__init__(
            self,
            xyoffset = xyoffset,
            set_past_y_to_0 = set_past_y_to_0,
            use_true_y_in_val = use_true_y_in_val,
            lookback_max = lookback_max
        )
        # partial measurement related
        self.reset_instead_meas = reset_instead_meas
        self.use_partial_meas = use_partial_meas 
        if reset_instead_meas == True:
            if self.resetm == False:
                print('! reset_instead_meas is True, setting resetm to True !')
            self.resetm = True
        if restarting == True:
            self.lookback_max = True
        if restarting == True or self.lookback_max == True:
            assert self.use_partial_meas == False, f"""
                If restarting is True, use_partial_meas must be False.
                Reason: there would be a different number of measurements, 
                and thus number of features, at each step, 
                which cannot be handled by the OLS fitter."""
        # 
        lookback = max(1, lookback)
        self.restarting = restarting 
        if restarting == True:
            lookback = 1
        self.lookback = lookback
        self.ylookback = lookback 
        self.xlookback = lookback
        self.add_y_to_input = add_y_to_input
        if self.add_y_to_input == False:
            self.ylookback = 0
        return

    def _set_unitary_and_meas(self):
        """Define unitary quantum circuit at every timestep.
        Sets self.unistep.
        Excludes input encoding and measurements.
        """
        self.dimx_wo_copies = self.dimx
        if self.ylookback > 0:
            self.dimx_wo_copies += self.dimy
        self.dimxqc = int(self.dimx_wo_copies * self.nenccopies)
        assert self.dimxqc <= self.nqubits, f'{self.dimxqc} {self.nqubits}'
        # dimension of the input data at each step
        if self.ylookback > 0:
            self.dmin = np.asarray([self.xmin, self.ymin]).reshape(-1) # (dimxqc,)
            self.dmax = np.asarray([self.xmax, self.ymax]).reshape(-1) # (dimxqc,)
        else:
            self.dmin = np.asarray(self.xmin).reshape(-1) # (dimxqc,)
            self.dmax = np.asarray(self.xmax).reshape(-1) # (dimxqc,)
        self.qin = [*range(self.dimxqc)]
        self.quni = [*range(self.nqubits)]
        if self.mend:
            self.qmeas = [*range(self.nqubits-self.nmeas, self.nqubits)]
        else:
            self.qmeas = [*range(self.nmeas)]
        self.qreset = []
        if self.resetm == True:
            self.qreset += self.qmeas
        if self.reseti == True:
            self.qreset += self.qin
        self.qreset = list(set(self.qreset))
        # classical bits for measuring
        self.ncbits = self.nqubits
        if self.use_partial_meas:
            self.ncbits += (self.nmeas * self.lookback)
            self.cbits_final_meas = [*range(self.ncbits-self.nqubits, self.ncbits)]
        else:
            self.cbits_final_meas = self.quni
        if self.unistep is None:
            self._set_unitary()
            # number of gates for one timestep without input encoding (reservoir)
            self.unistep.name = 'U'
            qct = transpile(self.unistep, backend=self.backend)
            self.ngates = weighted_gate_count(qct)
        else:
            print('! Unitary is already defined !')
        return
    
    def _get_step_features(self, angles, nsteps=1, saveqc=False):
        """Features for this step, provided the input angles.
        At every step, the circuit consists of lookback+1 steps
        
        angles (np.ndarray): (nsteps, dimxqc)
        nsteps (int): how many previous steps should be repeated (including the current step).
        
        Return:
            features for this step (1, dimf)
        """
        features_step = []
        for nax, ax in enumerate(self.measaxes):
            qc = QuantumCircuit(self.nqubits, self.ncbits)
            if self.qinit == 'h':
                qc.h(self.quni)
            for prevstep in range(nsteps-1):
                # input
                self._add_input_to_qc(qc=qc, angles=angles, step=prevstep)
                # unitary
                qc.append(self.unistep, self.quni)
                if self.nmeas > 0:
                    if self.reset_instead_meas:
                        qc.reset(self.qmeas)
                    elif self.use_partial_meas:
                        match ax:
                            # https://arxiv.org/abs/1804.03719
                            case 'z':
                                pass
                            case 'x':
                                qc.h(self.qmeas)
                            case 'y':
                                qc.sdg(self.qmeas)
                                qc.h(self.qmeas)
                            case _:
                                raise Warning(f'Invalid measaxes {self.measaxes}')
                        qc.measure(
                            qubit=self.qmeas, 
                            cbit=[*range(prevstep*self.nmeas, (prevstep+1)*self.nmeas)]
                        )
                    else:
                        # the cbits will be overwritten at every step, only the last one will be kept
                        qc.measure(qubit=self.qmeas, cbit=[*range(self.nmeas)])
            # final step
            # input
            self._add_input_to_qc(qc=qc, angles=angles, step=nsteps-1)
            # unitary
            qc.append(self.unistep, self.quni)
            # measure
            match ax:
                # https://arxiv.org/abs/1804.03719
                case 'z':
                    pass
                case 'x':
                    qc.h(self.quni)
                case 'y':
                    qc.sdg(self.quni)
                    qc.h(self.quni)
                case _:
                    raise Warning(f'Invalid measaxes {self.measaxes}')
            qc.measure(qubit=self.quni, cbit=self.cbits_final_meas) 
            if saveqc:
                self.qc = qc
            compiled_qc = transpile(qc, self.backend)
            job = self.backend.run(compiled_qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            # qiskit counts are in reverse order
            counts = {k[::-1]: v for k, v in counts.items()}
            # turn measurements into features
            if self.use_partial_meas:
                for prev_step in range(nsteps-1):
                    counts_step = {}
                    b_step = b[prev_step * self.nmeas:(prev_step + 1) * self.nmeas]
                    for b, c in counts.items():
                        if b_step in counts_step.keys():
                            counts_step[b_step] += c
                        else:
                            counts_step[b_step] = c
                    features_step = self._step_meas_to_step_features(counts_step, features_step)
                # final step
                counts_final = {}
                b_final = b[len(self.quni):]
                for b, c in counts.items():
                    if b_step in counts_final.keys():
                        counts_final[b_final] += c
                    else:
                        counts_final[b_final] = c
                features_step = self._step_meas_to_step_features(counts_final, features_step)
            else:
                features_step = self._step_meas_to_step_features(counts_step=counts, features_step=features_step)
        return features_step

    def _get_input_t(self, xe_lookback, ye_lookback, t_in, t_pred):
        """
        Every step is a mini-episode:
        At every step, repeat past n steps + current step. 
        At the current step, feed in current action x(t) and previous ouput y(t-1).
        At the past n'th step, feed in action x(t-n) and previous output y(t-n-1).
        In training previous outpus y are the true values from the data.
        
        Returns:
            angles (np.ndarray): (lookback, dimxqc)
        """
        if self.restarting:
            return np.hstack([
                xe_lookback[0:t_in+1],
                ye_lookback[0:t_in+1],
            ])
        else:
            if self.lookback_max:
                if self.ylookback > 0:
                    # t_pred = 4, t_in = 3, lookback = 3 -> 1, 2, 3
                    return np.hstack([
                        xe_lookback[max(0, t_in-self.ylookback+1):t_in+1],
                        ye_lookback[max(0, t_in-self.ylookback+1):t_in+1], 
                    ])
                else:
                    return xe_lookback[max(0, t_in-self.xlookback+1):t_in+1]
            else:
                if self.ylookback > 0:
                    return np.hstack([
                        xe_lookback[t_in:t_in+self.xlookback],
                        ye_lookback[t_in:t_in+self.ylookback], 
                    ])
                else:
                    return xe_lookback[t_in:t_in+self.xlookback]
    
    def _t_input_to_t_features(self, input_t, x0, t_pred):
        """Features of a step, given the input.
        
        Calls angle_encoding and get_step_features.
        """
        # encode input for quantum circuit
        step_input_angles = self._angle_encoding(episode=input_t, dmin=self.dmin, dmax=self.dmax)
        # run circuit, get features
        if self.lookback_max:
            lookback = min(self.xlookback, t_pred)
        else:
            lookback = self.xlookback
        # print(f'lookback {lookback} t_pred {t_pred}')
        features_qc = self._get_step_features(
            angles=step_input_angles, 
            nsteps=t_pred if self.restarting == True else lookback,
            saveqc=True if t_pred == step_to_save_qc else False,
        )
        if self.add_x_as_feature:
            return np.hstack(features_qc + [x0]) 
        else:
            return np.hstack(features_qc)
        


class QRewindingStatevectorRC(QRewindingRC):
    """
    Statevector.

    Intended to be faster when using mid circuit measurements or resets.

    If set to restarting, will evolve one statevector through all steps.
    If set to rewinding, will create new statevector and evolve through past steps.

    When doing noisy simulations we need to move from Statevector to DensityMatrix.
    Considerably slower. 

    Args:
        sim_sampling (str): 
            | method to sample counts from statevector, which are then used to estimate expectation values.
            | qiskit: use qiskit's :code:`qiskit.quantum_info.Statevector.sample_counts()`. crazy slow.
            | multinomial: :code:`np.multinomial`. 10x faster and same result as qiskit.
                Speedup is less when compared to :code:`qiskit.quantum_info.DensityMatrix.sample_counts()`.
            | naive: uses exact probabilies to generate counts. expectation values are still estimated from counts.
            | exact: calculates expectation values directly from statevector.

    Additional arguments are passed to :class:`.QRewindingRC`.
    """
    model_name = 'rewinding_statevector_rc'

    def __init__(
        self,
        # QRewindingStatevectorRC
        sim_sampling = 'multinomial',
        # QRewindingRC
        use_partial_meas = False,
        reset_instead_meas = False,
        lookback = 3,
        add_y_to_input = True,
        restarting = False,
        mend = False,
        set_past_y_to_0 = True,
        use_true_y_in_val = False,
        # -
        # QOnlineReservoir
        washout = 0, # number of first steps of episode ignored by fitting/error metrics
        preloading = 0, # number of times the first step is repeated before episode actually starts
        mtype = 'projection',
        minc = True, 
        nmeas = 1, # number of measured qubits
        reseti = True, # reset before input
        resetm = True, # reset after measurements
        # -
        lookback_max = True,
        # PredictionModelBase
        rseed = 0,
        log = True,
        add_x_as_feature = True,
        # predicting multiple steps forward
        nyfuture = 1, 
        delete_future_y = True,
        # fitter
        fitter = 'sklearn',
        regression_model = 'regression',
        regression_alpha = 0.1,
        regression_l1 = 0.1,
        poly_degree = 3,
        # -
        # QuantumBase
        nqubits = 5,
        qctype = 'ising',
        qinit = 'none',
        nlayers = 1, # unitaries per timestep
        ftype = 0,
        enctype = 'angle',
        encaxes = 1,
        nenccopies = 1,
        encangle = 1, # if = 1, encoding is single qubit rotation from 0 to 1*Pi
        measaxes = 3, # number of axis for measurements
        shots = 2**13, # 8192
        # ising
        ising_t = 1,
        ising_jmax = 1,
        ising_h = .1,
        ising_wmax = 10,
        ising_random = True,
        ising_jpositive = False,
        ising_wpositive = False,
        # sim
        sim = 'aer_simulator',
        t1 = 50,
        sim_method = 'statevector',
        sim_precision = 'double',
    ) -> None:
        QRewindingRC.__init__(
            self,
            # QRewindingRC
            use_partial_meas = use_partial_meas,
            reset_instead_meas = reset_instead_meas,
            lookback = lookback,
            lookback_max = lookback_max,
            restarting = restarting,
            mend = mend,
            set_past_y_to_0 = set_past_y_to_0,
            add_y_to_input = add_y_to_input,
            use_true_y_in_val = use_true_y_in_val,
            # -
            # QOnlineReservoir
            washout = washout,
            preloading = preloading,
            mtype = mtype,
            minc = minc, 
            nmeas = nmeas,
            reseti = reseti, 
            resetm = resetm, 
            # -
            # PredictionModelBase
            rseed = rseed,
            log = log,
            add_x_as_feature = add_x_as_feature,
            # predicting multiple steps forward
            nyfuture = nyfuture, 
            delete_future_y = delete_future_y,
            # fitter
            fitter = fitter,
            regression_model = regression_model,
            regression_alpha = regression_alpha,
            regression_l1 = regression_l1,
            poly_degree = poly_degree,
            # -
            # QuantumBase
            nqubits = nqubits,
            qctype = qctype,
            qinit = qinit,
            nlayers = nlayers, 
            ftype = ftype,
            enctype = enctype,
            encaxes = encaxes,
            nenccopies = nenccopies,
            encangle = encangle,
            measaxes = measaxes, 
            shots = shots, # 8192
            # ising
            ising_t = ising_t,
            ising_jmax = ising_jmax,
            ising_h = ising_h,
            ising_wmax = ising_wmax,
            ising_random = ising_random,
            ising_jpositive = ising_jpositive,
            ising_wpositive = ising_wpositive,
            # sim
            sim = sim,
            t1 = t1,
            sim_method = sim_method,
            sim_precision = sim_precision,
        )
        # - Set simulator
        self.sim_sampling = sim_sampling
        if self.sim_sampling == 'naive':
            self.shots = 2**13
        if self.sim_method == 'density_matrix':
            self.dm = True
            if self.sim_precision == 'single':
                print("! Setting sim_precision='single' with sim_method='density_matrix' can lead to dm.trace!=1.")
        elif self.sim_method == 'statevector':
            self.dm = False
        # Set statevector
        # used for restarting
        self._init_statevector()
        # - 
        if resetm == False:
            raise NotImplementedError('Only resetm=True (partial tracing the statevector after getting the counts) has been implemented')
        self.noisy_dm = False
        if self.backend.options.noise_model != None:
            self.noisy_dm = True
            if self.sim_method != 'density_matrix':
                raise NotImplementedError(
                    f"""Noise models for Statevector hasnt been implemented: 
                    {self.backend.options.noise_model}, {self.sim_method}.
                    Set sim_method='density_matrix'."""
                )
    
    def _init_statevector(self):
        self.statev = qi.Statevector.from_label('0'*self.nqubits) # difference SV
        if self.qinit == 'h':
            self.statev = qi.Statevector.from_label('+'*self.nqubits)
        if self.dm:
            self.statev = qi.DensityMatrix(self.statev.data)
        return
    
    def _get_input_t(self, xe_lookback, ye_lookback, t_in, t_pred=None):
        """
        Every step is a mini-episode:
        At every step, repeat past n steps + current step. 
        At the current step, feed in current action x(t) and previous ouput y(t-1).
        At the past n'th step, feed in action x(t-n) and previous output y(t-n-1).
        In training previous outpus y are the true values from the data.
        
        Returns:
            angles (np.ndarray): (lookback, dimxqc)
        """
        if t_in == 0:
            self._init_statevector()
        # if restarting is True, will return one step at a time
        if self.lookback_max:
            if self.ylookback > 0:
                # t_pred = 4, t_in = 3, lookback = 3 -> 1, 2, 3
                return np.hstack([
                    xe_lookback[max(0, t_in-self.ylookback+1):t_in+1],
                    ye_lookback[max(0, t_in-self.ylookback+1):t_in+1], 
                ])
            else:
                return xe_lookback[max(0, t_in-self.xlookback+1):t_in+1]
        else:
            if self.ylookback > 0:
                return np.hstack([
                    xe_lookback[t_in:t_in+self.xlookback],
                    ye_lookback[t_in:t_in+self.ylookback], 
                ])
            else:
                return xe_lookback[t_in:t_in+self.xlookback]
    
    def _t_input_to_t_features(self, input_t, x0, t_pred):
        """Features of a step, given the input.
        
        Calls angle_encoding and get_step_features.
        """
        # encode input for quantum circuit
        step_input_angles = self._angle_encoding(episode=input_t, dmin=self.dmin, dmax=self.dmax)
        # run circuit, get features
        if self.lookback_max: 
            lookback = min(self.xlookback, t_pred)
        else:
            lookback = self.xlookback
        features_qc = self._get_step_features(
            angles=step_input_angles, 
            nsteps=lookback, 
            saveqc=True if t_pred == step_to_save_qc else False,
        )
        if self.add_x_as_feature:
            return np.hstack(features_qc + [x0]) 
        else:
            return np.hstack(features_qc)
    
    def _get_step_features(self, angles, nsteps=1, saveqc=False):
        """Features for this step. 
        At every step, the circuit consists of lookback+1 steps.
        statev: qiskit.quantum_info.Statevector() or qiskit.quantum_info.DensityMatrix()
        
        angles (np.ndarray): (nsteps, dimxqc)
        nsteps (int): how many previous steps should be repeated (including the current step).
            
        
        Returns:
            features for this step (1, dimf)
        """
        features_step = []
        if self.restarting == False: 
            self._init_statevector()
        assert np.shape(angles)[0] == nsteps, f'{np.shape(angles)}[0] == {nsteps}'
        for prevstep in range(nsteps-1):
            if self.restarting == True: 
                raise ValueError('Shouldnt be here')
            qc_prevsteps = QuantumCircuit(self.nqubits)
            if self.noisy_dm == True: # set get dm
                # self.statev = _allow_invalid_dm(self.statev)
                qc_prevsteps.set_density_matrix(self.statev)
            # input
            self._add_input_to_qc(qc=qc_prevsteps, angles=angles, step=prevstep)
            # unitary
            qc_prevsteps.append(self.unistep, self.quni)
            # run qc
            if self.noisy_dm == True: 
                # get dm
                qc_prevsteps.save_density_matrix(qubits=None, label="dm", conditional=False)
                qc_prevsteps = transpile(qc_prevsteps, self.backend)
                job = self.backend.run(qc_prevsteps)
                result = job.result()
                self.statev = result.data()['dm']
            else:
                # evolve
                self.statev = self.statev.evolve(qc_prevsteps)
            if (self.nmeas > 0) and (self.use_partial_meas == True):
                # get measurements for all axis
                # instead Statevector.expectation_value(oper=, qargs=)
                for nax, ax in enumerate(self.measaxes): 
                    statev_ax = self._get_rotated_sv(ax=ax, nqubits=qc_prevsteps.num_qubits, qargs=self.qmeas)
                    if self.sim_sampling == 'exact':
                        self._step_meas_to_step_features_sv(sv=statev_ax, features_step=features_step)
                    else:
                        counts_step = self._counts_from_sv(sv=statev_ax)
                        # counts_step = statev_ax.sample_counts(qargs=self.qmeas, shots=self.shots)
                        features_step = self._step_meas_to_step_features(counts_step, features_step)
            if len(self.qreset) > 0:
                self._reset_sv()
        # del qc_prevsteps
        # final step
        qc_final = QuantumCircuit(self.nqubits)
        if self.noisy_dm == True: # set get dm
            # self.statev = _allow_invalid_dm(self.statev)
            qc_final.set_density_matrix(self.statev)
        # input
        # if nsteps-1 >= 0
        self._add_input_to_qc(qc=qc_final, angles=angles, step=nsteps-1)
        # unitary
        qc_final.append(self.unistep, self.quni)
        # run qc
        if self.noisy_dm == True:
            qc_final.save_density_matrix(qubits=None, label="dm", conditional=False)
            qc_final = transpile(qc_final, self.backend)
            job = self.backend.run(qc_final)
            result = job.result()
            self.statev = result.data()['dm']
        else:
            self.statev = self.statev.evolve(qc_final)
        if saveqc:
            self.qc = qc_final
        # measure
        for nax, ax in enumerate(self.measaxes):
            statev_ax = self._get_rotated_sv(ax=ax, nqubits=qc_final.num_qubits, qargs=self.quni)
            if self.sim_sampling == 'exact':
                self._step_meas_to_step_features_sv(sv=statev_ax, features_step=features_step)
            else:
                # get counts
                counts_final = self._counts_from_sv(sv=statev_ax)
                # turn measurements into features
                features_step = self._step_meas_to_step_features(counts_step=counts_final, features_step=features_step)
        if self.restarting == True:  
            if len(self.qreset) > 0:
                self._reset_sv() 
        return features_step
    
    def _get_rotated_sv(self, ax, nqubits, qargs) -> qiskit.quantum_info.Statevector:
        qc_ax = QuantumCircuit(nqubits)
        if self.noisy_dm == True:
            # self.statev = _allow_invalid_dm(self.statev)
            qc_ax.set_density_matrix(self.statev)
        match ax:
            # https://arxiv.org/abs/1804.03719
            case 'z':
                pass
            case 'x':
                qc_ax.h(qargs)
            case 'y':
                qc_ax.sdg(qargs)
                qc_ax.h(qargs)
            case _:
                raise Warning(f'Invalid measaxes {self.measaxes}')
        if self.noisy_dm == True:
            qc_ax.save_density_matrix(qubits=None, label="dm", conditional=False)
            qc_ax = transpile(qc_ax, self.backend)
            job = self.backend.run(qc_ax)
            result = job.result()
            statev_ax = result.data()['dm']
        else:
            statev_ax = self.statev.evolve(qc_ax)
        return statev_ax
    
    def _reset_sv(self) -> qiskit.quantum_info.Statevector:
        # reset measured qubits
        if self.resetm or self.reset_instead_meas:
            statev_meas = qi.partial_trace(self.statev, qargs=self.qreset)
            statev_reset = qi.Statevector.from_label('0'*len(self.qreset))
            if self.dm:
                statev_reset = qi.DensityMatrix(statev_reset.data)
            if self.mend:
                self.statev = statev_reset.tensor(statev_meas)
            else:
                self.statev = statev_meas.tensor(statev_reset)
        else:
            # project statevector instead of resetting
            # self.statev = self.statev.measure(shots=1)
            raise NotImplementedError()
        return

    def _counts_from_sv(self, sv):
        """Naive alternative to Qiskit's statevector.get_counts() using probabilities.
        
        Qiskit becomes very slow for growing number of qubits. 
        I think Qiskit uses a metropolis hasting (Monte Carlo) algorithm under the hood.

        Args:
            sv (qiskit.quantum_info.statevector)
        
        Returns:
            counts (dict)
        """
        if self.sim_sampling == 'qiskit':
            return sv.sample_counts(qargs=self.quni, shots=self.shots)
        elif self.sim_sampling == 'inverse':
            raise NotImplementedError(f"sim_sampling '{self.sim_sampling}' is not implemented in 'counts_from_sv'.")
        elif self.sim_sampling == 'multinomial':
            probs = sv.probabilities() # (2^N,)
            counts_mn = np.random.multinomial(n=self.shots, pvals=probs) 
            return {format(i, "b").zfill(sv.num_qubits): c for i, c in enumerate(counts_mn)} 
            # return {format(i, "b").zfill(sv.num_qubits)[::-1]: c for i, c in enumerate(counts_mn)} 
        elif self.sim_sampling == 'naive':
            return sv.probabilities_dict()
            # return {key: p*self.shots for key, p in probs.items()}
        else:
            raise NotImplementedError(f"sim_sampling '{self.sim_sampling}' is not implemented in 'counts_from_sv'.")


def _allow_invalid_dm(sv):
    # https://github.com/Qiskit/qiskit-terra/blob/26886a1b2926a474ea06aa3f9ce9e11e6ce28020/qiskit/quantum_info/states/densitymatrix.py#L188
    # https://github.com/Qiskit/qiskit-terra/blob/26886a1b2926a474ea06aa3f9ce9e11e6ce28020/qiskit/quantum_info/operators/mixins/tolerances.py#L22
    # default: atol 1e-8, rtol 1e-5
    # sv._RTOL_DEFAULT = 1e-3 # doesnt change anything
    # sv._ATOL_DEFAULT = 1e-5
    sv.__class__._MAX_TOL = 1e-3
    sv.__class__.rtol = 1e-3
    sv.__class__.atol = 1e-5
    # assert isinstance(sv, qi.DensityMatrix), f'{type(sv)}'
    # assert sv.is_valid(), f'trace {sv.trace()}, hermitian {is_hermitian_matrix(sv.data)}, positive semidefinite {is_positive_semidefinite_matrix(sv.data)}'
    return sv

class QRewindingRC_Neat(StepwiseModelBase, QContinuousRC):
    """
    An improved version of class designed to be run on the quantum computer, for ease of understanding.

    Takes x(t) (action) and previous ouput y(t-1) as input.
    Repeats past n steps and the current step, then measures all qubits.

    During training the model takes all x(t) as given and uses the correct y(t-1) from the data.
    During inference (validating) uses the previous predicted output ~ y(t-1) and x(t) can depend on the previous prediction.
    
    Args:
        nmeas (int): 
            Number of qubits to be measured at every rewinded (non-final) step.
            A final step all qubits are measured.
        reset_instead_meas (bool):
            Reset qubits instead of measuring at every rewinded (non-final) step.
        resetm (bool):
            Reset qubits after measurement at every rewinded (non-final) step.
        use_partial_meas (bool):
            If partial measurements at every rewinded (non-final) step are to be used for features.
            Set to 'False' if 'reset_instead_meas' is 'True'.
        lookback (int):
            Number of steps to be rewinded at every step.
        restarting (bool):
            If to restart at every step (rewind to the beginning).
        lookback_max (bool):
            If restarting is True, this is ignored (and set to True).
            If True, the number of steps to be rewinded is the maximum possible (all previous steps),
            but never goes further back than step 0.
            If False, steps before t=0 are possibly repeated as well, with input values depending on :code:`set_past_y_to_0`.
        add_y_to_input (bool):
            If to use previous y(t) in addition to x(t) as input.
        file_name' (str): Name of the resulting parquet file saved in "experiments". Defaults to model_name.
    
    Additional arguments are passed to :class:`src.basemodel.StepwiseModelBase`
    and :class:`src.continuous.QOnlineReservoir`.
    """
    model_name = 'neat_QPU'

    def __init__(
        self,
        # QRewindingRC_Neat
        simulate=False, # Whether to use a simulator or a real backend.
        manual=True, # If true; use a custom chosen layout/circuit.
        
        #Batch Executor parameters
        max_batch=4500,
        max_circuit=500,
        max_concurrent=100,
        strip_metadata = False,
        initial_layout = None,
        max_tries = 10,
        calibration_set_id = None,
        verbose = True,
        # Information used in transpiling: Outdated.
        qubits=None,
        reduced_coupling_map = None,
        total_req=None,

        #Reservoir parameters.
        use_partial_meas = False,
        reset_instead_meas = True,
        lookback = 3,
        lookback_max = True,
        restarting = False,
        mend = False,
        add_y_to_input = True,
        # -
        # QOnlineReservoir
        washout = 0, # number of first steps of episode ignored by fitting/error metrics
        preloading = 0, # number of times the first step is repeated before episode actually starts
        mtype = 'projection',
        minc = True, 
        nmeas = 1, # number of measured qubits
        reseti = True, # reset before input
        resetm = False, # reset after measurements
        nenccopies = 1,
        # -
        # StepwiseModelBase
        xyoffset = 1,
        set_past_y_to_0 = True,
        use_true_y_in_val = False,
        # -
        # PredictionModelBase
        rseed = 0,
        log = True,
        add_x_as_feature = True,
        # predicting multiple steps forward
        nyfuture = 1, 
        delete_future_y = True,
        # fitter
        fitter = 'sklearn',
        regression_model = 'regression',
        regression_alpha = 0.1,
        regression_l1 = 0.1,
        poly_degree = 3,
        # -
        # QuantumBase
        nqubits = 5,
        qctype = 'ising',
        qinit = 'none',
        nlayers = 1, # unitaries per timestep
        ftype = 0,
        enctype = 'angle',
        encaxes = 1, # number of axis for encoding
        measaxes = 3,
        encangle = 1, # if = 1, encoding is single qubit rotation from 0 to 1*Pi
        shots = 2**13, # 8192
        # ising
        ising_t = 1,
        ising_jmax = 1,
        ising_h = .1,
        ising_wmax = 10,
        ising_random = True,
        ising_jpositive = False,
        ising_wpositive = False,
        # sim (not in practise how simulator is defined.?)
        sim = 'aer_simulator',
        t1 = 50,
        sim_method = 'statevector',
        sim_precision = 'single',
        
        #Code for Mapomatic (if installed).
        QPU='Apollo',
        opt_level=3,

        #added to change file names:
        file_name = None,     
        # use to overwrite manual request:
        confirm = None
    ) -> None:
        # Here is simulate = True, we will run the circuits on a backend simulator.
        self.simulate=simulate
        self.manual=manual
        self.confirm = confirm
        #
        QContinuousRC.__init__(
            self,
            washout = washout,
            preloading = preloading,
            mtype = mtype,
            minc = minc, 
            mend = mend,
            nmeas = nmeas,
            reseti = reseti, 
            resetm = resetm, 
            nenccopies = nenccopies,
            # -
            # PredictionModelBase
            rseed = rseed,
            log = log,
            add_x_as_feature = add_x_as_feature,
            # predicting multiple steps forward
            nyfuture = nyfuture, 
            delete_future_y = delete_future_y,
            # fitter
            fitter = fitter,
            regression_model = regression_model,
            regression_alpha = regression_alpha,
            regression_l1 = regression_l1,
            poly_degree = poly_degree,
            # -
            # QuantumBase
            nqubits = nqubits,
            qctype = qctype,
            qinit = qinit,
            nlayers = nlayers, 
            ftype = ftype,
            enctype = enctype,
            measaxes = measaxes,
            encaxes = encaxes,
            encangle = encangle,
            shots = shots, # 8192
            # ising
            ising_t = ising_t,
            ising_jmax = ising_jmax,
            ising_h = ising_h,
            ising_wmax = ising_wmax,
            ising_random = ising_random,
            ising_jpositive = ising_jpositive,
            ising_wpositive = ising_wpositive,
            # sim
            sim = sim,
            t1 = t1,
            sim_method = sim_method,
            sim_precision = sim_precision,
        )
        StepwiseModelBase.__init__(
            self,
            xyoffset = xyoffset,
            set_past_y_to_0 = set_past_y_to_0,
            use_true_y_in_val = use_true_y_in_val,
            lookback_max = lookback_max
        )
        # partial measurement related
        self.reset_instead_meas = reset_instead_meas
        self.use_partial_meas = use_partial_meas 
        if reset_instead_meas == True:
            if self.resetm == False:
                print('! reset_instead_meas is True, setting resetm to True !')
            self.resetm = True
        if restarting == True:
            self.lookback_max = True
        if restarting == True or self.lookback_max == True:
            assert self.use_partial_meas == False, f"""
                If restarting is True, use_partial_meas must be False.
                Reason: there would be a different number of measurements, 
                and thus number of features, at each step, 
                which cannot be handled by the OLS fitter."""
        # 
        lookback = max(1, lookback)
        self.restarting = restarting 
        if restarting == True:
            lookback = 1
        self.lookback = lookback
        self.ylookback = lookback 
        self.xlookback = lookback
        self.add_y_to_input = add_y_to_input
        if self.add_y_to_input == False:
            self.ylookback = 0

        #Set up Garnet batch submitter.
        self.max_batch=max_batch
        self.max_circuit = max_circuit
        self.max_concurrent = max_concurrent
        self.strip_metadata = strip_metadata
        self.initial_layout = initial_layout
        self.max_tries = max_tries
        self.verbose = verbose 
        self.calibration_set_id = calibration_set_id
        self.executor=Resonance_batched_executor(backend=self.backend,max_batch=self.max_batch,max_circuit=self.max_circuit,strip_metadata=self.strip_metadata,initial_layout=self.initial_layout,max_tries=self.max_tries,max_concurrent=self.max_concurrent)
        
        #For Aniket's code
        self.opt_level=opt_level
        self.QPU=QPU

        if self.restarting == True:
            raise ValueError('Failed to open database')('Cannot Calculate size of circuit automatically in advance: please enter qubit names!') 
        else: nsteps=lookback
        
        #self.dimxqc=int(self.dimx * self.nenccopies)
        #self.memory_size=self.nqubits-self.dimxqc
        #self.total_req=self.memory_size+nsteps*self.dimxqc
        #data_obj.dimx <- doesn't work because of this reason!

        self.total_req=total_req
        if qubits == None:
            if self.total_req == None:
                self.qubits = list(self.backend._qb_to_idx.keys())
            else:
                #print('You have specified ',self.total_req, 'qubits required. Correct?')
                self.qubits = list(self.backend._qb_to_idx.keys())[:self.total_req]
            
        else:
            self.qubits = qubits
        



        if initial_layout==None:
            self.initial_layout = [self.backend.qubit_name_to_index(name) for name in self.qubits]
        else:
            self.initial_layout = initial_layout

        if reduced_coupling_map == None:
            self.backend.coupling_map.make_symmetric()
            self.reduced_coupling_map = self.backend.coupling_map.reduce(self.initial_layout)
        else:
            self.reduced_coupling_map=reduced_coupling_map


        self.file_name=file_name
        if file_name != None:
            self.model_name = file_name
        return
    
    def produce_manual_circuit(self,thetac,nsteps):
        self.all_meas=[]
        Manual_Layout_List=[[15,14,9,18,19],
                            [15,14,9,8,13],
                            [15,14,9,11,16],
                            [15,14,9,5,10]]

        assert self.backend.num_qubits==20, 'Must use Garnet as Backend'
        qc = QuantumCircuit(self.backend.num_qubits, self.backend.num_qubits)
        
        #And the current circuit.
        c_circuit = QuantumCircuit(self.nqubits,self.dimxqc)

        self.angle_names=[]
        self.all_meas=[]


            
        
        #Manually create a swap:
        cSWAP=QuantumCircuit(2,0)
        cSWAP.swap(0,1)
        gSWAP=transpile(cSWAP, basis_gates=['r', 'cz'], optimization_level = 0)
        self.SWAP=gSWAP
            

        for step in range(nsteps): # Loop over all steps used..:
            c_qubits=Manual_Layout_List[step]
            # Add undefined inputs:
            qn = 0 # qubit counter for loop
            for c in range(self.nenccopies):
                for d in range(int(self.dimxqc / self.nenccopies)): # dimx
                    phi_c = Parameter('phi_'+str(step)+'_'+str(qn))
                    #print(step,qn,c_qubits[self.memory_size+0*self.dimxqc+self.qin[qn]])
                    if self.enctype == 'angle':
                        match self.encaxes[c % len(self.encaxes)]:
                            case 'x':
                                qc.r(theta=phi_c,phi=0, qubit=c_qubits[self.memory_size+self.qin[qn]])
                            case 'y':
                                qc.r(theta=phi_c,phi=np.pi/2, qubit=c_qubits[self.memory_size+self.qin[qn]])
                            case 'z':
                                qc.r(theta=np.pi/2,phi=0, qubit=c_qubits[self.memory_size+self.qin[qn]])
                                qc.r(theta=-phi_c,phi=np.pi/2, qubit=c_qubits[self.memory_size+self.qin[qn]])
                                qc.r(theta=-np.pi/2,phi=0, qubit=c_qubits[self.memory_size+self.qin[qn]])
                            case _:
                                raise Warning(f'Invalid encaxes={self.encaxes}')
                    elif self.enctype == 'ryrz': 
                        # self.encaxes = ['ry', 'rz']
                        #qc.h(qubit=self.memory_size+0*self.dimxqc+self.qin[qn])
                        #qc.ry(np.arctan(phi_c) + np.pi/4, qubit=self.memory_size+step*self.dimxqc+self.qin[qn])
                        #qc.rz(np.arctan(phi_c**2) + np.pi/4, qubit=self.memory_size+step*self.dimxqc+self.qin[qn])

                        #h
                        qc.r(theta=np.pi/2,phi=np.pi/2, qubit=self.memory_size+self.qin[qn])
                        qc.r(theta=np.pi,phi=0, qubit=self.memory_size+self.qin[qn])
                        #y
                        qc.r(theta=np.arctan(phi_c) + np.pi/4,phi=np.pi/2, qubit=self.memory_size+0*self.dimxqc+self.qin[qn])
                        #z
                        qc.r(theta=np.pi/2,phi=0, qubit=self.memory_size+0*self.dimxqc+self.qin[qn])
                        qc.r(theta=np.arctan(phi_c**2) + np.pi/4,phi=np.pi/2, qubit=self.memory_size+0*self.dimxqc+self.qin[qn])
                        qc.r(theta=-np.pi/2,phi=0, qubit=self.memory_size+0*self.dimxqc+self.qin[qn])

                    else:
                        raise Warning(f'Invalid enctype {self.enctype}')
                    qn += 1
                
                    

                
            for i in range(len(c_qubits)):
                #H
                qc.r(theta=np.pi/2,phi=np.pi/2, qubit=c_qubits[i])
                qc.r(theta=np.pi,phi=0, qubit=c_qubits[i])



            # CZ
            qc.cz(c_qubits[0],c_qubits[1])
                
            if step==0:
            #SWAP CZ
                qc=qc.compose(gSWAP,qubits=[c_qubits[1],c_qubits[2]])
                qc.cz(c_qubits[1],c_qubits[3])
                qc=qc.compose(gSWAP,qubits=[c_qubits[1],c_qubits[2]])
            elif step==1:
                qc.cz(c_qubits[2],c_qubits[3])
            elif step==2:
                qc=qc.compose(gSWAP,qubits=[c_qubits[2],10]) # use auxillary.
                qc.cz(10,c_qubits[3]) 
                qc=qc.compose(gSWAP,qubits=[c_qubits[2],10])
                
            elif step==3:
                qc=qc.compose(gSWAP,qubits=[c_qubits[2],c_qubits[4]]) # use auxillary.
                qc.cz(c_qubits[4],c_qubits[3]) 
                qc=qc.compose(gSWAP,qubits=[c_qubits[2],c_qubits[4]])

            else:
                print('Error! Too many steps')

            qc.cz(c_qubits[1],c_qubits[2])
            qc.cz(c_qubits[3],c_qubits[4])

            #Final CZ
            if step==0:
            #SWAP CZ
                qc.cz(c_qubits[0],c_qubits[4])
            elif step==1:
                qc=qc.compose(gSWAP,qubits=[c_qubits[1],c_qubits[4]]) # use auxillary.
                qc.cz(c_qubits[0],c_qubits[1])
                qc=qc.compose(gSWAP,qubits=[c_qubits[1],c_qubits[4]])
            elif step==2:
                qc.cz(c_qubits[0],c_qubits[4])
                
            elif step==3:
                qc.cz(c_qubits[0],c_qubits[4])
            else:
                print('Error! Too many steps')

            #XROT
            for i in range(len(c_qubits)):
                #X
                qc.r(theta=thetac[i],phi=0, qubit=c_qubits[i])
                #H
                qc.r(theta=np.pi/2,phi=np.pi/2, qubit=c_qubits[i])
                qc.r(theta=np.pi,phi=0, qubit=c_qubits[i])

                

                



        
            #We will keep track of which measurements to perform here:
            if self.nmeas > 0:
                adj_meas=[k + self.memory_size for k in self.qmeas]
                
                self.all_meas  += [[Manual_Layout_List[step][k] for k in adj_meas]]

        #In the last step, we measure ALL qubits.
        self.all_meas[-1]=Manual_Layout_List[-1]
        return qc
    
    def produce_automatic_circuit(self,thetac,nsteps):
        self.all_meas=[] # we need to keep this for set-up of the final circuits.

        # We produce a custom version of the "manual constuction".
        Manual_Layout_List=[]
        nmem = self.memory_size
        for step in range(self.lookback):
            memory_qubits=list(range(nmem))
            data_qubits=list(range(nmem+step*self.dimxqc,nmem+(step+1)*self.dimxqc))
            Manual_Layout_List+= [memory_qubits + data_qubits]
        print(Manual_Layout_List)
        if max(Manual_Layout_List[-1])>self.backend.num_qubits:
            print('warning: circuit too large for backend!')
            qc = QuantumCircuit(Manual_Layout_List[-1],Manual_Layout_List[-1])
        else:
        #Overall circuit!
            qc = QuantumCircuit(self.backend.num_qubits, self.backend.num_qubits)
        
        #And the current circuit.
        c_circuit = QuantumCircuit(self.nqubits,self.dimxqc)

        self.angle_names=[]
        self.all_meas=[]


            
        
        #Manually create a swap:
        cSWAP=QuantumCircuit(2,0)
        cSWAP.swap(0,1)
        #gSWAP=transpile(cSWAP, basis_gates=['r', 'cz'], optimization_level = 0)
        
        #For the automatic circuits, we use the qiskit gates and allow transpilation.
        self.SWAP=cSWAP
            

        for step in range(nsteps): # Loop over all steps used..:
            c_qubits=Manual_Layout_List[step]
            # Add undefined inputs:
            qn = 0 # qubit counter for loop
            for c in range(self.nenccopies):
                for d in range(int(self.dimxqc / self.nenccopies)): # dimx
                    phi_c = Parameter('phi_'+str(step)+'_'+str(qn))
                    #print(step,qn,c_qubits[self.memory_size+0*self.dimxqc+self.qin[qn]])
                    if self.enctype == 'angle':
                        match self.encaxes[c % len(self.encaxes)]:
                            case 'x':
                                qc.rx(theta=phi_c,qubit=c_qubits[self.memory_size+self.qin[qn]])
                            case 'y':
                                qc.ry(theta=phi_c, qubit=c_qubits[self.memory_size+self.qin[qn]])
                            case 'z':
                                qc.rz(phi=phi_c, qubit=c_qubits[self.memory_size+self.qin[qn]])
                            case _:
                                raise Warning(f'Invalid encaxes={self.encaxes}')
                    elif self.enctype == 'ryrz': 
                        self.encaxes = ['ry', 'rz']
                        qc.h(qubit=self.memory_size+0*self.dimxqc+self.qin[qn])
                        qc.ry(np.arctan(phi_c) + np.pi/4, qubit=self.memory_size+step*self.dimxqc+self.qin[qn])
                        qc.rz(np.arctan(phi_c**2) + np.pi/4, qubit=self.memory_size+step*self.dimxqc+self.qin[qn])

                        
                    else:
                        raise Warning(f'Invalid enctype {self.enctype}')
                    qn += 1
                
                    

                
            for i in range(len(c_qubits)):
                #H
                qc.h(qubit=c_qubits[i])
                #qc.r(theta=np.pi,phi=0, qubit=c_qubits[i])



            # CZ
            for i in range(len(c_qubits)-1):
                qc.cz(control_qubit=c_qubits[i],target_qubit=c_qubits[i+1])
            qc.cz(control_qubit=c_qubits[-1],target_qubit=c_qubits[0])

            #XROT
            for i in range(len(c_qubits)):
                #X
                qc.rx(theta=thetac[i],qubit=c_qubits[i])
                #H
                qc.h(qubit=c_qubits[i])
                #qc.r(theta=np.pi,phi=0, qubit=c_qubits[i])

                

                



        
            #We will keep track of which measurements to perform here:
            if self.nmeas > 0:
                adj_meas=[k + self.memory_size for k in self.qmeas]
                
                self.all_meas  += [[Manual_Layout_List[step][k] for k in adj_meas]]

        #In the last step, we measure ALL qubits.
        self.all_meas[-1]=Manual_Layout_List[-1]

        #We now have to transpile the circuit!
        ## define the circuit to be run on the hardware
        opt_level = self.opt_level ## defines the optimization level


        #Not necessary...
        
        #if (self.QPU == 'Apollo'):
        #    qubit_names = ["QB3", "QB8", "QB4", "QB9", "QB13", "QB14", "QB5", "QB10", "QB15", "QB6", "QB11", "QB16", "QB7", "QB12", "QB17", "QB18", "QB19", "QB20", "QB1", "QB2"]
        #elif (self.QPU == 'Adonis'):
        #    qubit_names = ["QB1", "QB3", "QB2", "QB4", "QB5"]
        
        # We allow qiskit to perform transpilation!
        self.untranspiled_qc=qc
        tqc = transpile(qc,backend=self.backend,optimization_level=opt_level)
        # We need to see how the qubits have been permuted:

        init_perm_lib_qiskit=tqc.layout.initial_layout
        final_perm_lib_qiskit=tqc.layout.final_layout
        # This is the wrong form for our purposes: we need to swap it:
        end_mapping={final_perm_lib_qiskit[k].index:k for k in range(20)}
        begin_mapping={init_perm_lib_qiskit[k].index:k for k in range(20)}

        self.all_meas=[[end_mapping[begin_mapping[i]] for i in k] for k in self.all_meas]





        return tqc

    
    def _set_undefined_circuit(self):
        """Define quantum circuit at every timestep.

        Returns:
            self.unistep (qiskit.QuantumCircuit): unitary at every timestep.
        """

        # Useful definitions:
        #self.dimx_wo_copies = self.dimx
        #self.dimxqc = int(self.dimx * self.nenccopies)
        self.qin = [*range(self.dimxqc)]
        


        if self.lookback_max:
            ylookback = 1
            xlookback = 1
        else:
            ylookback = self.ylookback
            xlookback = self.xlookback
            
        lookback = self.xlookback

        self.memory_size=self.nqubits-self.dimxqc

        self.total_req=self.memory_size+lookback*self.dimxqc

        nsteps=lookback

        #All the parameters for the circuit.
        rng = Generator(PCG64(seed=self.rseed))
                    
        t = self.ising_t 
        #jmax = self.ising_jmax, 
        h = self.ising_h
            #mode = 'nn',
        wmax = self.ising_wmax 
        
        #jpositive = self.ising_jpositive
        wpositive = self.ising_wpositive
        rseed = self.rseed
        random = self.ising_random,
        if random:
            #if jpositive:
            #    jlow, jhigh = 0, jmax
            #else:
            #    jlow, jhigh = -jmax, jmax
            if wpositive:
                wlow, whigh = 0, wmax
            else:
                wlow, whigh = -wmax, wmax
        else:
            #jlow, jhigh = jmax, jmax
            wlow, whigh = wmax, wmax

        thetac=t * (h + rng.uniform(low=wlow, high=whigh,size=5)) # hardcoded size!

        if self.manual==True:
            qc= self.produce_manual_circuit(thetac,nsteps)
            print('Using Hard-coded layout')
            
            # only works if the number of inputs < 4 !
            assert self.lookback <= 4 , "longer circuits not manually encoded!"
            

                #Now we need to compile the circuit
                        
    
                        #aux_circc = QuantumCircuit(self.backend.num_qubits, qc.num_clbits) this is full circ..
                        #qc = qc.compose(optimize_single_qubit_gates(transpile(c_circuit, basis_gates=['r', 'cz'], optimization_level = opt_level, coupling_map=reduced_map)), qubits=init_layout)

        #print(qc.draw())
        else:
            #NotImplementedError('not yet implemented....')
            qc= self.produce_automatic_circuit(thetac,nsteps) 

        self.transpiled_uni=qc
        self.param_map={k.name:k for k in qc.parameters}

        return
    



        
    # We have split the "run" function up into multiple parts; for ease of exectution.
    #This part is mainly concerned with setting up the experiment.
    def prepare(self,dataobj):
        self.start_time = time.time()
        if self.log:
            self._init_logging()
        self.data = dataobj
        self._set_fitter()
        self._set_data_dims(self.data)

        #Some useful definitions:
        self.dimx_wo_copies = self.dimx
        if self.ylookback > 0:
            #print(self.dimx_wo_copies)
            self.dimx_wo_copies += self.dimy
            #print(self.dimx_wo_copies)
        
        self.dimxqc = int(self.dimx_wo_copies * self.nenccopies)
        
        assert self.dimxqc <= self.nqubits, f'{self.dimxqc} {self.nqubits}'
        # dimension of the input data at each step
       
        if self.ylookback > 0:
            self.dmin = np.asarray([self.xmin, self.ymin]).reshape(-1) # (dimxqc,)
            self.dmax = np.asarray([self.xmax, self.ymax]).reshape(-1) # (dimxqc,)
        else:
            self.dmin = np.asarray(self.xmin).reshape(-1) # (dimxqc,)
            self.dmax = np.asarray(self.xmax).reshape(-1) # (dimxqc,)
        self.qin = [*range(self.dimxqc)]
        self.quni = [*range(self.nqubits)]
        if self.mend:
            self.qmeas = [*range(self.nqubits-self.nmeas, self.nqubits)]
        else:
            self.qmeas = [*range(self.nmeas)]

        self._set_undefined_circuit()

    def _get_input_t(self, xe_lookback, ye_lookback, t_in, t_pred):
        """
        Every step is a mini-episode:
        At every step, repeat past n steps + current step. 
        At the current step, feed in current action x(t) and previous ouput y(t-1).
        At the past n'th step, feed in action x(t-n) and previous output y(t-n-1).
        In training previous outpus y are the true values from the data.
        
        Returns:
            angles (np.ndarray): (lookback, dimxqc)
        """
        if self.restarting:
            return np.hstack([
                xe_lookback[0:t_in+1],
                ye_lookback[0:t_in+1],
            ])
        else:
            if self.lookback_max:
                if self.ylookback > 0:
                    # t_pred = 4, t_in = 3, lookback = 3 -> 1, 2, 3
                    return np.hstack([
                        xe_lookback[max(0, t_in-self.ylookback+1):t_in+1],
                        ye_lookback[max(0, t_in-self.ylookback+1):t_in+1], 
                    ])
                else:
                    return xe_lookback[max(0, t_in-self.xlookback+1):t_in+1]
            else:
                if self.ylookback > 0:
                    return np.hstack([
                        xe_lookback[t_in:t_in+self.xlookback],
                        ye_lookback[t_in:t_in+self.ylookback], 
                    ])
                else:
                    return xe_lookback[t_in:t_in+self.xlookback]

    def define_training_circuit(self):
        
        #First we create a file to save the circuits in:
        self.file_name_train = self.file_name +"_train"
        if os.path.isdir(self.file_name_train+'_circuits'):
            None
        else:
            os.makedirs(self.file_name_train+'_circuits')

        if self.lookback_max:
            ylookback = 1
            xlookback = 1
        else:
            ylookback = self.ylookback
            xlookback = self.xlookback
        self.xtrain = [] # (episodes, steps, dimx)
        self.ftrain = [] # (episodes, steps, dimf)



        qc_list=[]

            
           

        lookback = self.xlookback

        nsteps=lookback

        #Only including 4 layouts: Remember the count starts at 0!
        #   Custom_Layouts=[{qr[0]: 15, qr[1]: 14, qr[2]: 9,qr[3]: 18, qr[5]: 9}, #t-3
        #                {qr[0]: 15, qr[1]: 14, qr[2]: 9,qr[3]: 8, qr[5]: 13}, #t-2
        #                {qr[0]: 15, qr[1]: 14, qr[2]: 9,qr[3]: 11, qr[5]: 6}, #t-1
        #                {qr[0]: 15, qr[1]: 14, qr[2]: 9,qr[3]: 5, qr[5]: 10}] #t=0
            

        #self.memory_size=self.nqubits-self.dimxqc

        #self.total_req=self.memory_size+lookback*self.dimxqc


           
            
        for e, xe in enumerate(self.data.xtrain):
            # steps = np.shape(xe)[0]
            steps_max = 200
            
            #This set up repeated versions of the first input..
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

                #means the episode has come to an end.
                if x0 == False:
                    break
                #We also save future y values:
                y1 = self.data.ytrain[e][t_pred]
                # save current ouput
                ye_lookback[t_in+ylookback] = y1 

                #Get the input to encode for both x and y.
                input_t = self._get_input_t(xe_lookback=xe_lookback, ye_lookback=ye_lookback, t_pred=t_pred, t_in=t_in)

                # f1 = self._t_input_to_t_features(input_t=input_t, x0=x0, t_pred=t_pred)
                #def _t_input_to_t_features(self, input_t, x0, t_pred):

                
                # encode input for quantum circuit
                #Now we have the angles to encode into the circuit.
                step_input_angles = self._angle_encoding(episode=input_t, dmin=self.dmin, dmax=self.dmax)
                # run circuit, get features
                if self.lookback_max:
                    lookback = min(self.xlookback, t_pred)
                else:
                    lookback = self.xlookback
                
                #features_qc = self._get_step_features(
                #    angles=step_input_angles, 
                #    nsteps=t_pred if self.restarting == True else lookback,
                #    saveqc=True if t_pred == step_to_save_qc else False,
                #)

                if self.restarting == True:
                    nsteps=t_pred  
                else: nsteps=lookback

                if t_pred == step_to_save_qc:
                    saveqc = True
                else:
                    saveqc = False
                    
                #def _get_step_features(self, angles, nsteps=1, saveqc=False):
                self.memory_size=self.nqubits-self.dimxqc

                self.total_req=self.memory_size+nsteps*self.dimxqc





                
                for nax, ax in enumerate(self.measaxes):
                    # First, we take the undefined uni
                    qc = self.transpiled_uni.copy()
                    if self.qinit == 'h':
                        print('not possible for this class')
                    #We then loop over the lookback, to feed in the previous values!
                    for prevstep in range(nsteps):
                        # input
                        qn = 0 # qubit counter for loop
                        for c in range(self.nenccopies):
                            for d in range(int(self.dimxqc / self.nenccopies)): # dimx
                                qc=qc.assign_parameters({self.param_map['phi_'+str(prevstep)+'_'+str(qn)]:step_input_angles[prevstep, d]})

                                qn += 1
                    # = 0 # qubit counter for loop
                    #for c in range(self.nenccopies):
                    #    for d in range(int(self.dimxqc / self.nenccopies)): # dimx
                    #        
                    #        qc=qc.assign_parameters({self.param_map['phi_'+str(nsteps-1)+'_'+str(qn)]:step_input_angles[nsteps-1, d]})
                    #
                    #        qn += 1

                    # One ALL steps are fed in, we can optimise the rotations:
                    #qc=optimize_single_qubit_gates(qc)
                    #We can now add the measurements:
                    qc.barrier()        
                    for prevstep in range(nsteps-1):

                        
                        
                        #
                        if self.nmeas > 0:
                            #We have to adjust the qubit number.
                            
                            adj_meas=self.all_meas[prevstep]
                            if self.reset_instead_meas:

                                warnings.warn("Reset not possible - qubit will not be measured")
                                #qc.reset(self.qmeas)
                                
                            elif self.use_partial_meas: #means use the measurements as features.
                                match ax:
                                    # https://arxiv.org/abs/1804.03719
                                    case 'z':
                                        pass
                                    case 'x':
                                        qc.r(theta=np.pi/2,phi=np.pi/2, qubit=adj_meas)
                                        qc.r(theta=np.pi,phi=0, qubit=adj_meas)

                                    case 'y':
                                        qc.r(theta=np.pi/2,phi=0, qubit=adj_meas)
                                        qc.r(theta=np.pi/2,phi=np.pi/2, qubit=adj_meas)
                                        qc.r(theta=-np.pi/2,phi=0, qubit=adj_meas)
                                        qc.r(theta=np.pi/2,phi=np.pi/2, qubit=adj_meas)
                                        qc.r(theta=np.pi,phi=0, qubit=adj_meas)
                                    case _:
                                        raise Warning(f'Invalid measaxes {self.measaxes}')
                                qc.measure(
                                    qubit=adj_meas,
                                    cbit=adj_meas # for convenience, we match up the cbits and qbits.
                                )
                            else:
                                warnings.warn("All measurements are mapped to the first classical registers - are you sure you want this?")
                                # the cbits will be overwritten at every step, only the last one will be kept
                                qc.measure(qubit=adj_meas, cbit=[*range(self.nmeas)])
                    # final step
                    # input

                    #self._add_input_to_qc(qc=qc, angles=step_input_angles, step=nsteps-1)            # unitary 
                    #qc.append(self.unistep,[*range(self.memory_size)]+[*range(self.memory_size+(nsteps-1)*self.dimxqc,self.memory_size+(nsteps)*self.dimxqc)])
                    # measure
                    final_meas=self.all_meas[-1] #Should contain ALL QUBITS.
                    match ax:
                        # https://arxiv.org/abs/1804.03719
                        case 'z':
                            pass
                        case 'x':
                            qc.r(theta=np.pi/2,phi=np.pi/2, qubit=final_meas)
                            qc.r(theta=np.pi,phi=0, qubit=final_meas)
                        case 'y':
                            qc.r(theta=np.pi/2,phi=0, qubit=final_meas)
                            qc.r(theta=np.pi/2,phi=np.pi/2, qubit=final_meas)
                            qc.r(theta=-np.pi/2,phi=0, qubit=final_meas)
                            qc.r(theta=np.pi/2,phi=np.pi/2, qubit=final_meas)
                            qc.r(theta=np.pi,phi=0, qubit=final_meas)
                        case _:
                            raise Warning(f'Invalid measaxes {self.measaxes}')
                    qc.measure(qubit=final_meas, cbit=final_meas) #we have replaced the self.cbits_final with final_meas.
                    if saveqc:
                        self.qc = qc
                        
                        #print(self.qc)

                    

                    #compiled_qc = transpile(qc, self.backend)
                    # Replace with Alessio's Transpilation.
                    #pre_trans=pre_trans+[qc]
                    #qc_aux1 = transpile(qc, self.backend, coupling_map=self.reduced_coupling_map,optimization_level=2)
                    #qc_aux2 = transpile(qc_aux1, self.backend, optimization_level=0, initial_layout=self.initial_layout)
                    #compiled_qc= optimize_single_qubit_gates(qc_aux2)

                    # Here we have to stop!!!
                    qc_list=qc_list+[qc]

            
                #IMPORTANT: Must update the time in parameter 
                t_in+=1

            #Printed every episode.
            if self.verbose ==True:
                print('transpiled circuits %s' % (np.round(e/len(self.data.xtrain),4)),end='\r')
        
        #Done!
        self.qc_list=qc_list
        with open(self.file_name_train+'_circuits/saved_circuits','wb') as saved_circ:
            qpy.dump(qc_list,saved_circ)
        
        if self.verbose==True:
            print('',end='\n')
            print('saved circuits')
        
        return 
    
    def interpret_training_results(self,count_list=None):
        # To be run when ALL circuits have been successfully run.
        self.file_name_train = self.file_name +"_train"
        if count_list==None:
            try:
                with open(self.file_name_train+'/results.json','r') as read_results:
                            count_list=json.load(read_results)
                if self.verbose:
                    print('Loaded counts from file')
            except:
                NotADirectoryError('Could not find results file!')

            
            
        if self.lookback_max:
            ylookback = 1
            xlookback = 1
        else:
            ylookback = self.ylookback
            xlookback = self.xlookback
        self.xtrain = [] # (episodes, steps, dimx)
        self.ftrain = [] # (episodes, steps, dimf)


        lookback = self.xlookback

        nsteps=lookback

        self.memory_size=self.nqubits-self.dimxqc

        self.total_req=self.memory_size+lookback*self.dimxqc

        circuit_count=0

        for e, xe in enumerate(self.data.xtrain):
            steps_max = 200
            xe_lookback, ye_lookback = self._init_t_inputs(
            x0=self.data.xtrain[e][0], y0=self.data.ytrain[e][0], 
            steps_max=steps_max
                )
            t_in = 0
            xe = [] # actions in this episode
            fe = [] # features in this episode

            for t_pred in range(self.xyoffset, steps_max):
                x0 = self._policy(0, e=e, step=t_in, train=True, offset=self.xyoffset)
                #x0 = self._policy(0, e=e, step=t_in, train=True, offset=self.xyoffset)
                # save current input
                xe_lookback[t_in+xlookback-1] = x0
                if x0 == False:
                    break

                features_step = []

                for nax, ax in enumerate(self.measaxes):

                    counts = count_list[circuit_count]

                    circuit_count = circuit_count + 1 

                    

                    # qiskit counts are in reverse order
                    counts = {k[::-1]: v for k, v in counts.items()}
                    # turn measurements into features

                    ### Code could be improved here....
                    ####

                    if self.use_partial_meas:
                        for prev_step in range(nsteps-1):
                            counts_step = {}
                            
                            for b, c in counts.items():
                                
                                adj_meas  =self.all_meas[prev_step]
                                b_step = citemgetter(adj_meas,b) # change range.
                                if b_step in counts_step.keys():
                                    counts_step[b_step] += c
                                else:
                                    counts_step[b_step] = c
                            #print('before:',len(features_step))
                            features_step = self._step_meas_to_step_features(counts_step, features_step,True)
                            
                        # final step
                        counts_final = {}
                        
                        for b, c in counts.items():
                            final_meas=self.all_meas[-1]
                            b_final = citemgetter(final_meas,b)
                            if b_final in counts_final.keys():
                                counts_final[b_final] += c
                            else:
                                counts_final[b_final] = c
                        features_step = self._step_meas_to_step_features(counts_final, features_step,True)
                        
                    else:
                        counts_final = {}
                        #features_step = self._step_meas_to_step_features(counts_step=counts, features_step=features_step)

                        for b, c in counts.items():
                            final_meas=self.all_meas[-1]
                            b_final = citemgetter(final_meas,b)

                            if b_final in counts_final.keys():
                                counts_final[b_final] += c
                            else:
                                counts_final[b_final] = c

                        #Here this measurement is added to the features for this step.
                        features_step = self._step_meas_to_step_features(counts_final, features_step,True)
                #features_qc is now called features_step 

                #Now all measurements are done!
                if self.add_x_as_feature:
                    
                    #features_step = np.hstack(features_step + [x0])
                    #print(features_step,self.data.xtrain[e][t_pred])
                    features_step = np.hstack(features_step + [x0])
                else:
                    features_step = np.hstack(features_step)


                #f1 is now called feature_step
                    
                y1 = self.data.ytrain[e][t_pred]
                # save current ouput
                ye_lookback[t_in+ylookback] = y1 
                # save 
                xe.append(x0)
                fe.append(features_step) # (steps, 1, dimf)
                t_in += 1
            # episode over
            if self.xyoffset == 1:
                xe += [self.data.xtrain[e][-1]] # add last step
            self.xtrain.append(np.vstack(xe))
            self.ftrain.append(np.vstack(fe)) # (episodes, steps-xyoffset, dimf)
            assert np.allclose(self.xtrain[e], self.data.xtrain[e]), f'{self.xtrain[e] - self.data.xtrain[e]}'
                
        # all episodes over
        
        self.dimf = np.shape(features_step)[1]
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
        try:
            self.ytrain = [
                self.weights.predict(fe)[:, :self.dimy].reshape(np.shape(fe)[0], self.dimy) # step 1, ..., n
                for fe in self.ftrain
            ]
        except:
            assert self.dimy == 1, 'incorrect dimy!'
            self.ytrain = [
                (self.weights.predict(fe).reshape(-1,1))[:, :self.dimy].reshape(np.shape(fe)[0], self.dimy) # step 1, ..., n
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
        elif self.data.ynorm == 'std':
            self.ytrain_nonorm = unstandardize(data=self.ytrain, dmin=self.data.ymin_nonorm, dmax=self.data.ymax_nonorm,doldmin=self.data.ymin, doldmax=self.data.ymax)
        elif self.data.ynorm == 'scale':
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
    
    def run_circuits(self,case_in=None,Time=0,wait=False,recheck=601):
        # This function is used to run circuits via the exectuor, or simulator...
        try: 
            qc_list=self.qc_list
        except:
                if self.verbose:
                    print('No list found: loading from file')

                if case_in.casefold()=='train':
                    self.file_name_train = self.file_name +"_train"
                    with open(self.file_name_train+'_circuits/saved_circuits','rb') as saved_circ:
                        qc_list=qpy.load(saved_circ)
                elif case_in.casefold()=='val' or case_in.casefold()=='validate' or case_in.casefold()=='validation':
                    self.file_name_val=self.file_name+'_val'
                    with open(self.file_name_val+('/%s/saved_circuits' % (Time)),'rb') as circ_file:
                        qc_list=qpy.load(circ_file)
        
        if case_in.casefold()=='train':
            run_filename=self.file_name +"_train"
        elif case_in.casefold()=='val' or case_in.casefold()=='validate' or case_in.casefold()=='validation':
            run_filename=self.file_name+'_val/%s' % Time
        else:
            NotImplementedError('only "train" or "val" allowed as cases!')
        if self.simulate: #just use the Aersimulator to get results.
            sim_backend = AerSimulator(method='statevector')
            count_list=sim_backend.run(qc_list,shots=self.shots).result().get_counts()
        else:

            if wait==False:
                #Run just one instance of the executor.
                
                self.executor.run(run_filename, qc_list, shots = self.shots, verbose = self.verbose, calibration_set_id = self.calibration_set_id)
            else:
                self.executor.run(run_filename, qc_list, shots = self.shots, verbose = self.verbose, calibration_set_id = self.calibration_set_id)
                if self.verbose :
                    print('Waiting mode enabled: Will try to resubmit every %s seconds' % (recheck))
                    number_completed=0
                    number_required=self.executor.number_of_batches
                    while number_completed < number_required:
                        time.sleep(recheck)
                        status_update,full_up=self.executor.status(run_filename,update=True)
                        status_dict={k[0]:k[1] for k in status_update}
                        if status_dict['DONE']>= number_required:
                            break
                        if (status_dict['INITIALIZING']+status_dict['QUEUED']+status_dict['RUNNING']+status_dict['VALIDATING'])<self.executor.max_concurrent:
                            self.executor.run(run_filename, qc_list, shots = self.shots, verbose = self.verbose, calibration_set_id = self.calibration_set_id)
                        #status_update,full_up=self.executor.status(self.file_name_train,update=True)
                        print(''*55,end='\r')
                        print(status_update,end='\r')

            
            #check that all files are done, then we can retrieve them.
            status_update,full_up=self.executor.status(run_filename,update=True)
            status_dict={k[0]:k[1] for k in status_update}
            if status_dict['DONE']>= self.executor.number_of_batches:
                count_list=self.executor.return_results(run_filename,update=True)

        if self.simulate:
            if os.path.isdir(run_filename):
                None
            else:
                os.mkdir(run_filename)
            with open(run_filename+'/results.json','w') as results_file:
                json.dump(count_list,results_file,indent=2)
        else:
            if status_dict['DONE']>= self.executor.number_of_batches:
                with open(run_filename+'/results.json','w') as results_file:
                    json.dump(count_list,results_file,indent=2)


        self.count_list=count_list
        return
    

#################################################################################################
    
    def analy_train(self,features='None'):
            #Need expectation values!
        nin=self.dimxqc
        nmem=self.memory_size
        tMeas=self.all_meas
        XL = []
        YL = []
        ZL = []
        partialX=list(''.join(k) for k in product('IX',repeat=nin))
        partialX.remove('II')
        partialY=list(''.join(k) for k in product('IY',repeat=nin))
        partialY.remove('II')
        partialZ=list(''.join(k) for k in product('IZ',repeat=nin))
        partialZ.remove('II')
        TotalExp='I'*20
        nsteps=self.lookback
        for i in range(nsteps-1):
            
            XL += [replacer(TotalExp,k,tMeas[i]) for k  in partialX ]
            YL += [replacer(TotalExp,k,tMeas[i]) for k  in partialY ]
            ZL += [replacer(TotalExp,k,tMeas[i]) for k  in partialZ ]
        
        
        #Now add manually the final measurements:
        #fbody=self.ftype
        #original_meas_x=['I'*(nmem+nin-fcurr)+'X'*fcurr for fcurr in range(1,fbody+1)]
        #original_meas_y=['I'*(nmem+nin-fcurr)+'Y'*fcurr for fcurr in range(1,fbody+1)]
        #original_meas_z=['I'*(nmem+nin-fcurr)+'Z'*fcurr for fcurr in range(1,fbody+1)]
        #for fcurr in range(fbody):
        #    xstrings=list(set(list(''.join(k) for k in permutations(original_meas_x[fcurr], nmem+nin))))
        #    xfinal=[replacer(TotalExp,k,tMeas[-1]) for k in xstrings]
        #    XL += xfinal
        #    ystrings=list(set(list(''.join(k) for k in permutations(original_meas_y[fcurr], nmem+nin))))
        #    yfinal=[replacer(TotalExp,k,tMeas[-1]) for k in ystrings]
        #    YL += yfinal
        #    zstrings=list(set(list(''.join(k) for k in permutations(original_meas_z[fcurr], nmem+nin))))
        #    zfinal=[replacer(TotalExp,k,tMeas[-1]) for k in zstrings]
        #    ZL += zfinal

        #We add the measurements in the same order as the finite shot case:
        fbody=self.ftype
        for fcurr in range(1,fbody+1):
            positions=list(combinations(tMeas[-1],fcurr))
            xfinal=[replacer(TotalExp,'X'*fcurr,pos) for pos in positions]
            XL += xfinal
            
            yfinal=[replacer(TotalExp,'Y'*fcurr,pos) for pos in positions]
            YL += yfinal
            
            zfinal=[replacer(TotalExp,'Z'*fcurr,pos) for pos in positions]
            ZL += zfinal

            

        if self.lookback_max:
            ylookback = 1
            xlookback = 1
        else:
            ylookback = self.ylookback
            xlookback = self.xlookback
        self.xtrain = [] # (episodes, steps, dimx)
        self.ftrain = [] # (episodes, steps, dimf)


        lookback = self.xlookback

        nsteps=lookback

        self.memory_size=self.nqubits-self.dimxqc

        self.total_req=self.memory_size+lookback*self.dimxqc

        circuit_count=0

        for e, xe in enumerate(self.data.xtrain):
            steps_max = 200
            xe_lookback, ye_lookback = self._init_t_inputs(
            x0=self.data.xtrain[e][0], y0=self.data.ytrain[e][0], 
            steps_max=steps_max
                )
            t_in = 0
            xe = [] # actions in this episode
            fe = [] # features in this episode

            for t_pred in range(self.xyoffset, steps_max):

                x0 = self._policy(0, e=e, step=t_in, train=True, offset=self.xyoffset)
                #x0 = self._policy(0, e=e, step=t_in, train=True, offset=self.xyoffset)
                # save current input
                xe_lookback[t_in+xlookback-1] = x0
                if x0 == False:
                    break

                y1 = self.data.ytrain[e][t_pred]
                # save current ouput
                ye_lookback[t_in+ylookback] = y1 

                #Get the input to encode for both x and y.
                input_t = self._get_input_t(xe_lookback=xe_lookback, ye_lookback=ye_lookback, t_pred=t_pred, t_in=t_in)

                step_input_angles = self._angle_encoding(episode=input_t, dmin=self.dmin, dmax=self.dmax)
                # run circuit, get features
                if self.lookback_max:
                    lookback = min(self.xlookback, t_pred)
                else:
                    lookback = self.xlookback

                if self.restarting == True:
                    nsteps=t_pred  
                else: nsteps=lookback

                if t_pred == step_to_save_qc:
                    saveqc = True
                else:
                    saveqc = False
                    
                #def _get_step_features(self, angles, nsteps=1, saveqc=False):
                self.memory_size=self.nqubits-self.dimxqc

                self.total_req=self.memory_size+nsteps*self.dimxqc

                features_step = []

                ########################
                qc=self.transpiled_uni
                param_map={k.name:k for k in qc.parameters}
                shape_phi=step_input_angles.shape
                phi_assignment={param_map['phi_%s_%s' % (k,s)]:step_input_angles[k,s] for k in range(shape_phi[0]) for s in range(shape_phi[1])}
                explicit_val=qc.assign_parameters(phi_assignment)
                explicit_val.save_statevector()
                backend = AerSimulator(method='statevector')
                job =  backend.run(explicit_val,shots=1, memory=True)
                job_result = job.result()
                sv=job_result.get_statevector(explicit_val)

                X = [sv.expectation_value(Pauli(k[::-1])) for k in XL]
                Y = [sv.expectation_value(Pauli(k[::-1])) for k in YL]
                Z = [sv.expectation_value(Pauli(k[::-1])) for k in ZL]
                ########################
                features_step= np.array([Z + Y + X])

                #Now all measurements are done!
                if self.add_x_as_feature:
                    
                    #features_step = np.hstack(features_step + [x0])
                    #print(features_step,self.data.xtrain[e][t_pred])
                    features_step = np.hstack(features_step + [x0])
                else:
                    if features.casefold()=='with_inputs':
                        flat_inputs=[[k for j in input_t for k in j]]
                        features_step = np.hstack([features_step,np.array(flat_inputs)])
                    elif features.casefold()=='only_inputs':
                        flat_inputs=[[k for j in input_t for k in j]]
                        features_step = np.hstack( [np.array(flat_inputs)])
                    else:
                        None


                #f1 is now called feature_step
                    
                y1 = self.data.ytrain[e][t_pred]
                # save current ouput
                ye_lookback[t_in+ylookback] = y1 
                # save 
                xe.append(x0)
                fe.append(features_step) # (steps, 1, dimf)
                t_in += 1
            # episode over
            if self.xyoffset == 1:
                xe += [self.data.xtrain[e][-1]] # add last step
            self.xtrain.append(np.vstack(xe))
            self.ftrain.append(np.vstack(fe)) # (episodes, steps-xyoffset, dimf)
            assert np.allclose(self.xtrain[e], self.data.xtrain[e]), f'{self.xtrain[e] - self.data.xtrain[e]}'
                
        # all episodes over
        self.dimf = np.shape(features_step)[1]
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
        try:
            self.ytrain = [
                self.weights.predict(fe)[:, :self.dimy].reshape(np.shape(fe)[0], self.dimy) # step 1, ..., n
                for fe in self.ftrain
            ]
        except:
            assert self.dimy == 1, 'incorrect dimy!'
            self.ytrain = [
                (self.weights.predict(fe).reshape(-1,1))[:, :self.dimy].reshape(np.shape(fe)[0], self.dimy) # step 1, ..., n
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
        elif self.data.ynorm == 'scale':
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
    

    def analy_val(self,infmode='data',features='None'):
                #Need expectation values!
            nin=self.dimxqc
            nmem=self.memory_size
            tMeas=self.all_meas
            XL = []
            YL = []
            ZL = []
            partialX=list(''.join(k) for k in product('IX',repeat=nin))
            partialX.remove('II')
            partialY=list(''.join(k) for k in product('IY',repeat=nin))
            partialY.remove('II')
            partialZ=list(''.join(k) for k in product('IZ',repeat=nin))
            partialZ.remove('II')
            TotalExp='I'*20
            nsteps=self.lookback
            for i in range(nsteps-1):
                
                XL += [replacer(TotalExp,k,tMeas[i]) for k  in partialX ]
                YL += [replacer(TotalExp,k,tMeas[i]) for k  in partialY ]
                ZL += [replacer(TotalExp,k,tMeas[i]) for k  in partialZ ]
            
            
            #Now add manually the final measurements:
            #fbody=self.ftype
            #original_meas_x=['I'*(nmem+nin-fcurr)+'X'*fcurr for fcurr in range(1,fbody+1)]
            #original_meas_y=['I'*(nmem+nin-fcurr)+'Y'*fcurr for fcurr in range(1,fbody+1)]
            #original_meas_z=['I'*(nmem+nin-fcurr)+'Z'*fcurr for fcurr in range(1,fbody+1)]
            #for fcurr in range(fbody):
            #    xstrings=list(set(list(''.join(k) for k in permutations(original_meas_x[fcurr], nmem+nin))))
            #    xfinal=[replacer(TotalExp,k,tMeas[-1]) for k in xstrings]
            #    XL += xfinal
            #    ystrings=list(set(list(''.join(k) for k in permutations(original_meas_y[fcurr], nmem+nin))))
            #    yfinal=[replacer(TotalExp,k,tMeas[-1]) for k in ystrings]
            #    YL += yfinal
            #    zstrings=list(set(list(''.join(k) for k in permutations(original_meas_z[fcurr], nmem+nin))))
            #    zfinal=[replacer(TotalExp,k,tMeas[-1]) for k in zstrings]
            #    ZL += zfinal

                    #We add the measurements in the same order as the finite shot case:
            fbody=self.ftype
            for fcurr in range(1,fbody+1):
                positions=list(combinations(tMeas[-1],fcurr))
                xfinal=[replacer(TotalExp,'X'*fcurr,pos) for pos in positions]
                XL += xfinal
                
                yfinal=[replacer(TotalExp,'Y'*fcurr,pos) for pos in positions]
                YL += yfinal
                
                zfinal=[replacer(TotalExp,'Z'*fcurr,pos) for pos in positions]
                ZL += zfinal
    
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


            #inputs_all=[[] for e in range(nepisodes)]
            t_in = 0
            steps_max = 200


            lookback = self.xlookback

            nsteps=lookback

            self.memory_size=self.nqubits-self.dimxqc

            self.total_req=self.memory_size+lookback*self.dimxqc


            circuit_count=0
            #for e, xe in enumerate(self.data.xtrain):
            for e in range(nepisodes):
                steps_max = 200
                xe_lookback, ye_lookback = self._init_t_inputs(
                x0=self.data.xval[e][0], y0=self.data.yval[e][0], 
                steps_max=steps_max
                    )
                t_in = 0
                xe = [] # actions in this episode
                fe = [] # features in this episode
                ye = [] #predictions in this episode.

                for t_pred in range(self.xyoffset, steps_max):

                    x0 = self._policy(0, e=e, step=t_in, train=False, offset=self.xyoffset)
                    #x0 = self._policy(0, e=e, step=t_in, train=True, offset=self.xyoffset)
                    # save current input
                    xe_lookback[t_in+xlookback-1] = x0
                    if x0 == False:
                        break

 

                    #Get the input to encode for both x and y.
                    input_t = self._get_input_t(xe_lookback=xe_lookback, ye_lookback=ye_lookback, t_pred=t_pred, t_in=t_in)

                    step_input_angles = self._angle_encoding(episode=input_t, dmin=self.dmin, dmax=self.dmax)
                    # run circuit, get features
                    if self.lookback_max:
                        lookback = min(self.xlookback, t_pred)
                    else:
                        lookback = self.xlookback

                    if self.restarting == True:
                        nsteps=t_pred  
                    else: nsteps=lookback

                    if t_pred == step_to_save_qc:
                        saveqc = True
                    else:
                        saveqc = False
                        
                    #def _get_step_features(self, angles, nsteps=1, saveqc=False):
                    self.memory_size=self.nqubits-self.dimxqc

                    self.total_req=self.memory_size+nsteps*self.dimxqc

                    features_step = []

                    ########################
                    qc=self.transpiled_uni
                    param_map={k.name:k for k in qc.parameters}
                    shape_phi=step_input_angles.shape
                    phi_assignment={param_map['phi_%s_%s' % (k,s)]:step_input_angles[k,s] for k in range(shape_phi[0]) for s in range(shape_phi[1])}
                    explicit_val=qc.assign_parameters(phi_assignment)
                    explicit_val.save_statevector()
                    backend = AerSimulator(method='statevector')
                    job =  backend.run(explicit_val,shots=1, memory=True)
                    job_result = job.result()
                    sv=job_result.get_statevector(explicit_val)

                    X = [sv.expectation_value(Pauli(k[::-1])) for k in XL]
                    Y = [sv.expectation_value(Pauli(k[::-1])) for k in YL]
                    Z = [sv.expectation_value(Pauli(k[::-1])) for k in ZL]
                    ########################
                    features_step= np.array([Z + Y + X])

                    #Now all measurements are done!
                    if self.add_x_as_feature:
                        
                        #features_step = np.hstack(features_step + [x0])
                        #print(features_step,self.data.xtrain[e][t_pred])
                        features_step = np.hstack(features_step + [x0])
                    else:
                        if features.casefold()=='with_inputs':
                            flat_inputs=[[k for j in input_t for k in j]]
                            features_step = np.hstack([features_step,np.array(flat_inputs)])
                        elif features.casefold()=='only_inputs':
                            flat_inputs=[k for j in input_t for k in j]
                            features_step = np.hstack( [[np.array(flat_inputs)]])
                        else:
                            None                #f1 is now called feature_step

                    #This part is all from "val"

                    # predict output
                    y1 = max(
                        min(
                            self.weights.predict(features_step)[:, :self.dimy], 
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
                    # save - HERE WE HAVE TO CHANGE, DUE TO NEW FOR LOOPS. 
                    ye.append(y1)
                    
                    xe.append(x0)
                    fe.append(features_step) # (steps, 1, dimf)
                    t_in += 1

                if self.xyoffset > 0:
                    xe += [self.data.xval[e][-1]] # add last step
                    fe = [fe[0]] + fe # add first step
                    ye = [self.data.yval[e][0]] + ye # add first step
                
                self.xval.append(np.vstack(xe))
                self.fval.append(np.vstack(fe)) # (episodes, steps-xyoffset, dimf)
                self.yval.append(np.vstack(ye))
                assert np.allclose(self.xval[e], self.data.xval[e]), f'{self.xval[e] - self.data.xval[e]}'
                # assert np.shape(self.xval[-1])[0] == np.shape(self.fval[-1])[0], f'{np.shape(self.xval[-1])} {np.shape(self.fval[-1])} : {np.shape(self.data.xval[e])}'
                assert np.shape(self.xval[-1])[0] == np.shape(self.yval[-1])[0], f'{np.shape(self.xval[-1])} {np.shape(self.yval[-1])} : {np.shape(self.data.xval[e])}'
        # all episodes over
        # unnormalize
            if self.data.ynorm == 'norm':
                self.yval_nonorm = unnormalize(data=self.yval, dmin=self.data.ymin_nonorm, dmax=self.data.ymax_nonorm)
            elif self.data.ynorm == 'scale':
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

        
    def train(self,resume=True,wait=False,recheck=601):
        self.file_name_train = self.file_name +"_train"
        if resume==False:
            if self.confirm==False:
                confirm='y'
            else:
                confirm=input('This will delete any previously stored data - continue? y/n \n')
            if (confirm.casefold()=='yes' or confirm.casefold()=='y'):
                if os.path.isdir(self.file_name_train+'_circuits'):
                    shutil.rmtree(self.file_name_train+'_circuits')
                    os.makedirs(self.file_name_train+'_circuits')
                if os.path.isdir(self.file_name_train):
                    shutil.rmtree(self.file_name_train)
            #we now go through all steps...
            self.define_training_circuit()
            self.run_circuits(case_in='train',Time=None,wait=wait,recheck=recheck)
            if os.path.isfile(self.file_name_train+'/results.json'):
                self.interpret_training_results(self.count_list)
            else:
                print('cannot continue: please finish running circuits')


        else:
            if os.path.isdir(self.file_name_train+'_circuits'):
                with open(self.file_name_train+'_circuits/saved_circuits','rb') as saved_circuits:
                    self.qc = qpy.load(saved_circuits)

                if os.path.isfile(self.file_name_train+'/results.json'):
                    with open(self.file_name_train+'/results.json','r') as results:
                        self.count_list=json.load(results)
                    
                    if self.verbose:
                        print('resuming on interpretation')
                    self.interpret_training_results(self.count_list)
                else:
                    print('resuming on evaluating circuits')
                    self.run_circuits(case_in='train',Time=None,wait=wait,recheck=recheck)
                    if os.path.isfile(self.file_name_train+'/results.json'):
                        self.interpret_training_results(self.count_list)
                    else:
                        print('cannot continue: please finish running circuits')


            else:
                print('resuming on defining circuits')
                self.define_training_circuit()
                self.run_circuits(case_in='train',Time=None,wait=wait,recheck=recheck)
                if os.path.isfile(self.file_name_train+'/results.json'):
                    self.interpret_training_results(self.count_list)
                else:
                    print('cannot continue: please finish running circuits')

        return
    
    def reinterpret_with_inputs(self,count_list=None):
        # To be run when ALL circuits have been successfully run.
        self.file_name_train = self.file_name +"_train"
        if count_list==None:
            try:
                with open(self.file_name_train+'/results.json','r') as read_results:
                            count_list=json.load(read_results)
                if self.verbose:
                    print('Loaded counts from file')
            except:
                NotADirectoryError('Could not find results file!')

            
            
        if self.lookback_max:
            ylookback = 1
            xlookback = 1
        else:
            ylookback = self.ylookback
            xlookback = self.xlookback
        self.xtrain = [] # (episodes, steps, dimx)
        self.ftrain = [] # (episodes, steps, dimf)


        lookback = self.xlookback

        nsteps=lookback

        self.memory_size=self.nqubits-self.dimxqc

        self.total_req=self.memory_size+lookback*self.dimxqc

        circuit_count=0


        """
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

                #means the episode has come to an end.
                if x0 == False:
                    break
                #We also save future y values:
                y1 = self.data.ytrain[e][t_pred]
                # save current ouput
                ye_lookback[t_in+ylookback] = y1 

                #Get the input to encode for both x and y.
                input_t = self._get_input_t(xe_lookback=xe_lookback, ye_lookback=ye_lookback, t_pred=t_pred, t_in=t_in)
"""

        for e, xe in enumerate(self.data.xtrain):
            steps_max = 200
            xe_lookback, ye_lookback = self._init_t_inputs(
            x0=self.data.xtrain[e][0], y0=self.data.ytrain[e][0], 
            steps_max=steps_max
                )
            t_in = 0
            xe = [] # actions in this episode
            fe = [] # features in this episode

            for t_pred in range(self.xyoffset, steps_max):
                x0 = self._policy(0, e=e, step=t_in, train=True, offset=self.xyoffset)
                #x0 = self._policy(0, e=e, step=t_in, train=True, offset=self.xyoffset)
                # save current input
                xe_lookback[t_in+xlookback-1] = x0
                if x0 == False:
                    break
                #We want to add this to the features.
                input_t = self._get_input_t(xe_lookback=xe_lookback, ye_lookback=ye_lookback, t_pred=t_pred, t_in=t_in)

                features_step = []

                for nax, ax in enumerate(self.measaxes):

                    counts = count_list[circuit_count]

                    circuit_count = circuit_count + 1 

                    

                    # qiskit counts are in reverse order
                    counts = {k[::-1]: v for k, v in counts.items()}
                    # turn measurements into features

                    ### Code could be improved here....
                    ####

                    if self.use_partial_meas:
                        for prev_step in range(nsteps-1):
                            counts_step = {}
                            
                            for b, c in counts.items():
                                
                                adj_meas  =self.all_meas[prev_step]
                                b_step = citemgetter(adj_meas,b) # change range.
                                if b_step in counts_step.keys():
                                    counts_step[b_step] += c
                                else:
                                    counts_step[b_step] = c
                            features_step = self._step_meas_to_step_features(counts_step, features_step,True)
                        # final step
                        counts_final = {}
                        
                        for b, c in counts.items():
                            final_meas=self.all_meas[-1]
                            b_final = citemgetter(final_meas,b)
                            if b_final in counts_final.keys():
                                counts_final[b_final] += c
                            else:
                                counts_final[b_final] = c
                        features_step = self._step_meas_to_step_features(counts_final, features_step,True)
                    else:
                        counts_final = {}
                        #features_step = self._step_meas_to_step_features(counts_step=counts, features_step=features_step)

                        for b, c in counts.items():
                            final_meas=self.all_meas[-1]
                            b_final = citemgetter(final_meas,b)

                            if b_final in counts_final.keys():
                                counts_final[b_final] += c
                            else:
                                counts_final[b_final] = c

                        #Here this measurement is added to the features for this step.
                        features_step = self._step_meas_to_step_features(counts_final, features_step,True)
                #features_qc is now called features_step 

                #Now all measurements are done!
                if True:
                    flat_inputs=[[k for j in input_t for k in j]]
                
                    #features_step = np.hstack(features_step + [x0])
                    #print(features_step,self.data.xtrain[e][t_pred])
                    features_step = np.hstack(features_step + [np.array(flat_inputs)])
                else:
                    features_step = np.hstack(features_step)

                #f1 is now called feature_step
                    
                y1 = self.data.ytrain[e][t_pred]
                # save current ouput
                ye_lookback[t_in+ylookback] = y1 
                # save 
                xe.append(x0)
                fe.append(features_step) # (steps, 1, dimf)
                t_in += 1
            # episode over
            if self.xyoffset == 1:
                xe += [self.data.xtrain[e][-1]] # add last step
            self.xtrain.append(np.vstack(xe))
            self.ftrain.append(np.vstack(fe)) # (episodes, steps-xyoffset, dimf)
            assert np.allclose(self.xtrain[e], self.data.xtrain[e]), f'{self.xtrain[e] - self.data.xtrain[e]}'
                
        # all episodes over
        self.dimf = np.shape(features_step)[1]
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
        elif self.data.ynorm == 'scale':
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

    def reinterpret_only_inputs(self,count_list=None):
        # To be run when ALL circuits have been successfully run.
        self.file_name_train = self.file_name +"_train"
        if count_list==None:
            try:
                with open(self.file_name_train+'/results.json','r') as read_results:
                            count_list=json.load(read_results)
                if self.verbose:
                    print('Loaded counts from file')
            except:
                NotADirectoryError('Could not find results file!')

            
            
        if self.lookback_max:
            ylookback = 1
            xlookback = 1
        else:
            ylookback = self.ylookback
            xlookback = self.xlookback
        self.xtrain = [] # (episodes, steps, dimx)
        self.ftrain = [] # (episodes, steps, dimf)


        lookback = self.xlookback

        nsteps=lookback

        self.memory_size=self.nqubits-self.dimxqc

        self.total_req=self.memory_size+lookback*self.dimxqc

        circuit_count=0


        """
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

                #means the episode has come to an end.
                if x0 == False:
                    break
                #We also save future y values:
                y1 = self.data.ytrain[e][t_pred]
                # save current ouput
                ye_lookback[t_in+ylookback] = y1 

                #Get the input to encode for both x and y.
                input_t = self._get_input_t(xe_lookback=xe_lookback, ye_lookback=ye_lookback, t_pred=t_pred, t_in=t_in)
"""

        for e, xe in enumerate(self.data.xtrain):
            steps_max = 200
            xe_lookback, ye_lookback = self._init_t_inputs(
            x0=self.data.xtrain[e][0], y0=self.data.ytrain[e][0], 
            steps_max=steps_max
                )
            t_in = 0
            xe = [] # actions in this episode
            fe = [] # features in this episode

            for t_pred in range(self.xyoffset, steps_max):
                x0 = self._policy(0, e=e, step=t_in, train=True, offset=self.xyoffset)
                #x0 = self._policy(0, e=e, step=t_in, train=True, offset=self.xyoffset)
                # save current input
                xe_lookback[t_in+xlookback-1] = x0
                if x0 == False:
                    break
                #We want to add this to the features.
                input_t = self._get_input_t(xe_lookback=xe_lookback, ye_lookback=ye_lookback, t_pred=t_pred, t_in=t_in)

                features_step = []

                for nax, ax in enumerate(self.measaxes):

                    counts = count_list[circuit_count]

                    circuit_count = circuit_count + 1 

                    

                    # qiskit counts are in reverse order
                    counts = {k[::-1]: v for k, v in counts.items()}
                    # turn measurements into features

                    ### Code could be improved here....
                    ####

                    if self.use_partial_meas:
                        for prev_step in range(nsteps-1):
                            counts_step = {}
                            
                            for b, c in counts.items():
                                
                                adj_meas  =self.all_meas[prev_step]
                                b_step = citemgetter(adj_meas,b) # change range.
                                if b_step in counts_step.keys():
                                    counts_step[b_step] += c
                                else:
                                    counts_step[b_step] = c
                            features_step = self._step_meas_to_step_features(counts_step, features_step,True)
                        # final step
                        counts_final = {}
                        
                        for b, c in counts.items():
                            final_meas=self.all_meas[-1]
                            b_final = citemgetter(final_meas,b)
                            if b_final in counts_final.keys():
                                counts_final[b_final] += c
                            else:
                                counts_final[b_final] = c
                        features_step = self._step_meas_to_step_features(counts_final, features_step,True)
                    else:
                        counts_final = {}
                        #features_step = self._step_meas_to_step_features(counts_step=counts, features_step=features_step)

                        for b, c in counts.items():
                            final_meas=self.all_meas[-1]
                            b_final = citemgetter(final_meas,b)

                            if b_final in counts_final.keys():
                                counts_final[b_final] += c
                            else:
                                counts_final[b_final] = c

                        #Here this measurement is added to the features for this step.
                        features_step = self._step_meas_to_step_features(counts_final, features_step,True)
                #features_qc is now called features_step 

                #Now all measurements are done!
                if True:
                    flat_inputs=[[k for j in input_t for k in j]]
                
                    #features_step = np.hstack(features_step + [x0])
                    #print(features_step,self.data.xtrain[e][t_pred])

                    #we overwrite the features just with the inputs.
                    features_step = np.hstack( [np.array(flat_inputs)])
                else:
                    features_step = np.hstack(features_step)

                #f1 is now called feature_step
                    
                y1 = self.data.ytrain[e][t_pred]
                # save current ouput
                ye_lookback[t_in+ylookback] = y1 
                # save 
                xe.append(x0)
                fe.append(features_step) # (steps, 1, dimf)
                t_in += 1
            # episode over
            if self.xyoffset == 1:
                xe += [self.data.xtrain[e][-1]] # add last step
            self.xtrain.append(np.vstack(xe))
            self.ftrain.append(np.vstack(fe)) # (episodes, steps-xyoffset, dimf)
            assert np.allclose(self.xtrain[e], self.data.xtrain[e]), f'{self.xtrain[e] - self.data.xtrain[e]}'
                
        # all episodes over
        self.dimf = np.shape(features_step)[1]
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
        elif self.data.ynorm == 'scale':
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

    def val(self, infmode='data', nepisodes=None,resume=False,wait=True,recheck=601):
        self.file_name_val = self.file_name +"_val"
        if resume==False:
            if self.confirm==False:
                confirm='y'
            else:
                confirm=input('This will delete any previously stored data - continue? y/n \n')
            if (confirm.casefold()=='yes' or confirm.casefold()=='y'):
                if os.path.isdir(self.file_name_val+'_circuits'):
                    shutil.rmtree(self.file_name_val+'_circuits')
                    os.makedirs(self.file_name_val+'_circuits')
                if os.path.isdir(self.file_name_val):
                    shutil.rmtree(self.file_name_val)
        

        if os.path.isdir(self.file_name_val):
            None
        else:
            os.makedirs(self.file_name_val)


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

        xe_all = [[] for e in range(nepisodes)] # actions for ALL episodes!
        fe_all = [[] for e in range(nepisodes)] # features in this episode
        ye_all = [[] for e in range(nepisodes)]
        ypred_all = [[] for e in range(nepisodes)]
        #inputs_all=[[] for e in range(nepisodes)]
        t_in = 0
        steps_max = 200

        try:
            qc = self.transpiled_uni.copy()
        except:
            raise NotImplementedError("Must train before validation!")

        for t_pred in range(self.xyoffset,steps_max):
            print(''*20,end='\r')
            print(t_pred,end='\r')
            
            completed=0
            if resume==True:
                    if os.path.isdir(self.file_name_val+('/%s' % (t_pred))):
                        if os.path.isfile(self.file_name_val+('/%s' % (t_pred))+'/results.json'):
                            #No need to rerun circuits

                            ##HAVE TO CHECK IF ANALYSIS DONE!!
                            completed=1
                            

            if completed==0: # if completed is not 1, we have to check manually.
                try:
                    #If circuits are saved; we can load them.
                    with open(self.file_name_val+('/%s/saved_circuits' % (t_pred)),'rb') as circ_file:
                        if resume==False:
                            os.remove(circ_file)
                        qc_list=qpy.load(circ_file)
                    with open(self.file_name_val+('/%s/saved_xe.pkl' % (t_pred)),'rb') as xe_file:
                        xe_all=pkl.load(xe_file)
                    with open(self.file_name_val+('/%s/saved_ye.pkl' % (t_pred)),'rb') as ye_file:
                        ye_all=pkl.load(ye_file)
                except:
                    #if we cannot load the transpiled circuits, we have to do it manually.
                    
                    #try to load the previous iteration's files.
                    try:
                        with open(self.file_name_val+('/%s/saved_xe.pkl' % (t_pred-1)),'rb') as xe_file:
                            xe_all=pkl.load(xe_file)
                        with open(self.file_name_val+('/%s/saved_ye.pkl' % (t_pred-1)),'rb') as ye_file:
                            ye_all=pkl.load(ye_file)
                    except:
                        None


                    qc_list,xe_all,ye_all =self.define_validation_circuit(t_pred,nepisodes,t_in,xe_all,ye_all)
                    #print(xe_all,ye_all)
                    
                    with open(self.file_name_val+('/%s/saved_xe.pkl' % (t_pred)),'wb') as xe_file:
                        pkl.dump(xe_all,xe_file)
                    with open(self.file_name_val+('/%s/saved_ye.pkl' % (t_pred)),'wb') as ye_file:
                       pkl.dump(ye_all,ye_file)
                if len(qc_list)<1: # if there are no more circuits to be run!
                    break
                #now we have to run the circuits!
                self.run_circuits(case_in='val',Time=t_pred,wait=wait,recheck=recheck)
                #regardless of completion, need to run interpretation:
                if os.path.isfile(self.file_name_val+'/%s/results.json' % t_pred):
                     
                     #have to load the correct files:
                    try:
                        with open(self.file_name_val+('/%s/saved_ye.pkl' % (t_pred)),'rb') as ye_file:
                            ye_all=pkl.load(ye_file)
                        with open(self.file_name_val+('/%s/saved_fe.pkl' % (t_pred-1)),'rb') as fe_file:
                            fe_all=pkl.load(fe_file)
                        with open(self.file_name_val+('/%s/saved_ypred.pkl' % (t_pred-1)),'rb') as ypred_file:
                            ypred_all=pkl.load(ypred_file)
                    except:
                        None


                    fe_all,ye_all,ypred_all=self.interpret_validation_results(t_pred,nepisodes,t_in,xe_all,fe_all,ye_all,ypred_all,self.count_list,feature_form='normal')
                    with open(self.file_name_val+('/%s/saved_ye.pkl' % (t_pred)),'wb') as ye_file:
                        pkl.dump(ye_all,ye_file)
                    with open(self.file_name_val+('/%s/saved_fe.pkl' % (t_pred)),'wb') as fe_file:
                        pkl.dump(fe_all,fe_file)
                    with open(self.file_name_val+('/%s/saved_ypred.pkl' % (t_pred)),'wb') as ypred_file:
                        pkl.dump(ypred_all,ypred_file)
                else:
                    print('cannot continue: please finish running circuits')  
                    break
            #finally, we update t_in.
            t_in+=1
                # all episodes over
                
        # if we broke the loop, then we must take the previous time step:
        final_t=t_pred-1

        # one more loop:
        with open(self.file_name_val+('/%s/saved_xe.pkl' % (final_t)),'rb') as xe_file:
            xe_all=pkl.load(xe_file)
        with open(self.file_name_val+('/%s/saved_ypred.pkl' % (final_t)),'rb') as ypred_file:
            ypred_all=pkl.load(ypred_file)
        with open(self.file_name_val+('/%s/saved_fe.pkl' % (final_t)),'rb') as fe_file:
            fe_all=pkl.load(fe_file)
        for e in range(nepisodes):
            
            step_length=len(self.data.xval[e])
            relevant_x=xe_all[e][xlookback-1:step_length+xlookback-1]
            if self.xyoffset == 1:
                relevant_x[-1]= self.data.xval[e][-1] # add last step
            #need to start in the right place
            self.xval.append(np.vstack(relevant_x))
        
            self.fval.append(np.vstack(fe_all[e][:step_length])) # (episodes, steps, dimf)
            # add first step - this will be deleted later!
            relevant_y=ypred_all[e][:step_length]
            relevant_y.insert(0,np.array([self.data.yval[e][0]]))
            self.yval.append(np.vstack(relevant_y)) # (episodes, steps, dimy) 
            assert np.allclose(self.xval[e], self.data.xval[e]), f'{self.xval[e] - self.data.xval[e]}'
            # assert np.shape(self.xval[-1])[0] == np.shape(self.fval[-1])[0], f'{np.shape(self.xval[-1])} {np.shape(self.fval[-1])} : {np.shape(self.data.xval[e])}'
            assert np.shape(self.xval[-1])[0] == np.shape(self.yval[-1])[0], f'{np.shape(self.xval[-1])} {np.shape(self.yval[-1])} : {np.shape(self.data.xval[e])}'
    


        # unnormalize
        if self.data.ynorm == 'norm':
            self.yval_nonorm = unnormalize(data=self.yval, dmin=self.data.ymin_nonorm, dmax=self.data.ymax_nonorm)
        elif self.data.ynorm == 'scale':
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
        
    def val_with_inputs(self, infmode='data', nepisodes=None,resume=False,wait=True,recheck=601):
        self.file_name_val = self.file_name +"_val"
        if resume==False:
            if self.confirm==False:
                confirm='y'
            else:
                confirm=input('This will delete any previously stored data - continue? y/n \n')
            if (confirm.casefold()=='yes' or confirm.casefold()=='y'):
                if os.path.isdir(self.file_name_val+'_circuits'):
                    shutil.rmtree(self.file_name_val+'_circuits')
                    os.makedirs(self.file_name_val+'_circuits')
                if os.path.isdir(self.file_name_val):
                    shutil.rmtree(self.file_name_val)
        

        if os.path.isdir(self.file_name_val):
            None
        else:
            os.makedirs(self.file_name_val)


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

        xe_all = [[] for e in range(nepisodes)] # actions for ALL episodes!
        fe_all = [[] for e in range(nepisodes)] # features in this episode
        ye_all = [[] for e in range(nepisodes)]
        ypred_all = [[] for e in range(nepisodes)]
        #inputs_all=[[] for e in range(nepisodes)]
        t_in = 0
        steps_max = 200

        try:
            qc = self.transpiled_uni.copy()
        except:
            raise NotImplementedError("Must train before validation!")

        for t_pred in range(self.xyoffset,steps_max):
            print(''*20,end='\r')
            print(t_pred,end='\r')
            
            completed=0
            if resume==True:
                    if os.path.isdir(self.file_name_val+('/%s' % (t_pred))):
                        if os.path.isfile(self.file_name_val+('/%s' % (t_pred))+'/results.json'):
                            #No need to rerun circuits

                            ##HAVE TO CHECK IF ANALYSIS DONE!!
                            completed=1
                            

            if completed==0: # if completed is not 1, we have to check manually.
                try:
                    #If circuits are saved; we can load them.
                    with open(self.file_name_val+('/saved_circuits/%s' % (t_pred)),'rb') as circ_file:
                        if resume==False:
                            os.remove(circ_file)
                        qc_list=qpy.load(circ_file)
                    with open(self.file_name_val+('/saved_circuits/%s/saved_xe.pkl' % (t_pred)),'rb') as xe_file:
                        xe_all=pkl.load(xe_file)
                    with open(self.file_name_val+('/saved_circuits/%s/saved_ye.pkl' % (t_pred)),'rb') as ye_file:
                        ye_all=pkl.load(ye_file)
                except:
                    #if we cannot load the transpiled circuits, we have to do it manually.
                    
                    qc_list,xe_all,ye_all =self.define_validation_circuit(t_pred,nepisodes,t_in,xe_all,ye_all)
                    #print(xe_all,ye_all)
                    #if os.path.isdir(self.file_name_val+('/%s' % (t_pred)))==False:
                    #    os.mkdir(self.file_name_val+('/%s' % (t_pred)))
                    with open(self.file_name_val+('/saved_circuits/%s/saved_xe.pkl' % (t_pred)),'wb') as xe_file:
                        pkl.dump(xe_all,xe_file)
                    with open(self.file_name_val+('/saved_circuits/%s/saved_ye.pkl' % (t_pred)),'wb') as ye_file:
                       pkl.dump(ye_all,ye_file)
                if len(qc_list)<1: # if there are no more circuits to be run!
                    break
                #now we have to run the circuits!
                self.run_circuits(case_in='val',Time=t_pred,wait=wait,recheck=recheck)
                #regardless of completion, need to run interpretation:
                if os.path.isfile(self.file_name_val+'/%s/results.json' % t_pred):
                    fe_all,ye_all,ypred_all=self.interpret_validation_results(t_pred,nepisodes,t_in,xe_all,fe_all,ye_all,ypred_all,self.count_list,feature_form='add_inputs')
                    with open(self.file_name_val+('/saved_circuits/%s/saved_ye.pkl' % (t_pred)),'wb') as ye_file:
                        pkl.dump(ye_all,ye_file)
                    with open(self.file_name_val+('/saved_circuits/%s/saved_fe.pkl' % (t_pred)),'wb') as fe_file:
                        pkl.dump(fe_all,fe_file)
                    with open(self.file_name_val+('/saved_circuits/%s/saved_ypred.pkl' % (t_pred)),'wb') as ypred_file:
                        pkl.dump(ypred_all,ypred_file)
                else:
                    print('cannot continue: please finish running circuits')  
                    break
            #finally, we update t_in.
            t_in+=1
                # all episodes over
                
        # if we broke the loop, then we must take the previous time step:
        final_t=t_pred-1

        # one more loop:
        with open(self.file_name_val+('/saved_circuits/%s/saved_xe.pkl' % (final_t)),'rb') as xe_file:
            xe_all=pkl.load(xe_file)
        with open(self.file_name_val+('/saved_circuits/%s/saved_ypred.pkl' % (final_t)),'rb') as ypred_file:
            ypred_all=pkl.load(ypred_file)
        with open(self.file_name_val+('/saved_circuits/%s/saved_fe.pkl' % (final_t)),'rb') as fe_file:
            fe_all=pkl.load(fe_file)
        for e in range(nepisodes):
            
            step_length=len(self.data.xval[e])
            relevant_x=xe_all[e][xlookback-1:step_length+xlookback-1]
            if self.xyoffset == 1:
                relevant_x[-1]= self.data.xval[e][-1] # add last step
            #need to start in the right place
            self.xval.append(np.vstack(relevant_x))
        
            self.fval.append(np.vstack(fe_all[e][:step_length])) # (episodes, steps, dimf)
            # add first step - this will be deleted later!
            relevant_y=ypred_all[e][:step_length]
            relevant_y.insert(0,np.array([self.data.yval[e][0]]))
            self.yval.append(np.vstack(relevant_y)) # (episodes, steps, dimy) 
            assert np.allclose(self.xval[e], self.data.xval[e]), f'{self.xval[e] - self.data.xval[e]}'
            # assert np.shape(self.xval[-1])[0] == np.shape(self.fval[-1])[0], f'{np.shape(self.xval[-1])} {np.shape(self.fval[-1])} : {np.shape(self.data.xval[e])}'
            assert np.shape(self.xval[-1])[0] == np.shape(self.yval[-1])[0], f'{np.shape(self.xval[-1])} {np.shape(self.yval[-1])} : {np.shape(self.data.xval[e])}'
    


        # unnormalize
        if self.data.ynorm == 'norm':
            self.yval_nonorm = unnormalize(data=self.yval, dmin=self.data.ymin_nonorm, dmax=self.data.ymax_nonorm)
        elif self.data.ynorm == 'scale':
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

    def val_only_inputs(self, infmode='data', nepisodes=None,resume=False,wait=True,recheck=601):
        self.file_name_val = self.file_name +"_val"
        if resume==False:
            if self.confirm==False:
                confirm='y'
            else:
                confirm=input('This will delete any previously stored data - continue? y/n \n')
            if (confirm.casefold()=='yes' or confirm.casefold()=='y'):
                if os.path.isdir(self.file_name_val+'_circuits'):
                    shutil.rmtree(self.file_name_val+'_circuits')
                    os.makedirs(self.file_name_val+'_circuits')
                if os.path.isdir(self.file_name_val):
                    shutil.rmtree(self.file_name_val)
        

        if os.path.isdir(self.file_name_val):
            None
        else:
            os.makedirs(self.file_name_val)


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

        xe_all = [[] for e in range(nepisodes)] # actions for ALL episodes!
        fe_all = [[] for e in range(nepisodes)] # features in this episode
        ye_all = [[] for e in range(nepisodes)]
        ypred_all = [[] for e in range(nepisodes)]
        #inputs_all=[[] for e in range(nepisodes)]
        t_in = 0
        steps_max = 200

        try:
            qc = self.transpiled_uni.copy()
        except:
            raise NotImplementedError("Must train before validation!")

        for t_pred in range(self.xyoffset,steps_max):
            print(''*20,end='\r')
            print(t_pred,end='\r')
            
            completed=0
            if resume==True:
                    if os.path.isdir(self.file_name_val+('/%s' % (t_pred))):
                        if os.path.isfile(self.file_name_val+('/%s' % (t_pred))+'/results.json'):
                            #No need to rerun circuits

                            ##HAVE TO CHECK IF ANALYSIS DONE!!
                            completed=1
                            

            if completed==0: # if completed is not 1, we have to check manually.
                try:
                    #If circuits are saved; we can load them.
                    with open(self.file_name_val+('/%s/saved_circuits' % (t_pred)),'rb') as circ_file:
                        if resume==False:
                            os.remove(circ_file)
                        qc_list=qpy.load(circ_file)
                    with open(self.file_name_val+('/saved_circuits/%s/saved_xe.pkl' % (t_pred)),'rb') as xe_file:
                        xe_all=pkl.load(xe_file)
                    with open(self.file_name_val+('/saved_circuits/%s/saved_ye.pkl' % (t_pred)),'rb') as ye_file:
                        ye_all=pkl.load(ye_file)
                except:
                    #if we cannot load the transpiled circuits, we have to do it manually.
                    
                    qc_list,xe_all,ye_all =self.define_validation_circuit(t_pred,nepisodes,t_in,xe_all,ye_all)
                    #print(xe_all,ye_all)
                    
                    with open(self.file_name_val+('/saved_circuits/%s/saved_xe.pkl' % (t_pred)),'wb') as xe_file:
                        pkl.dump(xe_all,xe_file)
                    with open(self.file_name_val+('/saved_circuits/%s/saved_ye.pkl' % (t_pred)),'wb') as ye_file:
                       pkl.dump(ye_all,ye_file)
                if len(qc_list)<1: # if there are no more circuits to be run!
                    break
                #now we have to run the circuits!
                self.run_circuits(case_in='val',Time=t_pred,wait=wait,recheck=recheck)
                #regardless of completion, need to run interpretation:
                if os.path.isfile(self.file_name_val+'/%s/results.json' % t_pred):
                    fe_all,ye_all,ypred_all=self.interpret_validation_results(t_pred,nepisodes,t_in,xe_all,fe_all,ye_all,ypred_all,self.count_list,feature_form='inputs_only')
                    with open(self.file_name_val+('/saved_circuits/%s/saved_ye.pkl' % (t_pred)),'wb') as ye_file:
                        pkl.dump(ye_all,ye_file)
                    with open(self.file_name_val+('/saved_circuits/%s/saved_fe.pkl' % (t_pred)),'wb') as fe_file:
                        pkl.dump(fe_all,fe_file)
                    with open(self.file_name_val+('/saved_circuits/%s/saved_ypred.pkl' % (t_pred)),'wb') as ypred_file:
                        pkl.dump(ypred_all,ypred_file)
                else:
                    print('cannot continue: please finish running circuits')  
                    break
            #finally, we update t_in.
            t_in+=1
                # all episodes over
                
        # if we broke the loop, then we must take the previous time step:
        final_t=t_pred-1

        # one more loop:
        with open(self.file_name_val+('/saved_circuits/%s/saved_xe.pkl' % (final_t)),'rb') as xe_file:
            xe_all=pkl.load(xe_file)
        with open(self.file_name_val+('/saved_circuits/%s/saved_ypred.pkl' % (final_t)),'rb') as ypred_file:
            ypred_all=pkl.load(ypred_file)
        with open(self.file_name_val+('/saved_circuits/%s/saved_fe.pkl' % (final_t)),'rb') as fe_file:
            fe_all=pkl.load(fe_file)
        for e in range(nepisodes):
            
            step_length=len(self.data.xval[e])
            relevant_x=xe_all[e][xlookback-1:step_length+xlookback-1]
            if self.xyoffset == 1:
                relevant_x[-1]= self.data.xval[e][-1] # add last step
            #need to start in the right place
            self.xval.append(np.vstack(relevant_x))
        
            self.fval.append(np.vstack(fe_all[e][:step_length])) # (episodes, steps, dimf)
            # add first step - this will be deleted later!
            relevant_y=ypred_all[e][:step_length]
            relevant_y.insert(0,np.array([self.data.yval[e][0]]))
            self.yval.append(np.vstack(relevant_y)) # (episodes, steps, dimy) 
            assert np.allclose(self.xval[e], self.data.xval[e]), f'{self.xval[e] - self.data.xval[e]}'
            # assert np.shape(self.xval[-1])[0] == np.shape(self.fval[-1])[0], f'{np.shape(self.xval[-1])} {np.shape(self.fval[-1])} : {np.shape(self.data.xval[e])}'
            assert np.shape(self.xval[-1])[0] == np.shape(self.yval[-1])[0], f'{np.shape(self.xval[-1])} {np.shape(self.yval[-1])} : {np.shape(self.data.xval[e])}'
    


        # unnormalize
        if self.data.ynorm == 'norm':
            self.yval_nonorm = unnormalize(data=self.yval, dmin=self.data.ymin_nonorm, dmax=self.data.ymax_nonorm)
        elif self.data.ynorm == 'scale':
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


        







    def define_validation_circuit(self,Time, nepisodes,t_in,xe_all,ye_all):
        if self.lookback_max:
            ylookback = 1
            xlookback = 1
        else:
            ylookback = self.ylookback
            xlookback = self.xlookback
        #creates the necessary circuits for a single validation time step.
        qc_list=[]
        for e in range(nepisodes):
            # steps = np.shape(xe)[0]
            steps_max = 200
            if t_in==0:
                xe_lookback, ye_lookback = self._init_t_inputs(
                    x0=self.data.xval[e][0], y0=self.data.yval[e][0], 
                    steps_max=steps_max
                )
                xe_all[e]=xe_lookback
                ye_all[e]=ye_lookback
                # save for evaluation
            xe_lookback=xe_all[e]
            ye_lookback=ye_all[e]
            #if e==0:
            #    print(xe_lookback[:10])


            # get x(t-1) (current action)
            x0 = self._policy(0, e=e, step=t_in, train=False, offset=self.xyoffset)
            # save current input
            
            xe_lookback[t_in+xlookback-1] = x0
            if x0 == False:
                continue # not break as we want to carry on with other e!
            input_t = self._get_input_t(xe_lookback=xe_lookback, ye_lookback=ye_lookback, t_pred=Time, t_in=t_in)
            # make features out of x(t) and previous {y(t-lookback)}
            

            #Here have to add in the edits!!!!!

            #f1 = self._t_input_to_t_features(input_t=input_t, x0=x0, t_pred=t_pred)
            #def _t_input_to_t_features(self, input_t, x0, t_pred):

            
            # encode input for quantum circuit
            step_input_angles = self._angle_encoding(episode=input_t, dmin=self.dmin, dmax=self.dmax)
            # run circuit, get features
            if self.lookback_max:
                lookback = min(self.xlookback, Time)
            else:
                lookback = self.xlookback
            


            #features_qc = self._get_step_features(
                
                #angles=step_input_angles, 
                #nsteps=t_pred if self.restarting == True else lookback,
                #saveqc=True if t_pred == step_to_save_qc else False,

            if self.restarting == True:
                nsteps=Time
            else: nsteps=lookback

            if Time == step_to_save_qc:
                saveqc = True
            else:
                saveqc = False
                

            self.memory_size=self.nqubits-self.dimxqc

            self.total_req=self.memory_size+nsteps*self.dimxqc

            for nax, ax in enumerate(self.measaxes):
                
                qc = self.transpiled_uni.copy() # define the large circuit first of all.
                if self.qinit == 'h':
                    print('not possible for this class')
                for prevstep in range(nsteps-1):
                    # input
                    qn = 0 # qubit counter for loop
                    for c in range(self.nenccopies):
                        for d in range(int(self.dimxqc / self.nenccopies)): # dimx
                            qc=qc.assign_parameters({self.param_map['phi_'+str(prevstep)+'_'+str(qn)]:step_input_angles[prevstep, d]})

                            qn += 1
                qn = 0 # qubit counter for loop
                for c in range(self.nenccopies):
                    for d in range(int(self.dimxqc / self.nenccopies)): # dimx
                        
                        qc=qc.assign_parameters({self.param_map['phi_'+str(nsteps-1)+'_'+str(qn)]:step_input_angles[nsteps-1, d]})

                        qn += 1
                # qc=optimize_single_qubit_gates(qc)
                qc.barrier()        
                for prevstep in range(nsteps-1):

                    
                    
                    #
                    if self.nmeas > 0:
                        #We have to adjust the qubit number.
                        
                        #adj_meas=all_meas[prevstep]
                        
                        adj_meas  = self.all_meas[prevstep]
                        if self.reset_instead_meas:

                            warnings.warn("Reset not possible - qubit will not be measured")
                            #qc.reset(self.qmeas)
                            
                        elif self.use_partial_meas: #means use the measurements as features.
                            match ax:
                                # https://arxiv.org/abs/1804.03719
                                case 'z':
                                    pass
                                case 'x':
                                    qc.r(theta=np.pi/2,phi=np.pi/2, qubit=adj_meas)
                                    qc.r(theta=np.pi,phi=0, qubit=adj_meas)

                                case 'y':
                                    qc.r(theta=np.pi/2,phi=0, qubit=adj_meas)
                                    qc.r(theta=np.pi/2,phi=np.pi/2, qubit=adj_meas)
                                    qc.r(theta=-np.pi/2,phi=0, qubit=adj_meas)
                                    qc.r(theta=np.pi/2,phi=np.pi/2, qubit=adj_meas)
                                    qc.r(theta=np.pi,phi=0, qubit=adj_meas)
                                case _:
                                    raise Warning(f'Invalid measaxes {self.measaxes}')
                            qc.measure(
                                qubit=adj_meas,
                                cbit=adj_meas # for convenience, we match up the cbits and qbits.
                            )
                        else:
                            warnings.warn("All measurements are mapped to the first classical registers - are you sure you want this?")
                            # the cbits will be overwritten at every step, only the last one will be kept
                            qc.measure(qubit=adj_meas, cbit=[*range(self.nmeas)])
                # final step
                # input

                #self._add_input_to_qc(qc=qc, angles=step_input_angles, step=nsteps-1)            # unitary 
                #qc.append(self.unistep,[*range(self.memory_size)]+[*range(self.memory_size+(nsteps-1)*self.dimxqc,self.memory_size+(nsteps)*self.dimxqc)])
                # measure
                final_meas=adj_meas  = self.all_meas[-1]
                match ax:
                    # https://arxiv.org/abs/1804.03719
                    case 'z':
                        pass
                    case 'x':
                        qc.r(theta=np.pi/2,phi=np.pi/2, qubit=final_meas)
                        qc.r(theta=np.pi,phi=0, qubit=final_meas)
                    case 'y':
                        qc.r(theta=np.pi/2,phi=0, qubit=final_meas)
                        qc.r(theta=np.pi/2,phi=np.pi/2, qubit=final_meas)
                        qc.r(theta=-np.pi/2,phi=0, qubit=final_meas)
                        qc.r(theta=np.pi/2,phi=np.pi/2, qubit=final_meas)
                        qc.r(theta=np.pi,phi=0, qubit=final_meas)
                    case _:
                        raise Warning(f'Invalid measaxes {self.measaxes}')
                qc.measure(qubit=final_meas, cbit=final_meas) #we have replaced the self.cbits_final with final_meas.
                if saveqc:
                    self.qc = qc
                    
                #print(self.qc)

                        
    
                        #compiled_qc = transpile(qc, self.backend)
                        # Replace with Alessio's Transpilation.
                        #pre_trans=pre_trans+[qc]
                        #qc_aux1 = transpile(qc, self.backend, coupling_map=self.reduced_coupling_map,optimization_level=2)
                        #qc_aux2 = transpile(qc_aux1, self.backend, optimization_level=0, initial_layout=self.initial_layout)
                        #compiled_qc= optimize_single_qubit_gates(qc_aux2)

                        # Here we have to stop!!!


            

                            # Here we have to stop!!!
                qc_list=qc_list+[qc]
        #os.mkdir(self.file_name_val+('/%s' % (Time)))
        if os.path.isdir(self.file_name_val+('/saved_circuits'))==False :
            os.mkdir(self.file_name_val+('/saved_circuits'))
        if os.path.isdir(self.file_name_val+('/saved_circuits/%s' % (Time)))==False :
            os.mkdir(self.file_name_val+('/saved_circuits/%s' % (Time)))
#       if os.path.isdir(self.file_name_val+('/saved_circuits/%s/saved_circuits' % (Time))) == False :
#            os.mkdir(self.file_name_val+('/saved_circuits/%s/saved_circuits' % (Time)))
        if len(qc_list)>0:
            with open(self.file_name_val+('/saved_circuits/%s/saved_circuits' % (Time)),'wb') as circ_file:                        
                qpy.dump(qc_list,circ_file)

        self.qc_list=qc_list
        return qc_list,xe_all,ye_all
    
    def interpret_validation_results(self,Time,nepisodes,t_in,xe_all,fe_all,ye_all,ypred_all,count_list=None,feature_form='normal'):
                    
        # To be run when ALL circuits have been successfully run.
        self.file_name_val = self.file_name +"_val"

        if count_list==None:
            try:
                #self.file_name+'_val/%s' % Time
                with open((self.file_name_val+'/%s/results.json' % Time),'r') as read_results:
                            count_list=json.load(read_results)
                if self.verbose:
                    print('Loaded counts from file')
            except:
                NotADirectoryError('Could not find results file!')


        if self.lookback_max:
            ylookback = 1
            xlookback = 1
        else:
            ylookback = self.ylookback
            xlookback = self.xlookback
    #    self.xtrain = [] # (episodes, steps, dimx)
    #    self.ftrain = [] # (episodes, steps, dimf)


        lookback = self.xlookback

        nsteps=lookback

        self.memory_size=self.nqubits-self.dimxqc

        self.total_req=self.memory_size+lookback*self.dimxqc


        circuit_count=0
        #for e, xe in enumerate(self.data.xtrain):
        for e in range(nepisodes):
            xe_lookback=xe_all[e]
            ye_lookback=ye_all[e]


            # get x(t-1) (current action)
            x0 = self._policy(0, e=e, step=t_in, train=False, offset=self.xyoffset)
            # save current input
            #xe_lookback[t_in+xlookback-1] = x0
            if x0 == False:
                continue # not break as we want to carry on with other e!
            input_t = self._get_input_t(xe_lookback=xe_lookback, ye_lookback=ye_lookback, t_pred=Time, t_in=t_in)

            features_step = []

            for nax, ax in enumerate(self.measaxes):

                counts = count_list[circuit_count]
                #if e==0:
                #    print(len(count_list))
                circuit_count = circuit_count + 1 

                

                # qiskit counts are in reverse order
                counts = {k[::-1]: v for k, v in counts.items()}
                # turn measurements into features

                ### Code could be improved here....
                ####

                if self.use_partial_meas:
                    for prev_step in range(nsteps-1):
                        counts_step = {}
                        
                        for b, c in counts.items():
                            
                            adj_meas  = self.all_meas[prev_step]
                            b_step = citemgetter(adj_meas,b) # change range.
                            if b_step in counts_step.keys():
                                counts_step[b_step] += c
                            else:
                                counts_step[b_step] = c
                        features_step = self._step_meas_to_step_features(counts_step, features_step,True)
                    # final step
                    counts_final = {}
                    
                    for b, c in counts.items():
                        final_meas=self.all_meas[-1]
                        b_final = citemgetter(final_meas,b)
                        if b_final in counts_final.keys():
                            counts_final[b_final] += c
                        else:
                            counts_final[b_final] = c
                    features_step = self._step_meas_to_step_features(counts_final, features_step,True)
                else:
                    counts_final = {}
                    #features_step = self._step_meas_to_step_features(counts_step=counts, features_step=features_step)

                    for b, c in counts.items():
                        final_meas=self.all_meas[-1]
                        b_final = citemgetter(final_meas,b)

                        if b_final in counts_final.keys():
                            counts_final[b_final] += c
                        else:
                            counts_final[b_final] = c

                    features_step = self._step_meas_to_step_features(counts_final, features_step,True)
            #features_qc is now called features_step 
            if feature_form=='normal':
                if self.add_x_as_feature:
                    features_step = np.hstack(features_step + [x0]) 
                else:
                    features_step = np.hstack(features_step)
            elif feature_form=='add_inputs':
                flat_inputs=[[k for j in input_t for k in j]]
                features_step = np.hstack(features_step + [np.array(flat_inputs)])
            elif feature_form=='inputs_only':
                flat_inputs=[[k for j in input_t for k in j]]
                features_step = np.hstack([np.array(flat_inputs)])
                

            #f1 is now called feature_step

            #This part is all from "val"

            # predict output
            y1 = max(
                min(
                    self.weights.predict(features_step)[:, :self.dimy], 
                    np.asarray(self.ymax)
                ), 
                np.asarray(self.ymin)
            ).reshape(1, -1)
            if self.use_true_y_in_val: # debugging only
                # get true output
                ye_lookback[t_in+ylookback] = self.data.yval[e][Time]
            else:
                # save current ouput
                ye_lookback[t_in+ylookback] = y1 
            # save - HERE WE HAVE TO CHANGE, DUE TO NEW FOR LOOPS. 
            ypred_all[e].append(y1)
                
            
            #We dont update xe as already done.....
            #xe_all[e].append(x0)
            fe_all[e].append(features_step) # (steps, 1, dimf)

            # but we do update ye_all.
            ye_all[e]=ye_lookback


        return fe_all,ye_all,ypred_all


    


    