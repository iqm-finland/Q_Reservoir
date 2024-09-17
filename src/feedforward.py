# -*- coding: utf-8 -*-
"""Contains the classes :class:`.CPolynomialFeedforward` and :class:`.QExtremeLearningMachine`.

:class:`.CPolynomialFeedforward` is a classical polynomial fitter for comparisons.

:class:`.QExtremeLearningMachine` is a feedforward (MLP) style model with a quantum circuit as a feature extractor.
It works well, but on the reactor and Narma task having no unitary at all works just as well as a quantum unitary.
This indicates that in our trials, all the power comes from encoding the information and measuring it, 
basically transforming the basis into polynomials of sin and cos.
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import numpy as np
import sys
import os

sys.path.append(os.path.abspath('..')) # module: qrc_surrogate

from src.basemodel import PredictionModelBase, QuantumBase, StepwiseModelBase
from src.helpers import weighted_gate_count


step_to_save_qc = 5

class FeedForwardBase(PredictionModelBase, StepwiseModelBase):
    """Base class for FeedForward like methods.
    Provides a training loop for :class:`.CPolynomialFeedforward` and :class:`.QExtremeLearningMachine`.

    Arguments:
        xlookback (int): how many past x's are used as input to this step
        ylookback (int): how many past y's are used as input to this step
    
    Additional arguments are passed to :class:`src.basemodel.PredictionModelBase` 
    and :class:`src.basemodel.StepwiseModelBase`.
    """
    model_name = 'feedforwardbase'

    def __init__(
        self,
        # FeedForwardBase
        xlookback,
        ylookback,
        # StepwiseModelBase
        xyoffset,
        set_past_y_to_0 ,
        use_true_y_in_val,
        lookback_max,
        # PredictionModelBase
        washout,
        log,
        rseed,
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
        PredictionModelBase.__init__(
            self,
            washout = washout,
            log = log,
            rseed = rseed,
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
        )
        StepwiseModelBase.__init__(
            self,
            xyoffset = xyoffset,
            set_past_y_to_0 = set_past_y_to_0,
            use_true_y_in_val = use_true_y_in_val,
            lookback_max = lookback_max,
        )
        if xlookback is None:
            xlookback = 1
        if ylookback is None:
            ylookback = int(self.nqubits - xlookback)
        self.xlookback = xlookback
        self.ylookback = ylookback
        return
    
    def _get_input_t(self, xe_lookback, ye_lookback, t_in, t_pred=None):
        """Get input for this step from lookup tables.

        Arguments:
            xe_lookback (nd.array): lookup table for past actions in this episode. shape(steps_max, dimx).
            ye_lookback (nd.array): lookup table for past outputs in this episode. shape(steps_max, dimy).
            t_pred (int): timestep to be predicted.
            t_in (int): step in loop.

        Returns:
            input_t (np.ndarray): input for this step (1, xlookback+ylookback)
        """
        # x[t_in:t_in+self.xlooback]
        # y[t_in:t_in+self.xlooback]
        xmem = np.flip(xe_lookback[t_in:t_in+self.xlookback], axis=0)
        ymem = np.flip(ye_lookback[t_in:t_in+self.ylookback], axis=0) # e.g. [0:4] for step=0, ylookback=4
        # get input for this step
        input_t = np.vstack([xmem, ymem]).reshape(1, -1) # (xlookback, ylookback)
        return input_t
    

class CPolynomialFeedforward(FeedForwardBase):
    """Classical polynomial fitter for comparisons.

    Different during training and inference (validating).

    Arguments:
        lookback (int): how many past y's are used as input to this stap    
            (e.g. lookback=3: x(t-1), y(t-1), y(t-2), y(t-3), y(t-4))
        poly_degree (int): degree of polynomial features.
            e.g. poly_degree=3, lookback=5: 119 features per timestep
    
    Additional arguments are passed to :class:`.FeedForwardBase`.
    """
    model_name = 'baseline_rnn'

    def __init__(
        self,
        # 
        # FeedForwardBase
        set_past_y_to_0 = False,
        xlookback = 1,
        ylookback = 5,
        # 
        # StepwiseModelBase
        xyoffset = 1,
        use_true_y_in_val = False,
        lookback_max = True,
        # 
        # PredictionModelBase
        washout = 0,
        log = False,
        rseed = 0,
        add_x_as_feature = True,
        # predicting multiple steps forward
        nyfuture = 1, 
        delete_future_y = True,
        # fitter
        fitter = 'sklearn',
        regression_model = 'regression',
        regression_alpha = .1,
        regression_l1 = .5,
        poly_degree = 3,
    ) -> None:
        FeedForwardBase.__init__(
            self,
            # 
            # FeedForwardBase
            set_past_y_to_0 = set_past_y_to_0,
            use_true_y_in_val = use_true_y_in_val,
            xlookback = xlookback,
            ylookback = ylookback,
            xyoffset = xyoffset,
            # 
            # PredictionModelBase
            washout = washout,
            log = log,
            rseed = rseed,
            add_x_as_feature = add_x_as_feature,
            lookback_max = lookback_max,
            # predicting multiple steps forward
            nyfuture = nyfuture, 
            delete_future_y = delete_future_y,
            # fitter
            fitter = fitter,
            regression_model = regression_model,
            regression_alpha = regression_alpha,
            regression_l1 = regression_l1,
            poly_degree = poly_degree,
        )
        self.add_x_as_feature = True # polynomial features includes x
    
    def _set_unitary_and_meas(self):
        """Difference to QRC: no circuit.
        Overwrite method s.t. other QRC methods containing this method can be used
        """
        return
    
    def _t_input_to_t_features(self, input_t, x0, t_pred=None):
        """Features for this steps, given some input.

        Here: polynomial features.

        Args:
            input_t (nd.array): inputs for this step. shape(1, any).
            x0 (nd.array). current x. gets added as feature. shape(1, dimx).
            t_pred (int) = None: optional. in some model used to save the circuit at a specific step.
        """
        return self.poly_feat.fit_transform(input_t) # (1, dimf)
    

class QExtremeLearningMachine(FeedForwardBase, QuantumBase):
    """Feedforward model with quantum circuit as feature extractor.

    Takes x(t-1) (action) and previous ouputs y(t-n-1), ..., y(t-1) as input at each step.
    Performs one step, then measures all qubits.
    The results of the measurements are the features f(t).
    After all training steps, f(t) are fitted to y(t) via an Ordinary Linear System fitter.

    .. math:: 
    
        y_{t+1} = f(x_{t}, ..., x_{t-looback}, y_{t}, ..., y_{t-lookback})
    
    During training the model takes all x(t-1) as given and uses the correct {y(t-i)} from the data.
    During inference (validating) uses the previous predicted outputs {~y(t-i)} and x(t-1) can depend on the previous prediction.
    
    All arguments are passed to :class:`.FeedForwardBase` 
    and :class:`src.basemodel.QuantumBase`.
    """
    model_name = 'qelm'

    def __init__(
        self,
        # 
        # FeedForwardBase
        set_past_y_to_0 = False,
        use_true_y_in_val = False,
        xlookback = None,
        ylookback = None,
        xyoffset = 1,
        # 
        lookback_max = True,
        # PredictionModelBase
        washout = 0,
        log = True,
        rseed = 0,
        add_x_as_feature = True,
        # predicting multiple steps forward
        nyfuture = 1, 
        delete_future_y = True,
        # fitter
        fitter = 'sklearn',
        regression_model = 'regression',
        regression_alpha = .1,
        regression_l1 = .5,
        poly_degree = 3,
        # 
        # QuantumBase
        nqubits = 5,
        qctype = 'ising',
        qinit = 'none',
        nlayers = 1, # unitaries per timestep
        ftype = 0,
        enctype = 'angle',
        nenccopies = 1,
        encangle = 1, # if = 1, encoding is single qubit rotation from 0 to 1*Pi
        encaxes = 1, # axis for measurements
        measaxes = 3, # axis for measurements
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
        QuantumBase.__init__(
            self,
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
            shots = shots,
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
        FeedForwardBase.__init__(
            self,
            # 
            # FeedForwardBase
            set_past_y_to_0 = set_past_y_to_0,
            use_true_y_in_val = use_true_y_in_val,
            xlookback = xlookback,
            ylookback = ylookback,
            xyoffset = xyoffset,
            # 
            # PredictionModelBase
            washout = washout,
            log = log,
            rseed = rseed,
            add_x_as_feature = add_x_as_feature,
            nyfuture = nyfuture, 
            delete_future_y = delete_future_y,
            lookback_max = lookback_max,
            # fitter
            fitter = fitter,
            regression_model = regression_model,
            regression_alpha = regression_alpha,
            regression_l1 = regression_l1,
            poly_degree = poly_degree,
        )
        self.nenccopies = 1
        assert self.xlookback + self.ylookback <= self.nqubits, f'Check: x{self.xlookback}, y{self.ylookback}'
    
    def _set_unitary_and_meas(self) -> None:
        """Define unitary quantum circuit for every timestep.
        Sets self.unistep.
        Does not set input encoding or measurements.
        """
        self.dimxqc = int(((self.dimx * self.ylookback) + (self.dimy * self.xlookback)) * self.nenccopies)
        assert self.dimxqc <= self.nqubits, f'dim{self.dimxqc} nq{self.nqubits} l{self.xlookback},{self.ylookback} encc{self.nenccopies}'
        # dimension of the input data at each step
        self.dmin = np.asarray([[*[self.xmin for _ in range(self.xlookback)], *[self.ymin for _ in range(self.ylookback)]] for _ in range(self.nenccopies)]).reshape(-1) # (dimxqc,)
        self.dmax = np.asarray([[*[self.xmax for _ in range(self.xlookback)], *[self.ymax for _ in range(self.ylookback)]] for _ in range(self.nenccopies)]).reshape(-1) # (dimxqc,)
        self.qin = [*range(self.dimxqc)]
        self.qmeas = [*range(self.nqubits)]
        self.quni = [*range(self.nqubits)]
        if self.unistep is None:
            self._set_unitary()
            # number of gates for one timestep without input encoding (reservoir)
            self.unistep.name = 'U'
            qct = transpile(self.unistep, backend=self.backend)
            self.ngates = weighted_gate_count(qct)
        else:
            print('! Unitary is already defined !')
        return
    
    def _get_step_features(self, angles, saveqc=False):
        """Features for this step, provided the input angles.
        
        Args:
            angles (np.ndarray): angles for the input data (1, dimxqc)

        Return:
            features for this step (1, dimf)
        """
        features_step = []
        for nax, ax in enumerate(self.measaxes):
            qc = QuantumCircuit(self.nqubits, self.nqubits)
            if self.qinit == 'h':
                qc.h(self.quni)
            # input
            self._add_input_to_qc(qc=qc, angles=angles, step=0)
            # unitary
            for _ in range(self.nlayers):
                qc.append(self.unistep, self.quni)
            # measure
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
            qc.measure(qubit=self.qmeas, cbit=[*range(self.nqubits)])
            if saveqc:
                self.qc = qc
            compiled_qc = transpile(qc, self.backend)
            job = self.backend.run(compiled_qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            # post-process QC 
            features_step = self._step_meas_to_step_features(counts, features_step)
        return features_step
    
    def _t_input_to_t_features(self, input_t, x0, t_pred=None):
        """Features for this steps, given some input.

        Encodes input into angles, runs circuit, turns measurements into features (expectation values).

        Args:
            input_t (nd.array): inputs for this step. shape(1, any).
            x0 (nd.array). current x. gets added as feature. shape(1, dimx).
            t_pred (int) = None: optional. Used to save the circuit at the specified step.
        """
        # encode input for quantum circuit
        step_input_angles = self._angle_encoding(input_t, dmin=self.dmin, dmax=self.dmax)
        # run circuit, get features
        features_qc = self._get_step_features(
            angles=step_input_angles, 
            saveqc=True if t_pred == step_to_save_qc else False
        )
        if self.add_x_as_feature:
            return np.hstack(features_qc + [x0]) 
        else:
            return np.hstack(features_qc)