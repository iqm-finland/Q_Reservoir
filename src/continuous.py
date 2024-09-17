# -*- coding: utf-8 -*-
"""Contains :class:`.QContinuousRC` and :class:`.NaivePredictor`.

:class:`.QContinuousRC` is a reservoir computer which constructs one circuit spanning all timesteps.
Thus all inputs need to be known beforehand, and predictions of previous timesteps cannot be fed in again.
This is sometimes called online reservoir computing.

It is not suitable for the Reactor data, or Narma.
In theory it can do the short term memory task, but in practice it does not work well.
It works a little when using ancilla qubits for weak measurement, but not well enough to be useful.
I think the best way to make an online reservoir work is to use weak measurement simulations, 
see https://www.nature.com/articles/s41534-023-00682-z

:class:`.NaivePredictor` is a simple predictor which does not use a quantum circuit.
It can be used as a comparison, altough a better benchmark model is :class:`src.feedforward.CPolynomialFeedforward`.
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
from numpy.random import Generator, PCG64
import pathlib
import sys
import os

sys.path.insert(1, os.path.abspath('..')) # module: qrc_surrogate

from src.basemodel import PredictionModelBase, QuantumBase
from src.helpers import weighted_gate_count, unnormalize

module_path = pathlib.Path(__file__).parent.parent.resolve().__str__()

class QContinuousRC(PredictionModelBase, QuantumBase):
    """Continous (online) reservoir.

    Takes all actions x(t) as given and constructs one circuit spanning all timesteps t.
    Inputs current x(t) and partially measures the circuit at each step t.
    The results of the measurements are the features f(t).
    After all training steps, f(t) are fitted to y(t) via an Ordinary Linear System fitter. 

    Qubits which aren't measured act as a kind of memory of past steps. 
    
    Only takes in x (not y). Can theoretically do Narma, but not Reactor. 
    Does not work well in practice.

    .. math:: 
        y_t = f(x_t, ..., x_0)

    Arguments:
        washout (int) = 5: 
            number of first steps of episode ignored by fitting/error metrics.
        preloading (int) = 0: 
            number of times the first step is repeated before episode actually starts
        qctype (str) = 'ising':
            random, random_hpcnot, random_ht, random_clifford, circularcnot, 
            hcnot, spread_input, efficientsu2, downup, lstmv1, lstmv2, random_esn, empty
        mtype (str) = 'projection':
            | projection:
            | weak: weak measurement using ancilla qubits. See :func:`circuits.weak_measurement_circuit`.
            |   weak_angle (float) = .1: weak measurement angle. Gets multiplied by Pi/2. 
            |   =1 is a cnot from observed to ancilla qubit and basically a strong measurement.
        nmeas (int):
            number of qubits measured at each step.
        minc (bool) = True:
            When selecting qubits to measure, prioritise qubits which encode x
        mend (bool) = True:
            When selecting qubits to measure, prioritise qubits on the far side of the encoding qubits
        resetm (bool) = False:
            Reset qubits after measurement
        reseti (bool) = True:
            Reset qubits before input
    
    Additional arguments are passed to :class:`src.basemodel.PredictionModelBase` and :class:`src.basemodel.QuantumBase`.
    """
    model_name = 'online_reservoir'
    mtypes = ['projection', 'weak']
    naxs = [1, 2, 3]
    axis = ['z', 'x', 'y']

    def __init__(
        self, 
        # This class
        preloading = 0, 
        mtype = 'projection',
        minc = True, 
        mend = True,
        nmeas = 2, 
        reseti = True, 
        resetm = False, 
        weak_angle = .1, 
        # --
        # PredictionModelBase
        washout = 5,
        rseed = 0,
        log = True,
        add_x_as_feature = True,
        # predicting multiple steps forward
        nyfuture = 1, 
        delete_future_y = True,
        # fitter
        fitter = 'sklearn',
        regression_model = 'ridge',
        regression_alpha = 0.1,
        regression_l1 = 0.1,
        poly_degree = 3,
        # --
        # QuantumBase
        nqubits = 5,
        qctype = 'ising',
        qinit = 'none',
        nlayers = 1, 
        ftype = 0,
        enctype = 'angle',
        encaxes = 'x',
        measaxes = 'z',
        nenccopies = 1,
        encangle = 1, 
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
        PredictionModelBase.__init__(
            self,
            log = log,
            washout = washout,
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
        QuantumBase.__init__(
            self,
            nqubits = nqubits,
            qctype = qctype,
            qinit = qinit,
            nlayers = nlayers,
            ftype = ftype,
            enctype = enctype,
            encaxes = encaxes,
            measaxes = measaxes,
            nenccopies = nenccopies,
            encangle = encangle,
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
        # Circuit
        self.mtype = mtype
        self.minc = minc
        self.mend = mend
        self.nmeas = nmeas 
        self.resetm = resetm
        self.reseti = reseti
        self.weak_angle = weak_angle
        # Hyperparameters
        self.preloading = preloading
        # self.washout_eff = max(0, washout-preloading) # washout might not be necessary when doing preloading
        # --
        self._set_fitter()
    
    def _build_episode_circuit(self, unistep, angles, measax='z', steps=None, barriers=False):
        """Circuit for one episode.

        Arguments:
            unistep (QuantumCircuit): unitary for one timestep
            angles (list[np.array]): list of angles which encode input of this episode (step, dimx)
            measax (str): x,y,z. axis of measurement.
            steps (int): number of timesteps in this episode. Defaults to number of rows in angles.
            barriers (bool): add barriers between steps to make circuit illustrations.
        """
        if steps is None:
            steps = np.shape(angles)[0]
        assert unistep is not None
        # build
        qc = QuantumCircuit(self.nqubits, self.nmeas*steps)
        for step in range(steps):
            # add input 
            if self.reseti:
                qc.reset([q for q in self.qin if q in self.qmeas])
            self._add_input_to_qc(qc=qc, angles=angles, step=step)
            # random unitary (reservoir)
            if barriers: qc.barrier()
            for _ in range(self.nlayers):
                qc.append(unistep, qargs=[*range(unistep.num_qubits)])
            if barriers: qc.barrier()
            # measure
            cbits = [*range(self.nmeas*step, self.nmeas*(step+1))]
            # axis of measurement
            match measax:
                # https://arxiv.org/abs/1804.03719
                case 'z':
                    pass
                case 'x':
                    qc.h(self.qmeas)
                case 'y':
                    qc.sdg(self.qmeas)
                    qc.h(self.qmeas)
                case _:
                    raise Warning(f'Invalid measax: {measax}')
            qc.measure(self.qmeas, cbits)
            if self.resetm:
                qc.reset(self.qmeas) # better or worse?
            if barriers: qc.barrier()
        self.qc = qc
        return self.qc

    def _set_unitary_and_meas(self):
        """Define quantum circuit at every timestep.

        Returns:
            self.unistep (qiskit.QuantumCircuit): unitary at every timestep.
        """
        self.dimx_wo_copies = self.dimx
        self.dimxqc = int(self.dimx * self.nenccopies)
        self.qin = [*range(self.dimxqc)]
        if self.unistep is None:
            match self.mtype:
                case 'projection':
                    self.quni = [*range(self.nqubits)]
                    # input qubits: first few qubits of the circuit
                    self.qmeas = []
                    if self.minc:
                        # measure input qubits 
                        self.qmeas = self.qin[:self.nmeas]
                    if self.mend:
                        # measure the rest from the other end of the circuit
                        self.qmeas += [*range(self.nqubits - self.nmeas + len(self.qmeas), self.nqubits)]
                    else:
                        # measure the rest starting after input qubits
                        self.qmeas += [*range(len(self.qin), len(self.qin) + self.nmeas - len(self.qmeas))]
                    if len(self.qmeas) > self.nmeas:
                        # split measured qubits evenly between beginning and end of circuit
                        self.qmeas = self.qmeas[:int(self.nmeas/2)]
                        self.qmeas += [*range(self.nqubits - self.nmeas + len(self.qmeas), self.nqubits)]
                case 'weak':
                    # number of weak measurements (input qubits are already hard measured)
                    nwm = self.nmeas - self.dimxqc
                    self.nqubits += nwm
                    # qin + qmeas + qanc <= qubits
                    # qubits which are weakly measured through ancillas
                    qwm = [*range(self.dimxqc, self.dimxqc + nwm)]
                    # ancillas to measure
                    qanc = [*range(self.nqubits - nwm, self.nqubits)] 
                    self.qmeas = self.qin + qanc
                    assert len(self.qmeas) == self.nmeas
                    # qubits to apply unitary to
                    self.quni = [*range(self.nqubits - nwm)]
                case _:
                    raise Warning('Invalid meas_type')
            # unitary for every timestep
            self._set_unitary()
            # add weak masurement
            if self.mtype == 'weak':
                reserv = QuantumCircuit(self.nqubits)
                # reset ancillas
                reserv.reset(qanc)
                # random unitary
                reserv.append(self.unistep, self.quni)
                # weak measurement
                for wm, a in zip(qwm, qanc):
                    reserv.h(a)
                    reserv.ry(np.pi * self.weak_angle/2, a) # weak measurement angle
                    reserv.cx(wm, a)
                self.unistep = reserv
            # format circuit
            self.unistep.name = 'U'
            # number of gates for one timestep without input encoding (reservoir)
            qct = transpile(self.unistep)
            self.ngates = weighted_gate_count(qct)
        else:
            print('! Unitary is already defined !')
        return self.unistep
    
    def illustrate_circuit(self, dimx=1, steps=3, mockqc=False):
        """Schematic of circuit for first few timesteps."""
        # save previous state
        qc_saved = self.qc
        qctype_saved = self.qctype
        if mockqc:
            self.qctype = 'mockqc'
        self.dimx = dimx if self.dimx is None else self.dimx
        # build schematic
        rng = Generator(PCG64(seed=self.rseed))
        x_encoded = rng.random(size=(steps, self.dimx)) * np.pi
        print('illustrating circuit:', self.qctype)
        self._set_unitary_and_meas()
        unistep = self.unistep
        qc = self._build_episode_circuit(unistep = unistep, angles=x_encoded, steps=steps, barriers=True if not mockqc else False)
        qc_illustrated = self.qc.copy()
        if not mockqc:
            qc_illustrated = transpile(qc_illustrated, self.backend)
        # return to previous state
        self.qc = qc_saved
        self.qctype = qctype_saved
        return qc_illustrated
    
    def _get_all_episode_features(self, x):
        """Build, run, and get output from quantum circuit reservoir for all episodes.

        Can use preloading. Returns features with washout period but without preloading steps.

        Arguments:
            x:  list[np.array]: list of episodes (episodes, steps, dimx).

        Returns:
            features: list[np.array]:  list of means of measurements (episodes, steps, dimf)
        """
        # get features
        features = []
        for xe in x: # loop over episode
            # preloading
            # repeat first step self.preloading+1 times, the other steps once
            # xe = np.repeat(
            #     a=xe, 
            #     repeats=[self.preloading+1] + [1]*(np.shape(xe)[0]-1), 
            #     axis=0
            # )
            # get angles for circuit
            if self.enctype == 'angle':
                angles_e = self._angle_encoding(xe)
            elif self.enctype == 'ryrz':
                angles_e = xe
            else:
                raise Warning(f'Invalid enctype {self.enctype}')
            steps = np.shape(angles_e)[0]
            # run quantum circuits
            features_episode = [] # features for this episode. list of arrays. every array is one ax.
            for nax, ax in enumerate(self.measaxes):
                self.qc = self._build_episode_circuit(self.unistep, angles_e, measax=ax)
                counts = self._get_circuit_measurements(qc=self.qc) # (n_shots, steps*nmeas)
                # turn measurements around, since qiskit returns them in reverse order
                counts = {k[::-1]:v for k,v in counts.items()}
                # turn measurements into features
                features_all_steps_this_ax = []
                for step in range(steps):
                    counts_step = {}
                    for b, c in counts.items():
                        b_step = b[step * self.nmeas:(step + 1) * self.nmeas]
                        if b_step in counts_step.keys():
                            counts_step[b_step] += c
                        else:
                            counts_step[b_step] = c
                    # features this step. every array is one feature type. 
                    # len() 1,2, or 3 depending on ftype
                    features_step = self._step_meas_to_step_features(counts_step=counts_step, features_step=[])
                    features_all_steps_this_ax.append(np.hstack(features_step))
                # all steps done
                features_episode.append(np.vstack(features_all_steps_this_ax))
            # all ax done
            # (steps, nmeas*nax + dimx)
            if self.add_x_as_feature:
                features.append(np.hstack(features_episode + [xe]))
            else:
                features.append(np.hstack(features_episode))
        # all episodes done
        # (episodes, steps, dimf)
        self.dimf = np.shape(features[0])[1]
        return features
    
    def train(self):
        """Train the model on the training data for all episodes.

        Run circuit for all steps and collect features.
        If circuit takes in y as input, then use true y.
        After all steps and episodes, train linear fitter.

        Uses:
            xtrain: list of episodes. list(np.array) (episodes, steps, dimx)
            ytrain: list of episodes. list(np.array) (episodes, steps, dimy)

        Returns:
            list of predictions (episodes, steps, dimy)
            list of features (reservoir circuit outputs) (episodes, steps, dimf)
        """
        # run the reservoir for all episodes
        self.ftrain = self._get_all_episode_features(x=self.data.xtrain)
        # fit normalized x to normalized y
        self._fit(features=self.ftrain, ys=self.data.ytrain)
        self.ytrain = [self._predict(out=out) for out in self.ftrain]
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
        return self.ytrain, self.ftrain
    
    def val(self):
        """Same as :func:`.train()` but weights are fixed."""
        self.fval = self._get_all_episode_features(x=self.data.xval)
        self.yval = [self._predict(out=out) for out in self.fval]
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
        return self.yval, self.fval
    


class NaivePredictor(PredictionModelBase):
    """Simple prediction to compare the Reservoir + Linear System (OLS) Solver to.
    
    NaivePredictor does not look at input sequence (x), 
    but makes simple guesses based on previous target (y) steps.

    Arguments:
        pred_type (str):
            | naive: 
            |    Predict that sequence won't change. 
            |    :math:`y_t = y_{t-1}`
            | mean: 
            |    Predict that sequence is the mean. 
            |    Technically cheating since it uses y(t) of future timesteps.
            |    :math:`y_t = 1/N \sum_{i=0}^{N-1} y_{i}`
            | prevmean:
            |    Predict that sequence is the mean of previous timesteps.
            |    :math:`y_t = 1/N \sum_{i=0}^{t} y_{i}`

    Additional arguments are passed to :class:`src.basemodel.PredictionModelBase`.
    """
    model_name = 'baseline_prediction'

    def __init__( 
            self,
            # Parent
            rseed = 0,
            # This class
            predtype = 'naive',     
        ) -> None:
        super().__init__(
            rseed = rseed,
        )
        self.predtype = predtype
    
    def __str__(self):
        return f'''NaivePredictor.'''
    
    def _set_unitary_and_meas(self):
        """NaivePredictor does not use a quantum circuit."""
        return
    
    def _predict(self, y):
        """Predicts y from y (one episode).
        Args:
            y: np.array (steps, dimy)
        Returns:
            ypred: np.array (steps, dimy)
        """
        match self.predtype:
            case 'naive':
                ypred = np.roll(y, 1)
            case 'mean':
                ypred = np.ones(np.shape(y)) * np.mean(y, axis=0)
            case 'prevmean':
                ypred = y
                for s in range(self.len_seq):
                    ypred[s] = np.mean(y[:s], axis=0)
                # prediction = np.array([np.mean(target[:t], axis=0) for t in range(self.len_seq)])
            case _:
                raise Warning('Invalid pred_type')
        return ypred
    
    def train(self, y=None):
        """Train is the same as val (class has no trainable parameters).
        Args: 
            y: list of episodes
        Returns:
            list of predictions (list) (episodes, steps, dimy)
        """
        predtrain = [self._predict(yep) for yep in y]
        return predtrain
    
    def val(self, y=None):
        """Train is the same as val (class has no trainable parameters).
        Args: 
            y: list(np.array) (episodes, steps, dimy)
        Returns:
            predictions: list(np.array) (episodes, steps, dimy)
        """
        predtrain = [self._predict(yep) for yep in y]
        return predtrain
