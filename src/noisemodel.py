# -*- coding: utf-8 -*-
"""Provides the :class:`NoiseModel` class for representing a simple noise model based on T1, T2 thermal relaxation times.
"""
import numpy as np
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

def thermal_model(nqubits: int, t1_mean: float = 50, t2_mean: float = 70, verbose=False):
    """Thermal Relaxation example from Qiskit.
    
    https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html#Example-2:-T1/T2-thermal-relaxation
    
    T1 and T2 values for qubits sampled from normal distribution with mean 
    50 and 70 microsec.
    
    Args:
        t1_mean (float): 50 microseconds
        t2_mean (float): 70 microseconds
    
    Returns:
        qiskit_aer.noise.NoiseModel
    """
    rng = np.random.Generator(np.random.PCG64(seed=0))

    t1_mean = t1_mean * 1e3 # convert to nanoseconds
    t2_mean = t2_mean * 1e3 # convert to nanoseconds

    # T1 and T2 values for qubits 
    # Sampled from normal distribution mean 50 microsec
    T1s = np.abs(rng.normal(t1_mean, 10e3, nqubits).clip(min=1e-10))
    T2s = np.abs(rng.normal(t2_mean, 10e3, nqubits).clip(min=1e-10))
    if verbose:
        print(f'T1 mean {t1_mean:2e}')
        print(T1s)
        print(f'T2 mean {t2_mean:2e}')
        print(T2s)
    # a[a < 0] = 1e-10

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(nqubits)])

    # Instruction times (in nanoseconds)
    time_u1 = 0   # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100 # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000 # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                    for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
                for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
                for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
                for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                thermal_relaxation_error(t1b, t2b, time_cx))
                for t1a, t2a in zip(T1s, T2s)]
                for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    noise_thermal = NoiseModel()
    for j in range(nqubits):
        noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
        noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
        noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
        noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
        noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
        for k in range(nqubits):
            noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])
    
    return noise_thermal
