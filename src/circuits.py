# -*- coding: utf-8 -*-
"""Quantum Circuits to be used as fixed unitaries in the quantum reservoir computing.

The only one that is working in practice is :func:`.ising_circuit` at the moment.
Most of the other circuits were inspired by VQC Ansaetze.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
import numpy as np
from numpy.random import Generator, PCG64
import itertools


def ising_circuit(nqubits, t = 1, jmax = 1, h = .1, wmax = 10, rseed=0, mode=None, 
                  random=True, jpositive=False, wpositive=False):
    r"""Transverse field ising model.
    
    .. math:: 
        
        H = J X X + (h + D) Z \\

        U = e^{-iHt} ~ e^{-iJXXt} e^{-i(h+W)Zt} = Rxx(2Jt) Rz(2(h+W)t)

    https://arxiv.org/pdf/2103.05348.pdf

    Arguments:
        nqubits (int): number of qubits
        t (float): time in e^-iHt
        h (float): fixed magnetic field
        jmax (float): spin-spin coupling J in (-jmax, jmax)
        jpositive (bool): J in (0, jmax)
        wmax (float): onsite disorder strength W in (-wmax, wmax)
        wpositive (bool): D in (0, jmax)
        random (bool): if false, J=jmax and W=wmax
        mode (str): 
            | 'nn': Rxx on nearest neighbour, and Rz.
            | 'up_ladder': Rxx ladder from qubit 0 to qubit max, and Rz.
            | 'up_down_ladder': Rxx ladder from qubit 0 to qubit max, Rz, Rxx ladder from qubit max to qubit 0.
            | else: Rxx on all pairwise connections, and Rz.
        rseed (int): random seed for picking J, W
    
    Returns:
        QuantumCircuit
    """
    qc =  QuantumCircuit(nqubits)
    if random:
        if jpositive:
            jlow, jhigh = 0, jmax
        else:
            jlow, jhigh = -jmax, jmax
        if wpositive:
            wlow, whigh = 0, wmax
        else:
            wlow, whigh = -wmax, wmax
    else:
        jlow, jhigh = jmax, jmax
        wlow, whigh = wmax, wmax
    #
    rng = Generator(PCG64(seed=rseed))
    if mode == 'nn':
        for j in range(0, nqubits-1, 2):   
            qc.rxx(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
        for j in range(1, nqubits-1, 2):
            qc.rxx(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
        for j in range(0, nqubits):   
            qc.rz(phi = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
    elif mode in ['up_ladder', 'ladder']:
        for j in range(0, nqubits-1):
            qc.rxx(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
            qc.rz(phi = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
    elif mode == 'up_down_ladder':
        for j in range(0, nqubits-1):
            qc.rxx(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
            qc.rz(phi = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
        for j in range(nqubits-1, 0, -1):
            qc.rxx(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j-1)
    else:
        for j in range(nqubits):
            for i in range(j+1, nqubits):
                qc.rxx(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=i, qubit2=j)
            qc.rz(phi = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
    return qc

def ising_circuit_natural(nqubits, t = 1, jmax = 1, h = .1, wmax = 10, rseed=0, mode=None, 
                  random=True, jpositive=False, wpositive=False):
    r"""Transverse field ising model, adjusted to fit our natural 2-qubit gate.
    'Also has two possible ``adjustment methods``, to allow prevent the ZZ acting trivially.
    
    
    .. math:: 
        
        H = J Z Z + (h + D) X \\

        U = e^{-iHt} ~ e^{-iJZZt}e^{-i(h+W)Xt}  = RX(2(h+W)t)RZZ(2Jt) 

    https://arxiv.org/pdf/2103.05348.pdf

    Arguments:
        nqubits (int): number of qubits
        t (float): time in e^-iHt
        h (float): fixed magnetic field
        jmax (float): spin-spin coupling J in (-jmax, jmax)
        jpositive (bool): J in (0, jmax)
        wmax (float): onsite disorder strength W in (-wmax, wmax)
        wpositive (bool): D in (0, jmax)
        random (bool): if false, J=jmax and W=wmax
        mode (str): 
            | 'nn': Rxx on nearest neighbour, and Rz.
            | 'up_ladder': Rxx ladder from qubit 0 to qubit max, and Rz.
            | 'up_down_ladder': Rxx ladder from qubit 0 to qubit max, Rz, Rxx ladder from qubit max to qubit 0.
            | else: Rxx on all pairwise connections, and Rz.
        rseed (int): random seed for picking J, W
        adjustment (str):
            | 'Hadamard' : Performs a Hadamard on each gate at the start.
            | 'Swap Order': Performs 1 qubit then two qubit gates.
            | else -> Hadamard.
    
    Returns:
        QuantumCircuit
    """
    qc =  QuantumCircuit(nqubits)
    if random:
        if jpositive:
            jlow, jhigh = 0, jmax
        else:
            jlow, jhigh = -jmax, jmax
        if wpositive:
            wlow, whigh = 0, wmax
        else:
            wlow, whigh = -wmax, wmax
    else:
        jlow, jhigh = jmax, jmax
        wlow, whigh = wmax, wmax
    #
    rng = Generator(PCG64(seed=rseed))
    if mode == 'nn':
        for j in range(0, nqubits):   
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
        for j in range(0, nqubits-1, 2):   
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
        for j in range(1, nqubits-1, 2):
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)

    elif mode in ['up_ladder', 'ladder']:
        for j in range(0, nqubits):
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
        for j in range(0, nqubits-1):
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
            
    elif mode == 'up_down_ladder':
        for j in range(0, nqubits):
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
        for j in range(0,nqubits-1):
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
        for j in range(0, nqubits):
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
        for j in range(nqubits-1, 0, -1):
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j-1)
    else:
        for j in range(nqubits):
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
        for j in range(nqubits):
            for i in range(j+1, nqubits):
                qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=i, qubit2=j)
            
    return qc

def ising_circuit_naturalH(nqubits, t = 1, jmax = 1, h = .1, wmax = 10, rseed=0, mode=None, 
                  random=True, jpositive=False, wpositive=False):
    r"""Transverse field ising model. Each unitary is sandwiched with a round of Hadamards, to avoid
    trivial actions.
    
    .. math::
        
        H = J Z Z + (h + D) X \\

        U = e^{-iHt} ~ e^{-iJZZt} e^{-i(h+W)Xt} = Rzz(2Jt) Rx(2(h+W)t)

    https://arxiv.org/pdf/2103.05348.pdf

    Arguments:
        nqubits (int): number of qubits
        t (float): time in e^-iHt
        h (float): fixed magnetic field
        jmax (float): spin-spin coupling J in (-jmax, jmax)
        jpositive (bool): J in (0, jmax)
        wmax (float): onsite disorder strength W in (-wmax, wmax)
        wpositive (bool): D in (0, jmax)
        random (bool): if false, J=jmax and W=wmax
        mode (str): 
            | 'nn': Rzz on nearest neighbour, and Rz.
            | 'up_ladder': Rzz ladder from qubit 0 to qubit max, and Rz.
            | 'up_down_ladder': Rzz ladder from qubit 0 to qubit max, Rx, Rzz ladder from qubit max to qubit 0.
            | else: Rzz on all pairwise connections, and Rz.
        rseed (int): random seed for picking J, W
    
    Returns:
        QuantumCircuit
    """
    qc =  QuantumCircuit(nqubits)
    if random:
        if jpositive:
            jlow, jhigh = 0, jmax
        else:
            jlow, jhigh = -jmax, jmax
        if wpositive:
            wlow, whigh = 0, wmax
        else:
            wlow, whigh = -wmax, wmax
    else:
        jlow, jhigh = jmax, jmax
        wlow, whigh = wmax, wmax
    #
    rng = Generator(PCG64(seed=rseed))
    for j in range(0, nqubits):
        qc.h(qubit=j)

    if mode == 'nn':
        for j in range(0, nqubits-1, 2):   
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
        for j in range(1, nqubits-1, 2):
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
        for j in range(0, nqubits):   
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
    if mode == 'nnc': #closed loop.
        for j in range(0, nqubits-1, 2):   
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
        for j in range(1, nqubits-1, 2):
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
        qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=0, qubit2=nqubits-1)
        for j in range(0, nqubits):   
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
    elif mode in ['up_ladder', 'ladder']:
        for j in range(0, nqubits-1):
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
    elif mode == 'up_down_ladder':
        for j in range(0, nqubits-1):
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j+1)
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
        for j in range(nqubits-1, 0, -1):
            qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=j, qubit2=j-1)
    else:
        for j in range(nqubits):
            for i in range(j+1, nqubits):
                qc.rzz(theta = rng.uniform(low=jlow, high=jhigh) * t, qubit1=i, qubit2=j)
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)

    for j in range(0, nqubits):
        qc.h(qubit=j)
    return qc

def native_circuit(nqubits, t = 1, jmax = 1, h = .1, wmax = 10, rseed=0, mode=None, 
                  random=True, jpositive=False, wpositive=False):
    qc =  QuantumCircuit(nqubits)
    if random:
        if jpositive:
            jlow, jhigh = 0, jmax
        else:
            jlow, jhigh = -jmax, jmax
        if wpositive:
            wlow, whigh = 0, wmax
        else:
            wlow, whigh = -wmax, wmax
    else:
        
        jlow, jhigh = jmax, jmax
        wlow, whigh = wmax, wmax
    #
    rng = Generator(PCG64(seed=rseed))
    for j in range(0, nqubits):
        qc.h(qubit=j)

    if mode == 'nn':
        for j in range(0, nqubits-1, 2):   
            qc.cz( control_qubit=j, target_qubit=j+1)
        for j in range(1, nqubits-1, 2):
            qc.cz( control_qubit=j, target_qubit=j+1)
        for j in range(0, nqubits):   
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
    if mode == 'nnc': #closed loop.
        #print(wlow,whigh)
        for j in range(0, nqubits-1, 2):   
            qc.cz( control_qubit=j, target_qubit=j+1)
        for j in range(1, nqubits-1, 2):
            qc.cz( control_qubit=j, target_qubit=j+1)
        qc.cz(control_qubit=nqubits-1, target_qubit=0)
        for j in range(0, nqubits):
            thetac=t * (h + rng.uniform(low=wlow, high=whigh))
            #print(thetac)   
            qc.rx(theta = thetac, qubit=j)
        
    elif mode in ['up_ladder', 'ladder']:
        for j in range(0, nqubits-1):
            qc.cz( control_qubit=j, target_qubit=j+1)
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
    elif mode == 'up_down_ladder':
        for j in range(0, nqubits-1):
            qc.cz( control_qubit=j, target_qubit=j+1)
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)
        for j in range(nqubits-1, 0, -1):
            qc.cz( control_qubit=j, target_qubit=j-1)
    else:
        for j in range(nqubits):
            for i in range(j+1, nqubits):
                qc.cz( control_qubit=i, target_qubit=j)
            qc.rx(theta = t * (h + rng.uniform(low=wlow, high=whigh)), qubit=j)

    for j in range(0, nqubits):
        qc.h(qubit=j)
    return qc


def jiri_circuit():  
    """For 3 qubits.
    Just somewhat arbitrarily generated circuit from universal gates (p, h, cx).
    By Jirka Guth Jarkovsky.

    Returns:
        3-qubit QuantumCircuit
    """
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.p(1.754,0)
    qc.cnot(0,1)
    qc.h(2)
    qc.p(1.154,2)
    qc.p(2.211,1)
    qc.cnot(1,2)
    qc.h(1)
    qc.p(0.754,0)
    qc.p(1.154,1)
    qc.cnot(1,0)
    qc.p(0.884,2)
    return qc

def jiri_circuit_weak_measurement():
    """3 qubit unitary with weak measurements on qubit 0 and 1 using two extra ancillas.
    
    | Unitary is the same as in :func:`.jiri_circuit`
    | Weak measurement is the same as :func:`.weak_measurement_circuit`

    By Jirka Guth Jarkovsky.

    Returns:
        5-qubit QuantumCircuit
    """
    qc = QuantumCircuit(5)
    # reset ancillas
    qc.reset(3)
    qc.reset(4)
    # circuit fixed
    qc.h(0)
    qc.p(1.754,0)
    qc.cnot(0,1)
    qc.h(2)
    qc.p(1.154,2)
    qc.p(2.211,1)
    qc.cnot(1,2)
    qc.h(1)
    qc.p(0.754,0)
    qc.p(1.154,1)
    qc.cnot(1,0)
    qc.p(0.884,2)
    # Weak measurement
    qc.h([3])               # 1. apply H to get the ancilla in a state |+>
    qc.ry(-0.25,3)           # 2. Slightly rotate along the y axis to get a state sqrt(1/2+e)|0> + sqrt(1/2-e)|1>
    qc.cx(1,3)              # 3. CNOT to change into sqrt(1/2+e)|1> + sqrt(1/2-e)|0>
    # qc.measure(3, self.dim_outqc*step+1)  # 4. Measure. More likely to get 0 if control qubit was in state |0> and vice versa
    qc.h([4])
    qc.ry(-0.25,4)
    qc.cx(2,4)
    # qc.measure(4, self.dim_outqc*step+2)
    # qc.measure(0, self.dim_outqc*step)
    return qc

def weak_measurement_circuit(qwm, unitary, wmangle = -0.25):
    r"""Quantum circuit with weak measurement on qubits qwm using ancillas.

    | Weak measurement:
    | 1. apply H to get the ancilla in a state :math:`|+>`
    | 2. Slightly rotate along the y axis (Ry) to get a state :math:`\sqrt{1/2+e}|0> + sqrt{1/2-e}|1>`
    | 3. CNOT to change into :math:`\sqrt{1/2+e}|1> + \sqrt{1/2-e}|0>`

    By Jirka Guth Jarkovsky.

    Arguments:
        qwm (list): qubits to be weakly measured
        unitary (QuantumCircuit): unitary to be applied to be (partly) weakly measured
        wmangle (float): angle for Ry in the weak measurement. 
            =Pi/2 is basically a strong measurement, 
            since the ancilla will be in :math:`|1>` before the cnot and perfectly 
            correlated with the observed qubit.

    Returns:
        QuantumCircuit with qwm + unitary qubits 
    
    Usage:

    .. code-block:: python

        qc = weak_measurement_circuit(qwm, unitary) 
        qanc = [*range(unitary.num_qubits, qc.num_qubits)]
        qc.measure(qanc)

    """
    nqubits = len(qwm) + unitary.num_qubits
    quni = [*range(unitary.num_qubits)] # qubits to apply the unitary to
    qanc = [*range(unitary.num_qubits, nqubits)]
    qc = QuantumCircuit(nqubits)
    # reset ancillas
    qc.reset(qanc)
    qc.append(unitary, quni)
    # weak measurement
    for wm, a in zip(qwm, qanc):
        qc.h(a)
        qc.ry(wmangle, a)
        qc.cx(wm, a)
    return qc

def random_circuit(nqubits, ngates, rseed=0):
    """h, rx, ry, rz, p, cnot.
    
    Arguments:
        nqubits (int): number of qubits
        ngates (int): number of gates
        rseed (int): random seed
    
    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(nqubits)
    rng = Generator(PCG64(seed=rseed))
    gates = rng.integers(low=0, high=7, size=int(ngates))
    for gate in gates:
        qubit = rng.integers(nqubits)
        if gate == 0:
            qc.h(qubit)
        elif gate == 1:
            angle = rng.random() * np.pi * 2
            qc.rx(theta = angle, qubit=qubit)
        elif gate == 2:
            angle = rng.random() * np.pi * 2
            qc.ry(theta = angle, qubit=qubit)
        elif gate == 3:
            angle = rng.random() * np.pi
            qc.rz(phi = angle, qubit=qubit)
        elif gate == 4:
            angle = rng.random() * np.pi
            qc.p(theta = angle, qubit=qubit)
        else:
            q_wo_target = [*range(nqubits)]
            q_wo_target.remove(qubit)
            rng.shuffle(q_wo_target)
            qc.cnot(control_qubit=q_wo_target[0], target_qubit=qubit)
    return qc

def random_hpcnot_circuit(nqubits, ngates, rseed=0):
    """h, p, cnot.
    
    Arguments:
        nqubits (int): number of qubits
        ngates (int): number of gates
        rseed (int): random seed
        
    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(nqubits)
    rng = Generator(PCG64(seed=rseed))
    gates = rng.integers(low=0, high=3, size=int(ngates))
    for gate in gates:
        qubit = rng.integers(nqubits)
        if gate == 0:
            qc.h(qubit)
        elif gate == 1:
            angle = rng.random() * np.pi
            qc.p(theta = angle, qubit=qubit)
        elif gate == 2:
            q_wo_target = [*range(nqubits)]
            q_wo_target.remove(qubit)
            rng.shuffle(q_wo_target)
            qc.cnot(control_qubit=q_wo_target[0], target_qubit=qubit)
    return qc

def random_clifford_circuit(nqubits, ngates, rseed=0):
    """h, s, t, cnot.

    Arguments:
        nqubits (int): number of qubits
        ngates (int): number of gates
        rseed (int): random seed
        
    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(nqubits)
    rng = Generator(PCG64(seed=rseed))
    gates = rng.integers(low=0, high=5, size=int(ngates))
    for gate in gates:
        qubit = rng.integers(nqubits)
        if gate == 0:
            qc.h(qubit)
        elif gate == 1:
            angle = rng.random() * np.pi * 2
            qc.s(qubit=qubit)
        elif gate == 2:
            qc.t(qubit=qubit)
        else:
            q_wo_target = [*range(nqubits)]
            q_wo_target.remove(qubit)
            rng.shuffle(q_wo_target)
            qc.cnot(control_qubit=q_wo_target[0], target_qubit=qubit)
    return qc

def random_ht_circuit(nqubits, ngates, rseed=0):
    """h, tiffoli.

    Arguments:
        nqubits (int): number of qubits
        ngates (int): number of gates
        rseed (int): random seed
        
    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(nqubits)
    rng = Generator(PCG64(seed=rseed))
    gates = rng.integers(low=0, high=2, size=int(ngates))
    for gate in gates:
        qubit = rng.integers(nqubits)
        if gate == 0:
            qc.h(qubit)
        else:
            q_wo_target = [*range(nqubits)]
            q_wo_target.remove(qubit)
            rng.shuffle(q_wo_target)
            qc.toffoli(control_qubit1=q_wo_target[0], control_qubit2=q_wo_target[1], target_qubit=qubit)
    return qc

def hcnot_circuit(nqubits, rseed=0):
    """One of h, ry, rz on every qubit. cnot on every pair of qubits (fully connected).
    
    Arguments:
        nqubits (int): number of qubits
        rseed (int): random seed
        
    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(nqubits)
    rng = Generator(PCG64(seed=rseed))
    for q in range(nqubits):
        qc.h(q)
        qc.ry(theta = rng.random() * np.pi * 2, qubit=q)
        qc.rz(phi = rng.random() * np.pi, qubit=q)
    pairwise_combinations = list(itertools.combinations([*range(nqubits)], 2))
    # if nqubits > 7:
    #     raise Warning(f'{nqubits} means {2**nqubits}={len(pairwise_combinations)} pairwise combinations = cnots')
    for c, t in pairwise_combinations:
        qc.cx(control_qubit=c, target_qubit=t)
    return qc

def circularcnot_circuit(nqubits, rseed=0):
    """Nearest-neighbor cnots staircase + last qubit connecting back to first qubit.

    Inspired by:
    Studying quantum algorithms for particle track reconstruction in the LUXE experiment

    https://www.researchgate.net/figure/Layout-of-the-variational-quantum-circuit-using-the-TwoLocal-ansatz-with-R-Y-gates-and-a_fig1_358604035

    Arguments:
        nqubits (int): number of qubits
        rseed (int): random seed

    Returns:
        QuantumCircuit
    """
    rng = Generator(PCG64(seed=rseed))
    qc = QuantumCircuit(nqubits)
    for q in range(nqubits):
        qc.ry(theta = rng.random() * np.pi * 2, qubit=q)
        qc.rz(phi = rng.random() * np.pi, qubit=q)
        if q == nqubits-1:
            # last to first
            qc.cx(control_qubit=q, target_qubit=0)
        else:
            # first to second, second to third, ...
            qc.cx(control_qubit=q, target_qubit=q+1)
    return qc

def spread_circuit(nqubits, sources, rseed=0):
    """Input connected to all others via a direct cnot"""
    rng = Generator(PCG64(seed=rseed))
    qc = QuantumCircuit(nqubits)
    for q in range(nqubits):
        qc.ry(theta = rng.random() * np.pi * 2, qubit=q)
        qc.rz(phi = rng.random() * np.pi, qubit=q)
    for s in sources:
        for q in range(nqubits):
            if q not in sources:
                qc.cx(control_qubit=s, target_qubit=q)
    return qc

def efficientsu2_circuit(nqubits, rseed=0):
    """EfficientSU2 circuit from qiskit with random parameters
    https://qiskit.org/documentation/stubs/qiskit.circuit.library.EfficientSU2.html

    Arguments:
        nqubits (int): number of qubits
        rseed (int): random seed

    Returns:
        QuantumCircuit
    """
    rng = Generator(PCG64(seed=rseed))
    qc = EfficientSU2(nqubits)
    mappin = {p: rng.random() * np.pi for p in qc.ordered_parameters}
    qc = qc.assign_parameters(mappin)
    return qc

def downup_circuit(nqubits, qin=0, rseed=0):
    """H on qin, cnot staircase down, random rz on all qubits, cnot staircase up.

    This is a circuit that is used in the qiskit tutorial:
    https://qiskit.org/documentation/tutorials/circuits_advanced/01_advanced_circuits.html

    Arguments:
        nqubits (int): number of qubits
        qin (int | list): index of qubit to apply H to
        rseed (int): random seed

    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(nqubits)
    rng = Generator(PCG64(seed=rseed))
    qc.h(qin)
    # down
    for q in range(nqubits-1):
        qc.cx(q, q+1)
    for q in [*range(nqubits)]:
        qc.rz(phi = rng.random() * np.pi, qubit=q)
    # up
    for q in reversed(range(nqubits-1)):
        qc.cx(q, q+1)
    qc.h(qin)
    return qc

def fake_circuit(nqubits):
    """Empty circuit with label for drawing.
    
    Args:
        nqubits (int)

    Returns:
        QuantumCircuit
    """
    qc = QuantumCircuit(nqubits)
    qc.unitary(obj=np.eye(2**nqubits), qubits=[*range(nqubits)], label='U')
    return qc

