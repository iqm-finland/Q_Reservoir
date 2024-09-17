from qiskit_aer import Aer
from qiskit import execute, visualization
from qiskit.extensions import UnitaryGate
import random as system_random
import scipy.linalg as spl
import numpy as np
from src.utils import *

## generate haar single qubit unitaries
def CUE(random_gen,nh):
    """generate Haar random single qubit unitary matrix.

     Args:
         random_gen: random generator object.
         nh (int): size of the matrix.

     Returns:
         U (array): unitary matrix.

     """

    U = (random_gen.randn(nh,nh)+1j*random_gen.randn(nh,nh))/np.sqrt(2)
    q,r = spl.qr(U)
    d = np.diagonal(r)
    ph = d/np.absolute(d)
    U = np.multiply(q,ph,q)
    return U

def get_X(prob, NN):
    """Evalutes the purity function X for each applied random unitary..

     Args:
         prob (array): probability vector.
         NN (int): number of qubits.

     Returns:
         XX_e (int): the value of the function evaluated from the measurement data.
     """
    alphabet = "abcdefghijklmnopqsrtuvwxyz"
    alphabet_cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    Hamming_matrix = np.array([[1,-0.5],[-0.5,1]]) ## Hamming matrix for a single qubit
    ein_command = alphabet[0:NN]
    for ii in range(NN):
        ein_command += ','
        ein_command += alphabet[ii]+alphabet_cap[ii]
    ein_command += ','+ alphabet_cap[0:NN]
    Liste = [prob] + [Hamming_matrix]*NN + [prob]
    XX_e = np.einsum(ein_command, *Liste, optimize = True)*2**NN
    return XX_e
    
def unbias(X,NN,NM):
    """Unbias the purity function X for each applied random unitary..

     Args:
         X (float): value of the function.
         NN (int): number of qubits.
         NM (int): number of measurements.

     Returns:
         coupling_map (list): list of couplings within the qubits.
     """
    return X*NM**2/(NM*(NM-1)) - 2**NN/(NM-1)

def einsum_contraction_fidelity(nqubits):
    """Defines the contraction path to post-process data efficiently.

     Args:
         nqubits (int): number of qubits.

     Returns:
         contraction_path (str): the einsum contraction paths.
     """
    alphabet = "abcdefghijklmnopqsrtuvwxyz"
    alphabet_cap = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    ein_command = alphabet[0:nqubits]
    temppp = ''
    for ii in range(nqubits):
        ein_command += ','
        ein_command += alphabet[ii]+alphabet_cap[ii]
        temppp += alphabet[ii]+alphabet_cap[ii]
    ein_command += ','+ alphabet_cap[0:nqubits]
    
    return ein_command
    
def ObtainOutcomeProbabilities(NN,qstate,u,p):

    if qstate.shape == (2 ** NN,):
        return ObtainOutcomeProbabilities_pseudopure(NN, qstate, u, p)
    else:
        return ObtainOutcomeProbabilities_mixed(NN,qstate,u,p)

def ObtainOutcomeProbabilities_pseudopure(NN, psi, u, p):
    psi = np.reshape(psi, [2] * NN)
    for n in range(NN):
        psi = np.einsum(u[n], [NN, n], psi, list(range(NN)), list(range(n)) + [NN] + list(range(n + 1, NN)))
    probb = np.abs(np.ravel(psi))**2*(1-p) + p/2**NN ## makes the probabilities noisy by adding white noise
    probb /= sum(probb)
    return probb

def ObtainOutcomeProbabilities_mixed(NN, rho, u,p):
        prob_tensor = rho.reshape(tuple([2] * (2*NN)),order='C')
        for n in range(NN):
            prob_tensor = np.einsum(u[n], [2*NN, n], prob_tensor, list(range(NN))+list(range(n+NN,2*NN)), np.conjugate(u[n]), [2*NN,NN+n], list(range(n)) + [2*NN] + list(range(n + 1, NN)) + list(range(NN+n+1,2*NN)))
        probb= np.real(prob_tensor.reshape(2**NN))
        probb = (1-p)*probb + p/2**NN ## makes the probabilities noisy by adding white noise
        probb /= sum(probb)
        return probb

def get_RM_circuits(nqubits, Nu, algo):
    """Defines the circuits to be executed for randomized measurements.

     Args:
         nqubits (int): number of qubits.
         Nu (int): number random unitaries.
         algo (string): the algorithm in use.

     Returns:
         unitaries (array): array of unitaries used for the measurement.
         qclist (list): list of Quantum Circuits to be executed.
         qstates (Statevector): The pure statevector of the algorithm.

     """
    unitaries = np.zeros((Nu,nqubits,2,2),dtype=np.complex_)
    qc = get_circuit(algo, nqubits)
    backend = Aer.get_backend('statevector_simulator')
    qstate = execute(qc, backend).result().get_statevector(qc)
    
    a = system_random.SystemRandom().randrange(2 ** 32 - 1) #Init Random Generator
    random_gen = np.random.RandomState(a)
    qclist = []
    for iu in range(Nu):
        #qc = QuantumCircuit(num_qubits,num_qubits)
        
        qc.barrier()
        for z in range(0,nqubits):
            temp_U=CUE(random_gen,2)
            qc.append(UnitaryGate(temp_U),[z])
            unitaries[iu,z]=np.array(temp_U)
            
        qc.measure_all()
        qclist.append(qc)
        qc = get_circuit(algo, nqubits)
        
        
    return unitaries, qclist, qstate

def get_properties(Nu, Nm, nqubits, job_counts, qstate, unitaries):
    """Evaluates the fidelity <psi|rho|psi> from randomized measurements .

     Args:
         Nu (int): number of unitaries used.
         Nm (int): number of measurements.
         nqubits (int): number of qubits.
         job_counts (list): list of containing a dictionary of counts for each applied random unitary.
         qstate (Statevector): The pure statevector of the algorithm.
         unitaries (array): array of unitaries used for the measurement.


     Returns:
         fidelity (float): the fidelity and its uncertainty of the experimental state wrt to the pure state.
         purity (float): the purity of the experimental state.
     """
    F = []
    F_CRM = []
    p2 = []
    p2_CRM = []
    F_CRM_final = 0
    p2_CRM_final = 0
    get_bin = lambda x, n: format(x, 'b').zfill(n)
    #sorted_list = {get_bin(i, nqubits): 0 for i in range(2 ** nqubits)}
    Hamming_matrix = np.array([[1,-0.5],[-0.5,1]]) ## Hamming matrix for a single qubit
    contract_command = einsum_contraction_fidelity(nqubits)
    for iu in range(Nu):
        print('Postprocessing Fidelity {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
        bit_strings = {get_bin(i, nqubits): 0 for i in range(2 ** nqubits)}
        probb = np.zeros(2**nqubits, dtype = float)
        sorted_bit_strings = dict(sorted(job_counts[iu].items()))
        bit_strings.update(sorted_bit_strings)
        probb = np.array(list(bit_strings.values()))/Nm
        probb = np.reshape(probb, [2]*nqubits)
        probb_QM = ObtainOutcomeProbabilities(nqubits, qstate, unitaries[iu,:] , 0)
        probb_QM = np.reshape(probb_QM, [2]*nqubits)
        p2 += [unbias(get_X(probb, nqubits), nqubits, Nm)/Nu]
        p2_CRM += [get_X(probb_QM, nqubits)/Nu - 1/Nu]
        
        Listee = [probb] + [Hamming_matrix]*nqubits + [probb_QM]
        #Listee_R = [probb] + Hamming_matrix_cal + [probb_QM]
        F += [np.einsum(contract_command,*Listee, optimize = 'greedy')*2**nqubits/Nu]
        #F_R += np.einsum(contract_command,*Listee_R, optimize = 'greedy')*2**N/Nu
        Listee = [probb_QM] + [Hamming_matrix]*nqubits + [probb_QM]
        F_CRM  += [np.einsum(contract_command, *Listee, optimize='greedy')*2**nqubits/Nu  - 1/Nu]
        #F_CRM_R  -= np.einsum(contract_command, *Listee, optimize='greedy')*2**N/Nu  - 1/Nu
    
    F_CRM_final = [F[x]-F_CRM[x] for x in range(Nu)]
    p2_CRM_final = [p2[x] - p2_CRM[x] for x in range(Nu)]
    #F_CRM_R += F_R
    return [np.sum(F_CRM_final), np.std(F_CRM_final),  np.sum(p2_CRM_final),  np.std(p2_CRM_final)] #{'fidelity': np.sum(F_CRM_final), 'error': np.std(F_CRM_final), 'purity': np.sum(p2_CRM_final), 'error': np.std(p2_CRM_final)}

def get_purity(Nu, Nm, nqubits, job_counts):
    """Evaluates the purity of the prepared state rho (tr(rho^2)) from randomized measurements .

     Args:
         Nu (int): number of unitaries used.
         Nm (int): number of measurements.
         nqubits (int): number of qubits.
         job_counts (list): list of containing a dictionary of counts for each applied random unitary.

     Returns:
         p2 (float): the purity.
     """
    p2 = []
    for iu in range(Nu):
        print('Postprocessing Purity {:d} % \r'.format(int(100*iu/(Nu))),end = "",flush=True)
        bit_strings = job_counts[iu]
        probb = np.zeros(2**nqubits, dtype = float)
        for inm, ist in enumerate(bit_strings.keys()):
            probb[int(ist[::-1],2)] = bit_strings[ist]/Nm
        probb = np.reshape(probb, [2]*nqubits)
        p2 += [unbias(get_X(probb, nqubits),nqubits, Nm)]
    return np.sum(p2), np.std(p2)
