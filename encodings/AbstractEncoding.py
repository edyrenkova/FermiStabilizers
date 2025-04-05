from abc import ABC, abstractmethod
import cirq
import openfermion as of
import stim

from hamiltonians.FermionHamiltonian import FermionHamiltonian
from hamiltonians.QubitHamiltonian import QubitHamiltonian

class AbstractEncoding(ABC):
    @abstractmethod
    def __init__(self, fermion_hamiltonian: FermionHamiltonian):
        self.L = fermion_hamiltonian.L
        self.fermion_operator = fermion_hamiltonian.get_fermion_operator()

   
    SINGLE_Q_GATES = ['X', 'Y', 'Z', 'C_XYZ', 'C_ZYX', 'H', 'H_XY',
                      'H_XZ', 'H_YZ', 'S', 'S_DAG', 'SQRT_X', 'SQRT_X_DAG',
                      'SQRT_Y', 'SQRT_Y_DAG', 'SQRT_Z', 'SQRT_Z_DAG']

    TWO_Q_GATES = ['CNOT', 'CX', 'CXSWAP', 'CY', 'CZ', 'CZSWAP', 'ISWAP',
                   'ISWAP_DAG', 'SQRT_XX', 'SQRT_XX_DAG', 'SQRT_YY', 'SQRT_YY_DAG',
                   'SQRT_ZZ', 'SQRT_ZZ_DAG', 'SWAP', 'SWAPCX', 'SWAPCZ',
                   'XCX', 'XCY', 'XCZ', 'YCX', 'YCY', 'YCZ', 'ZCX',
                   'ZCY', 'ZCZ']

    @abstractmethod
    def get_stim_circuit(self, stabilizer_postselection: bool,
                         stabilizer_reconstruction: bool,
                          virtual_error_detection:bool, virtual_error_detection_rate:float,
                          flags_in_synd_extraction: bool,
                          type_of_logical_observables: str,
                          noiseless_inverse: bool,
                          p1: float, p2: float, psp: float, pi: float, pm: float):
        """
        Get the STIM circuit representation of the encoded operator.
        """
        pass

    @abstractmethod
    def _get_cirq_circuit(self):
        """
        Get the Cirq circuit representation of the encoded operator.
        """
        pass

    @abstractmethod
    def get_full_encoded_ham_length(self):
        """
        Get the length of the full Hamiltonian.
        """
        pass

    