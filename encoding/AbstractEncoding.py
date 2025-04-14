from abc import ABC, abstractmethod
import cirq
import openfermion as of
import stim

from hamiltonians.FermionHamiltonian import FermionHamiltonian
from hamiltonians.QubitHamiltonian import QubitHamiltonian

class AbstractEncoding(ABC):
    """
    Abstract class for encoding fermionic Hamiltonians into qubit Hamiltonians.

    This class provides the basic structure for implementing encodings for the purpose of creating STIM circuit simulations.

    Attributes:
        L (int): The size of the square fermionic LxL lattice.
        fermion_operator (FermionOperator): The fermionic operator to be encoded.
    """

    SINGLE_Q_GATES = ['X', 'Y', 'Z', 'C_XYZ', 'C_ZYX', 'H', 'H_XY',
                      'H_XZ', 'H_YZ', 'S', 'S_DAG', 'SQRT_X', 'SQRT_X_DAG',
                      'SQRT_Y', 'SQRT_Y_DAG', 'SQRT_Z', 'SQRT_Z_DAG']

    TWO_Q_GATES = ['CNOT', 'CX', 'CXSWAP', 'CY', 'CZ', 'CZSWAP', 'ISWAP',
                   'ISWAP_DAG', 'SQRT_XX', 'SQRT_XX_DAG', 'SQRT_YY', 'SQRT_YY_DAG',
                   'SQRT_ZZ', 'SQRT_ZZ_DAG', 'SWAP', 'SWAPCX', 'SWAPCZ',
                   'XCX', 'XCY', 'XCZ', 'YCX', 'YCY', 'YCZ', 'ZCX',
                   'ZCY', 'ZCZ']

    @abstractmethod
    def __init__(self, fermion_hamiltonian: FermionHamiltonian):
        """
        Initialize the AbstractEncoding object.

        Args:
            fermion_hamiltonian (FermionHamiltonian): The fermionic Hamiltonian to be encoded.
        """
        self.L = fermion_hamiltonian.L
        self.fermion_operator = fermion_hamiltonian.get_fermion_operator()

    @abstractmethod
    def get_stim_circuit(stabilizer_reconstruction: bool,
                          non_destructive_stabilizer_measurement_end: bool,
                          virtual_error_detection_rate:float, 
                          flags_in_synd_extraction: bool,
                          type_of_logical_observables: str,
                          global_parity_postselection: bool,
                          efficient_trotter_steps: int, #0 if not the efficient trotter type circuits
                          p1: float, p2: float, psp: float, pi: float, pm: float):
        """
        Get the STIM circuit representation of the encoded operator.

        Args:
            stabilizer_reconstruction (bool): Whether to use stabilizer reconstruction.
            non_destructive_stabilizer_measurement_end (bool): Whether to perform non-destructive stabilizer measurement at the end.
            virtual_error_detection_rate (float): The rate of virtual quantum error detection (VQED) (0 if not used, -1 for symmetry expansion).
            flags_in_synd_extraction (bool): Whether to include flags in syndrome extraction.
            type_of_logical_observables (str): The type of logical observables to use.
            global_parity_postselection (bool): Whether to use global parity postselection.
            efficient_trotter_steps (int): Number of Trotter steps (0 if not using Trotter circuits).
            p1 (float): Single qubit gate error rate.
            p2 (float): Two qubit gate error rate.
            psp (float): State preparation error rate.
            pi (float): Idling error rate.
            pm (float): Measurement error rate.
        """
        pass

    @abstractmethod
    def _get_cirq_circuit(self):
        """
        Get the Cirq circuit of the encoded Hamiltonian.

        Returns:
            cirq.Circuit: The Cirq circuit.
        """
        pass

    @abstractmethod
    def _get_full_encoded_ham_length(self):
        """
        Get the length of the full encoded Hamiltonian (number of terms).

        Returns:
            int: The length of the full Hamiltonian.
        """
        pass