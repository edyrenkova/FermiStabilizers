import openfermion as of
from openfermion.ops import FermionOperator, QubitOperator
from openfermion import jordan_wigner
from hamiltonians.FermionHamiltonian import FermionHamiltonian
from hamiltonians.QubitHamiltonian import QubitHamiltonian
import cirq
import stim
import stimcirq
from .AbstractEncoding import AbstractEncoding
import numpy as np


class JWEncoding(AbstractEncoding):
    """
    Class for Jordan-Wigner encoding of fermion Hamiltonians.
    """

    ###INITIALIZATION###
    def __init__(self, fermion_hamiltonian:FermionHamiltonian):
        """
        Initialize the JWEncoding object.

        Args:
            fermion_hamiltonian (FermionHamiltonian): The fermion Hamiltonian to encode. This already includes the full hamiltonian and the subset of terms.
        """
        self.fermion_ham = fermion_hamiltonian

        self.encoded_subset_operators = [jordan_wigner(op) for op in fermion_hamiltonian.hamiltonian_subset]

        [op.compress() for op in self.encoded_subset_operators]
        self.L = fermion_hamiltonian.L
        self.n_qubits = (fermion_hamiltonian.L) ** 2

    ###ENCODED HAMILTONIAN METHODS###
    def get_relevant_observable_counts(self):
        """
        Get the number of Z observables in the encoded Hamiltonian.

        Returns:
            int: The number of Z observables.
            int: The number of X observables.
        """
        full_fermion_hamiltonian = self.fermion_ham.full_hamiltonian
        full_encoded_ham = jordan_wigner(full_fermion_hamiltonian)
        full_encoded_ham.compress()
        num_Z_terms = np.sum([all([p[1]=='Z' for p in op]) for op in list(full_encoded_ham.terms)])
        num_X_terms = np.sum([all([p[1]=='X' for p in op]) for op in list(full_encoded_ham.terms)])
        
        return num_Z_terms, num_X_terms
    
    def _get_full_encoded_ham_length(self):
        full_fermion_hamiltonian = self.fermion_ham.full_hamiltonian
        full_encoded_ham = jordan_wigner(full_fermion_hamiltonian)
        full_encoded_ham.compress()
        return len(list(full_encoded_ham.terms.items()))
    


    ###LOGICAL ROTATIONS CIRCUIT METHODS###

    def _get_swap_network_full_ham_cirq(self):
        cirq_qubits = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()

        def col(q, L):
            return q % L

        def row(q, L):
            return (q // L) % L
        
        def paulistr_to_cirq(pauli_str):
            if pauli_str is cirq.ops.gate_operation.GateOperation or len(pauli_str) <= 1:
                return cirq.Circuit(cirq.decompose_once(pauli_str))
            ops = cirq.decompose_once(pauli_str ** (3))
            return cirq.Circuit(ops)
        
        def uLuR(qubits):
            n = len(qubits)
            ops = []
            for i in range(1, n - 1, 2):
                ops.append(cirq.CZ(qubits[i], qubits[i + 1]))
                ops.append(cirq.SWAP(qubits[i], qubits[i + 1]))
            for i in range(0, n - 1, 2):
                ops.append(cirq.CZ(qubits[i], qubits[i + 1]))
                ops.append(cirq.SWAP(qubits[i], qubits[i + 1]))
            return ops
        
        def hop_neighbors(qubits):
            ops = []
            for i in range(0, len(qubits) - 1, 2):
                xxpauli = cirq.X(qubits[i]) * cirq.X(qubits[i + 1])
                yypauli = cirq.Y(qubits[i]) * cirq.Y(qubits[i + 1])
                ops.append(paulistr_to_cirq(xxpauli))
                ops.append(paulistr_to_cirq(yypauli))
            
            for i in range(1, len(qubits) - 1, 2):
                xxpauli = cirq.X(qubits[i]) * cirq.X(qubits[i + 1])
                yypauli = cirq.Y(qubits[i]) * cirq.Y(qubits[i + 1])
                ops.append(paulistr_to_cirq(xxpauli))
                ops.append(paulistr_to_cirq(yypauli))
            return ops

        def hop(qubits, L):
            n = len(qubits)
            l = int(n ** (1 / 2))
            ops = []
            for i in range(n - L):
                if row(qubits[i].x, L) % 2 == 0 and col(qubits[i].x, L) == L - 1:
                    #ops.append(cirq.ISWAP(qubits[i], qubits[i + L]))
                    xxpauli = cirq.X(qubits[i]) * cirq.X(qubits[i + L])
                    yypauli = cirq.Y(qubits[i]) * cirq.Y(qubits[i + L])
                    ops.append(paulistr_to_cirq(xxpauli))
                    ops.append(paulistr_to_cirq(yypauli))

                elif row(qubits[i].x, L) % 2 == 1 and col(qubits[i].x, L) == 0:
                    #ops.append(cirq.ISWAP(qubits[i], qubits[i + L]))
                    xxpauli = cirq.X(qubits[i]) * cirq.X(qubits[i + L])
                    yypauli = cirq.Y(qubits[i]) * cirq.Y(qubits[i + L])
                    ops.append(paulistr_to_cirq(xxpauli))
                    ops.append(paulistr_to_cirq(yypauli))
            return ops

        for qubit in cirq_qubits:
            circuit.append(cirq.Z(qubit))

        for i in range(0, self.n_qubits, 2):
            if (i + 1) % self.L != 0:
                zzpauli = cirq.Z(cirq_qubits[i]) * cirq.Z(cirq_qubits[i + 1])
                circuit += paulistr_to_cirq(zzpauli)
        for i in range(1, self.n_qubits - 1, 2):
            if (i + 1) % self.L != 0:
                zzpauli = cirq.Z(cirq_qubits[i]) * cirq.Z(cirq_qubits[i + 1])
                circuit += paulistr_to_cirq(zzpauli)
        for i in range(self.n_qubits - self.L):
            zzpauli = cirq.Z(cirq_qubits[i]) * cirq.Z(cirq_qubits[i + self.L])
            circuit += paulistr_to_cirq(zzpauli)

        circuit += hop_neighbors(cirq_qubits)+(hop(cirq_qubits, self.L) + cirq.Circuit(uLuR(cirq_qubits))) * self.L

        return circuit

    def _get_trotterized_stim_mirror_circuit(self, trotter_steps):
        circuit = cirq.Circuit()
        for _ in range(trotter_steps):
            circuit += self._get_swap_network_full_ham_cirq()
        
        stim_optimized_circuit = self._cirq_to_stim_optimize(circuit)
        stim_mirror_circuit = self._make_mirror_circuit(stim_optimized_circuit)
        return stim_mirror_circuit

    def _get_cirq_circuit(self):
        """
        Get the Cirq circuit representation of the encoded operator (the subset).

        Returns:
            cirq.Circuit: The Cirq circuit representation.
        """
        def paulistr_to_cirq(pauli_str):
            if pauli_str is cirq.ops.gate_operation.GateOperation or len(pauli_str) <= 1:
                return cirq.Circuit(cirq.decompose_once(pauli_str))
            ops = cirq.decompose_once(pauli_str ** (3))
            return cirq.Circuit(ops)

        
        cirq_qubits = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()

        for cirq_q in cirq_qubits:
            circuit += cirq.I(cirq_q)

        ham_paulisum = [of.transforms.qubit_operator_to_pauli_sum(op) for op in self.encoded_subset_operators]
        ham_as_list = []
        for ham_term in ham_paulisum:

            ham_as_list += list(ham_term.__iter__())
        
        for ham_term in ham_as_list:
            mutable_term = cirq.MutablePauliString(ham_term)
            mutable_term.coefficient = 1
            circuit += paulistr_to_cirq(cirq.PauliString(mutable_term))
        return circuit
      
    def _make_mirror_circuit(self, stim_circuit):
        """
        Create a mirror circuit by appending the inverse of the given STIM circuit.

        Args:
            stim_circuit (stim.Circuit): The STIM circuit.

        Returns:
            stim.Circuit: The mirror circuit.
        """
        return stim_circuit + stim_circuit.inverse()

    def _add_noise_stim_circ(self, stim_circuit, p1=0, p2=0, psp=0, pi=0):
        """
        Add noise to the STIM circuit.

        Args:
            stim_circuit (stim.Circuit): The STIM circuit.
            p1 (float): Single-qubit depolarization probability.
            p2 (float): Two-qubit depolarization probability.
            psp (float): State preparation error probability.
            pi (float): Idle error probability.

        Returns:
            stim.Circuit: The modified STIM circuit with added noise.
        """
        noisy_circuit = stim.Circuit()
        elements_to_skip = ['I', 'QUBIT_COORDS']
        used_qubits_in_timestep = set()

        q_num = stim_circuit.num_qubits

        # clean the circuit:
        # get rid of I gates
        # get rid of a TICK at the front
        # get rid of any double TICKS
        stim_circuit_clean = stim.Circuit()

        for i in range(len(stim_circuit) - 1):
            if (stim_circuit[i].name == 'TICK' and stim_circuit[i + 1].name == 'TICK') or \
                    stim_circuit[i].name == 'I':
                continue
            else:
                stim_circuit_clean.append(stim_circuit[i])

        for circ_element in stim_circuit_clean:

            if circ_element.name in elements_to_skip:
                noisy_circuit.append(circ_element)

            elif circ_element.name != 'TICK':

                noisy_circuit.append(circ_element)

                if circ_element.name in self.SINGLE_Q_GATES:

                    if p1:
                        noisy_circuit.append('DEPOLARIZE1', circ_element.targets_copy(), p1)

                    for target in circ_element.targets_copy():
                        used_qubits_in_timestep.add(target.value)

                elif circ_element.name in self.TWO_Q_GATES:

                    if p2:
                        noisy_circuit.append('DEPOLARIZE2', circ_element.targets_copy(), p2)

                    for target in circ_element.targets_copy():
                        used_qubits_in_timestep.add(target.value)

                else:
                    print("UNRECOGNIZED GATE, ", circ_element.name)
                    return "ERROR"

            if circ_element.name == 'TICK':
                if pi:
                    for i in range(q_num):
                        if (not (i in used_qubits_in_timestep)):
                            noisy_circuit.append('DEPOLARIZE1', [i], pi)

                noisy_circuit.append('TICK')
                used_qubits_in_timestep = set()
        
        return noisy_circuit
    
    def _cirq_to_stim_optimize(self, cirq_circuit=None):
        """
        Convert the Cirq circuit to an "optimized" STIM circuit.

        Returns:
            stim.Circuit: The optimized STIM circuit.
        """
        if cirq_circuit is None:
            cirq_circuit = self._get_cirq_circuit()
        cirq_circuit = cirq.drop_empty_moments(cirq_circuit)
        optimized_cirq_circuits = [cirq_circuit,
                                   cirq.stratified_circuit(cirq_circuit),
                                   cirq.align_left(cirq_circuit),
                                   cirq.eject_phased_paulis(cirq_circuit),
                                   cirq.align_left(cirq.stratified_circuit(cirq_circuit)),
                                   cirq.stratified_circuit(cirq.align_left(cirq_circuit)),
                                   cirq.eject_phased_paulis(cirq.stratified_circuit(cirq_circuit)),
                                   cirq.stratified_circuit(cirq.eject_phased_paulis(cirq_circuit)),
                                   cirq.eject_phased_paulis(cirq.align_left(cirq_circuit))]

        return stimcirq.cirq_circuit_to_stim_circuit(
            min(optimized_cirq_circuits, key=lambda circuit: len(circuit))
        )


    ###STATE PREP (BASIS ROTATIONS) AND LOGICAL OBSERVABLES MEASUREMENT CIRCUIT METHODS###

    def _add_hopping_state_prep_measurements(self, stim_circuit, pm=0, p1=0):
        """
        Add best-case hopping operators state preparation and measurements to the STIM circuit.

        Measures all nearest neighbors in the chain.
        
        Args:
            stim_circuit (stim.Circuit): The STIM circuit.
            pm (float): Measurement error probability.
            
            Returns:
                stim.Circuit: The modified STIM circuit with hopping state preparation and measurements.
                int: The number of qubit observables added."""

        observable_count = 0
        stim_circuit_prepend = stim.Circuit()

        for i in range(self.n_qubits):
            stim_circuit_prepend.append("H", [i])
            if p1:
                stim_circuit_prepend.append('DEPOLARIZE1', [i], p1)
            stim_circuit.append("H", [i])
            if p1:
                stim_circuit.append('DEPOLARIZE1', [i], p1)
            stim_circuit.append("MZ", [i], pm)
            if i>0:
                stim_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1), stim.target_rec(-2)], observable_count)
                observable_count+=1
        
        return stim_circuit_prepend + stim_circuit, observable_count
        
    def _add_measurement_stim_circuit(self, stim_circ, pm=0, global_parity_postselection=False):
        """
        Add occupation basis measurement gates to the STIM circuit.

        Args:
            stim_circ (stim.Circuit): The STIM circuit.
            pm (float): Measurement error probability.

        Returns:
            stim.Circuit: The modified STIM circuit with measurement gates.
            int: The number of qubit observables added.
        """

        for i in range(self.n_qubits):
            stim_circ.append("M", [i], pm)
            stim_circ.append("OBSERVABLE_INCLUDE", stim.target_rec(-1), i)
        if global_parity_postselection:
            stim_circ.append('DETECTOR', [stim.target_rec(-i) for i in range(1, self.L**(2)+1)])
        return stim_circ, self.n_qubits
    
    ###FINAL STIM CIRCUIT METHODS###

    def get_stim_circuit(self,
                          stabilizer_reconstruction: bool,
                          non_destructive_stabilizer_measurement_end: bool,
                          virtual_error_detection_rate:float, #vqed_rate should be 0 if virtual error detection is not used
                          
                          flags_in_synd_extraction: bool,
                          type_of_logical_observables: str,
                          global_parity_postselection: bool,
                          efficient_trotter_steps: int, #efficient_trotter_steps should be 0 if efficient trotterization is not used, doesn't count mirror circuits
                          p1: float, p2: float, psp: float, pi: float, pm: float):
        """
        Get the STIM circuit representation of the encoded operator.

        Args:
            
        Returns:
            stim.Circuit: The STIM circuit representation.
            int: The number of qubit observables added.
        """

        # This is added for clarity, but the flags are not used in the JWEncoding class
        stabilizer_reconstruction = False
        virtual_error_detection_rate = 0
        flags_in_synd_extraction = False
        non_destructive_stabilizer_measurement_end = False

        if efficient_trotter_steps == 0:
            noisy_logical_circuit = self._add_noise_stim_circ(self._make_mirror_circuit(self._cirq_to_stim_optimize()), p1, p2, psp, pi)
        else:
            noisy_logical_circuit = self._add_noise_stim_circ(self._get_trotterized_stim_mirror_circuit(efficient_trotter_steps), p1, p2, psp, pi)

        if type_of_logical_observables == "Z":
            noisy_stim, obs_number = self._add_measurement_stim_circuit(noisy_logical_circuit, pm, global_parity_postselection)
        elif type_of_logical_observables == "efficient":
            noisy_stim, obs_number = self._add_hopping_state_prep_measurements(noisy_logical_circuit, pm, p1)

        psp_prepend = stim.Circuit()
        for i in range(noisy_stim.num_qubits):
            psp_prepend.append("X_ERROR", [i], psp)

        return psp_prepend+noisy_stim, obs_number, 0, 0, 0, 0, 0, 0 #only relevant numbers of Compact encoding: destr_meas_count, flag_meas_counter, stab_meas_counter, vqed_measure_counter