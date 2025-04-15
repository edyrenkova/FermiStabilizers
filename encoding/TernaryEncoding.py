import openfermion as of
from openfermion.ops import FermionOperator, QubitOperator
from hamiltonians.FermionHamiltonian import FermionHamiltonian
from hamiltonians.QubitHamiltonian import QubitHamiltonian
import cirq
import stim
import stimcirq
from .AbstractEncoding import AbstractEncoding
import sys

from openfermion.transforms import get_majorana_operator

from .g_matrix import pruned_sierpinski_G_matrix

from inv_maps_master.EncodingClasses import *
import os
import pickle

#works but hopping term measurement is not implemented and the occupation basis is questionable.

class TernaryEncoding(AbstractEncoding):

    ###INITIALIZATION###
    def __init__(self, fermion_hamiltonian:FermionHamiltonian = None):
        """
        Initializes a TernaryEncoding object.

        Args:
            fermion_hamiltonian (FermionHamiltonian, optional): The fermion Hamiltonian to be encoded. Defaults to None.
        """
        self.L = int(fermion_hamiltonian.L)

        self.n_qubits = self.L**2

        self.fermion_ham = fermion_hamiltonian

        self.encoded_subset_operators = [self._encode_ham(op) for op in self.fermion_ham.hamiltonian_subset]

        self.cirq_circuit = None

        self.cirq_circuit = self._get_cirq_circuit()

    ###ENCODED HAMILTONIAN METHODS###
    def _get_relevant_observable_counts(self):
        """
        Get the number of Z observables in the encoded Hamiltonian.

        Returns:
            int: The number of Z observables.
            int: The number of X observables (efficient).
        """
        full_fermion_hamiltonian = self.fermion_ham.full_hamiltonian
        full_encoded_ham = self._encode_ham(full_fermion_hamiltonian)
        full_encoded_ham.compress()
        num_Z_terms = np.sum([all([p[1]=='Z' for p in op]) for op in list(full_encoded_ham.terms)])
        
        return num_Z_terms, None
    
    def _get_full_encoded_ham_length(self):

        full_fermion_hamiltonian = self.fermion_ham.full_hamiltonian
        full_encoded_ham = self._encode_ham(full_fermion_hamiltonian)
        full_encoded_ham.compress()
        return len(list(full_encoded_ham.terms.items()))
    
    def _encode_ham(self, fermion_operator = None):

        if fermion_operator is None:
            fermion_operator = self.fermion_operator
        if fermion_operator.terms == {}:
            return QubitOperator()
        N = self.n_qubits
        
        G_pruned = pruned_sierpinski_G_matrix(N)

        # # Define the file path
        # file_path = f'pruned_sierpinski_trees/G_pruned_N_{N}.pkl'

        # # Check if the file exists
        # if os.path.exists(file_path):
        #     # Load the object from the file
        #     with open(file_path, 'rb') as file:
        #         G_pruned = pickle.load(file)
        # else:
        #     # Generate the object and save it to the file
        #     G_pruned = pruned_sierpinski_G_matrix(N)
        #     with open(file_path, 'wb') as file:
        #         pickle.dump(G_pruned, file)
    
        H = self._make_majorana_hamiltonian_for_ternary(N, fermion_operator)
        Hamiltonian_2 = Encoding_with_Hamiltonians(N, G_pruned, Fermionic_H = H)
        spin_H = Spin_Hamiltonian(N, Hamiltonian_2.Spin_H)
        written_H = spin_H.Written_H(Hamiltonian_2.Spin_H)
        qubit_operator = self._convert_to_qubit_operator(written_H)
        return qubit_operator

    def _make_majorana_hamiltonian_for_ternary(self, N, hamiltonian:FermionOperator):
        
        hubbard_majorana_op = get_majorana_operator(hamiltonian)
        input_dict = hubbard_majorana_op.terms

        max_index = max(max(key) for key in input_dict.keys() if key)
        output_list = []
        values_list = []
        majorana_ham_list = []

        for key in input_dict.keys():
            # Create a list of 0s with length 2 * (max_index + 1)
            presence_list = [0] * ((max_index + 1))

            # Mark the presence of each index in the tuple
            for idx in key:
                presence_list[idx] = 1

            # Reorder the list to have even indices first and then odd indices
            reordered_list = [presence_list[i] for i in range(0, len(presence_list), 2)] + \
                            [presence_list[i] for i in range(1, len(presence_list), 2)]

            # Add the reordered list to the output list
            output_list.append(reordered_list)

            values_list.append(input_dict[key])

        reordered_values_list = [values_list[i] for i in range(0, len(values_list), 2)] + \
                                [values_list[i] for i in range(1, len(values_list), 2)]

        majorana_ham_list.append(output_list)
        majorana_ham_list.append(reordered_values_list)

        H = Fermionic_Hamiltonian(N, majorana_ham_list)
        return H
    
    def _convert_to_qubit_operator(self, hamiltonian):
        # Initialize an empty QubitOperator
        qubit_operator = QubitOperator()

        # Extract the list of Pauli strings and coefficients
        pauli_strings = hamiltonian[0]
        coefficients = hamiltonian[1]

        # Iterate over the Pauli strings and coefficients
        for pauli_string, coefficient in zip(pauli_strings, coefficients):
            # Initialize an empty term
            term = ''
            # Iterate over the Pauli string to create the correct format
            for i, pauli in enumerate(pauli_string):
                if pauli != 'I':  # Skip the identity operators
                    term += f'{pauli}{i} '
            term = term.strip()  # Remove any trailing spaces
            
            # Create a term in the QubitOperator for the current Pauli string and coefficient
            qubit_term = QubitOperator(term, coefficient)
            
            # Add the term to the QubitOperator
            qubit_operator += qubit_term

        return qubit_operator
    
    ###LOGICAL ROTATIONS CIRCUIT METHODS###
    def _cirq_to_stim_optimize(self, cirq_circuit):
        """
        Convert the Cirq circuit to an "optimized" STIM circuit.

        Returns:
            stim.Circuit: The optimized STIM circuit.
        """
        #cirq_circuit = self._get_cirq_circuit()
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
    
    def _make_mirror_circuit(self, stim_circuit):
        """
        Create a mirror circuit by appending the inverse of the given STIM circuit.

        Args:
            stim_circuit (stim.Circuit): The STIM circuit.

        Returns:
            stim.Circuit: The mirror circuit.
        """
        return stim_circuit + stim_circuit.inverse()

    def _get_full_trotter_step_cirq_circuit(self, trotter_steps):

        full_fermion_hamiltonian = self.fermion_ham.full_hamiltonian
        full_encoded_ham = self._encode_ham(full_fermion_hamiltonian)
        full_encoded_ham.compress()
        def paulistr_to_cirq(pauli_str):
            if pauli_str is cirq.ops.gate_operation.GateOperation or len(pauli_str) <= 1:
                return cirq.Circuit(cirq.decompose_once(pauli_str))
            ops = cirq.decompose_once(pauli_str ** (3))
            return cirq.Circuit(ops)

        cirq_qubits = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()

        for cirq_q in cirq_qubits:
            circuit += cirq.I(cirq_q)

        ham_paulisum = [of.transforms.qubit_operator_to_pauli_sum(QubitOperator(op)) for op in list(full_encoded_ham.terms)]
        ham_as_list = []
        for ham_term in ham_paulisum:

            ham_as_list += list(ham_term.__iter__())
        
        for _ in range(trotter_steps):
            for ham_term in ham_as_list:
                mutable_term = cirq.MutablePauliString(ham_term)
                mutable_term.coefficient = 1
                circuit += paulistr_to_cirq(cirq.PauliString(mutable_term))
         
        return circuit


    def _get_cirq_circuit(self):
        """
        Get the Cirq circuit representation of the encoded operator.

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

        # deleting extra TICK
        #stim_circuit_clean = stim_circuit_clean[1:]

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
                        if (i not in used_qubits_in_timestep):
                            noisy_circuit.append('DEPOLARIZE1', i, pi)

                noisy_circuit.append('TICK')
                used_qubits_in_timestep = set()
        
        return noisy_circuit

    ###STATE PREP (BASIS ROTATIONS) AND LOGICAL OBSERVABLES MEASUREMENT CIRCUIT METHODS###

    def _add_measurement_stim_circuit(self, stim_circ, pm=0):
        """
        Add occupation basis measurement gates to the STIM circuit.

        Args:
            stim_circ (stim.Circuit): The STIM circuit.
            pm (float): Measurement error probability.

        Returns:
            stim.Circuit: The modified STIM circuit with measurement gates.
        """
        for i in range(self.n_qubits):
            stim_circ.append("M", [i], pm)
            stim_circ.append("OBSERVABLE_INCLUDE", stim.target_rec(-1), i)
        return stim_circ, self.n_qubits

    
    ###FINAL STIM CIRCUIT METHODS###

    def get_stim_circuit(self,
                          stabilizer_reconstruction: bool,
                          non_destructive_stabilizer_measurement_end: bool,
                          virtual_error_detection_rate:float, #vqed_rate should be 0 if virtual error detection is not used
                          flags_in_synd_extraction: bool,
                          type_of_logical_observables: str,
                          global_parity_postselection: bool,
                          efficient_trotter_steps: int, # 0 if not efficient trotter setting
                          p1: float, p2: float, psp: float, pi: float, pm: float):
        """
        Get the STIM circuit representation of the encoded operator.

        Args:
            
            stabilizer_reconstruction (bool): Whether to perform stabilizer reconstruction.
            non_destructive_stabilizer_measurement_end (bool): Whether to perform non-destructive stabilizer measurement at the end.
            virtual_error_detection_rate (float): The rate of virtual error detection.
            flags_in_synd_extraction (bool): Whether to use flags in syndrome extraction.
            type_of_logical_observables (str): The type of logical observables to add.
            global_parity_postselection (bool): Whether to use global parity postselection.
            p1 (float): Single-qubit depolarization probability.
            p2 (float): Two-qubit depolarization probability.
            psp (float): State preparation error probability.
            pi (float): Idle error probability.
            pm (float): Measurement error probability.


        Returns:
            stim.Circuit: The STIM circuit representation.
            int: The number of qubit observables added.
        """

        # This is added for clarity, but the flags are not used in the Ternary class
        
        stabilizer_reconstruction = False
        non_destructive_stabilizer_measurement_end = False
        virtual_error_detection_rate = 0
        flags_in_synd_extraction = False
        global_parity_postselection = False #don't know how to do in ternary case, the states don't get mapped as easily
        
        if efficient_trotter_steps == 0:
            optimized_circuit = self._cirq_to_stim_optimize(self._get_cirq_circuit())
            inverse_circuit = optimized_circuit.inverse()
            noisy_logical_circuit = self._add_noise_stim_circ(optimized_circuit+inverse_circuit, p1, p2, psp, pi)
        else:
            optimized_circuit =  self._cirq_to_stim_optimize(self._get_full_trotter_step_cirq_circuit(efficient_trotter_steps))
            inverse_circuit = optimized_circuit.inverse()
            noisy_logical_circuit = self._add_noise_stim_circ(optimized_circuit+inverse_circuit, p1, p2, psp, pi)
    
        if type_of_logical_observables == "Z":
            noisy_stim, obs_number = self._add_measurement_stim_circuit(noisy_logical_circuit, pm)
        elif type_of_logical_observables == "efficient":
            raise NotImplementedError("Efficient logical observables not implemented for TernaryEncoding")

        psp_prepend = stim.Circuit()
        for i in range(noisy_stim.num_qubits):
            psp_prepend.append("X_ERROR", [i], psp)
        
        return psp_prepend+noisy_stim, obs_number, 0, 0, 0, 0, 0, 0 #only relevant numbers of Compact encoding: destr_meas_count, flag_meas_counter, stab_meas_counter, vqed_measure_counter
    
            
    
    

    