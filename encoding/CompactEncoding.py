
from .FermionSquare import FermionSquare
import panqec
from hamiltonians.FermionHamiltonian import FermionHamiltonian
import cirq
import copy
import stim
import stimcirq
import numpy as np
from .AbstractEncoding import AbstractEncoding
from openfermion.ops import FermionOperator, QubitOperator
import openfermion as of
from hamiltonians.QubitHamiltonian import QubitHamiltonian


class CompactEncoding(AbstractEncoding):
    """
    Represents a Compact Encoding of a fermion Hamiltonian.

    Attributes:
        L (int): The size of the lattice.
        panqec_code (FermionSquare): The panqec code used for encoding.
        n_qubits (int): The number of qubits in the encoding.
        fermion_ham (FermionHamiltonian): The fermion Hamiltonian to be encoded.
        encoded_subset_operators (list): The encoded subset operators.
        full_encoded_ham (QubitOperator): The full encoded Hamiltonian.

    """
   

    ###INITIALIZATION###
    def __init__(self, fermion_hamiltonian:FermionHamiltonian):
        """
        Initializes a CompactEncoding object.

        Args:
            fermion_hamiltonian (FermionHamiltonian): The fermion Hamiltonian to be encoded
        """
        self.L = int(fermion_hamiltonian.L)
        self.panqec_code = FermionSquare(self.L)
        self.n_qubits = len(self.panqec_code.qubit_index.keys())
        self.fermion_ham = fermion_hamiltonian
        compact_ham_dicts = [self._encode_compact_ham_terms(op) for op in fermion_hamiltonian.hamiltonian_subset]
        self.encoded_subset_operators = [self._get_qubit_operator(op) for op in compact_ham_dicts]
        self.full_encoded_ham = self._encode_ham(fermion_hamiltonian.full_hamiltonian)
        self.full_encoded_ham.compress()
        self._measure_jk_dict = {}
        self._stab_indexes_in_order_of_meas = []
    
    ###ENCODED HAMILTONIAN METHODS###

    def _get_relevant_observable_counts(self):
        """
        Get the number of Z observables in the encoded Hamiltonian.

        Returns:
            int: The number of Z observables.
            int: The number of X observables (to keep consistent with JW but the hopping observables get counted later).
        """

        full_encoded_ham = self.full_encoded_ham
        num_Z_terms = np.sum([all([p[1]=='Z' for p in op]) for op in list(full_encoded_ham.terms)])
        
        return num_Z_terms, None
    
    def _encode_ham(self, f_operator:FermionOperator = None):
        """
        Encodes the fermion Hamiltonian with Compact Encoding.

        Ignores coefficients in the Hamiltonian.

        Args:
            f_operator (FermionOperator): The fermion Hamiltonian to be encoded.

        Returns:
            QubitOperator: The encoded Hamiltonian as a qubit operator (OpenFermion).
        """
        
        compact_ham_dict = self._encode_compact_ham_terms(f_operator)
        return self._get_qubit_operator(compact_ham_dict)

    def _encode_compact_ham_terms(self, f_operator:FermionOperator = None):
        """
        Encodes the fermion Hamiltonian with Compact Encoding and returns a list of terms.

        Ignores coefficients in the Hamiltonian.

        Args:
            f_operator (FermionOperator): The fermion Hamiltonian to be encoded.

        Returns:
            list: The encoded compact operators.
        """
        
        f_ham_terms = f_operator.terms
        compact_terms = []
        for f_key in f_ham_terms.keys():
            compact_terms.append(self._encode_compact_term(f_key))
        
        #flattening the list and deleting duplicates, 
        #because encoding individual terms can result common terms that would be combined in the encoded Hamiltonian.
        compact_terms = [item for sublist in compact_terms for item in sublist]

        unique_tuples = {tuple(sorted(d.items())) for d in compact_terms}
        compact_terms = [dict(t) for t in unique_tuples]

        return compact_terms

   
    def _encode_compact_term(self, f_term):
        """
        Encodes a single term of the fermion Hamiltonian with compact encoding.

        Args:
            f_term (tuple): The term to be encoded.

        Returns:
            list: The encoded operators.
        """
        compact_ops = []

        if len(f_term) == 4:
            # coulomb interaction
            q_coords = set()

            for f_index, dag in f_term:
                q_coords.add(self.panqec_code.get_qubit_coordinates()[f_index])
            q_coords = list(q_coords)
            compact_op_Zi = {}
            compact_op_Zi[q_coords[0]] = 'Z'
            compact_op_Zj = {}
            compact_op_Zj[q_coords[1]] = 'Z'
            compact_op_ZiZj = {}
            compact_op_ZiZj[q_coords[0]] = 'Z'
            compact_op_ZiZj[q_coords[1]] = 'Z'

            compact_ops.append(compact_op_Zi)
            compact_ops.append(compact_op_Zj)
            compact_ops.append(compact_op_ZiZj)

        else:
            # hopping or chem potential
            q_coords = dict()
            for f_index, dag in f_term:
                q_coord = self.panqec_code.get_qubit_coordinates()[f_index]
                q_coords[q_coord] = dag

            if len(q_coords.keys()) == 1:
                # chem potential (single Z)
                compact_op = {}
                q_coord = list(q_coords.keys())[0]
                compact_op[q_coord] = 'Z'
                compact_ops.append(compact_op)

            # half of the hopping term
            else:
               
                q_logicals_dict = self.panqec_code.get_qubit_logicals_dict()

                q_coord1 = list(q_coords.keys())[0]
                q_coord2 = list(q_coords.keys())[1]

                logicals1 = q_logicals_dict[q_coord1]
                logicals2 = q_logicals_dict[q_coord2]

                #finds the logical operator that contains both qubits
                overlap_dict = [d1 for d1 in logicals1 for d2 in logicals2 if d1 == d2][0]

                compact_op = copy.deepcopy(overlap_dict)
                
                #decides between E_ij*V_j and E_ij*V_i based on the dag value
                if ((q_coords[q_coord1] == 1 and overlap_dict[q_coord1] == 'X')
                        or (q_coords[q_coord2] == 1 and overlap_dict[q_coord2] == 'X')):
                    compact_op[q_coord2] = 'X'
                    compact_op[q_coord1] = 'X'
                else:
                    compact_op[q_coord2] = 'Y'
                    compact_op[q_coord1] = 'Y'

                compact_ops.append(compact_op)
        return compact_ops
    
    def _get_qubit_operator(self, encoded_ops = None):

        """
        Converts the encoded operators to a QubitOperator.

        Args:
            encoded_ops (list): The encoded operators.

        Returns:
            QubitOperator: The QubitOperator representation of the encoded operators.

        """
        
        hamiltonian = QubitOperator()
        pauli_operators = encoded_ops
        location_to_qubit = self.panqec_code.qubit_index
        for pauli_dict in pauli_operators:
            term = QubitOperator("", 1)
            for location, pauli in pauli_dict.items():
                qubit_index = location_to_qubit[location]
                term *= QubitOperator(f"{pauli}{qubit_index}", 1)
            
            hamiltonian += term
        return hamiltonian
    
    def _get_full_encoded_ham_length(self):
        """
        Get the number of terms in the full encoded Hamiltonian.

        Returns:
            int: The number of terms in the full encoded Hamiltonian.
        """
        
        return len(list(self.full_encoded_ham.terms.items()))
    
    def _get_num_Z_obs(self):
        """
        Get the number of Z observables in the encoded Hamiltonian.

        Returns:
            int: The number of Z observables.
        """
        num_Z_terms = np.sum([all([p[1]=='Z' for p in op]) for op in list(self.full_encoded_ham.terms)])
        return num_Z_terms
    
    ###NON-UNITARY STATE PREP CIRCUIT METHODS###
    '''
    These methods return circuits for non-unitary codespace preparation as well as counters for flag and stabilizer measurements.
    The counts are to be used when creating detectors in the stabilizer generators postselection circuits.
    '''
    def _all_stabilizer_generators_state_prep(self, p1=0, p2=0, pm=0, psp=0, pi=0, flags=False):
        """
        Create a circuit for state preparation by measuring all stabilizer generators.

        Args:
            p1 (float): Probability of depolarization for single-qubit gates.
            p2 (float): Probability of depolarization for two-qubit gates.
            pm (float): Probability of measurement error.
            psp (float): Probability of state preparation error.
            pi (float): Probability of idling error.
            flags (bool): Whether to include flag measurements.

        Returns:
            tuple: A tuple containing the circuit, flag measurement counter, and stabilizer measurement counter.
        
        """

        code = self.panqec_code
        circuit = stim.Circuit()
        qubit_coords = code.get_qubit_coordinates()
        num_qubits = len(qubit_coords)
        stab_coords = code.get_stabilizer_coordinates()
        num_stabs = len(stab_coords)
        ticktock_dict = self._make_ticktock_dict(0, num_stabs)
        timestep_dict = self._make_timestep_dict(0, num_stabs)
        self._stab_indexes_in_order_of_meas = range(num_stabs)

        if flags:
            for i_stab in range(2*num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        else: 
            for i_stab in range(num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        
        circuit.append("TICK")

        #adding stabilizer measurement gates in the order of timesteps
        for timestep in timestep_dict.keys():
            
            # iterating through all the qubits involved in this step to get the Pauli's in the stab measurements
            for q_loc in timestep_dict[timestep].keys():
                
                gate = timestep_dict[timestep][q_loc][0]
                q_ind = code.qubit_index[q_loc]
                i_stab = timestep_dict[timestep][q_loc][1]
                
                circuit.append("C"+gate, [i_stab+num_qubits, q_ind])
                if p2:
                    circuit.append("DEPOLARIZE2", [i_stab+num_qubits, q_ind], p2)
            
            circuit.append("TICK")
            
            #adding CZs for flag
            if flags and (timestep == 0 or timestep == 6):
                for i_stab in range(num_stabs):
                    circuit.append("CZ", [i_stab+num_qubits, i_stab+num_qubits+num_stabs])
                    if p2:
                        circuit.append("DEPOLARIZE2",[i_stab+num_qubits, i_stab+num_qubits+num_stabs], p2)
            
            #idling error on flag ancillas
            elif flags:
                for i_stab in range(num_stabs):
                    if pi:
                        circuit.append("DEPOLARIZE1",[i_stab+num_qubits+num_stabs], pi)
            
            #idling error on data qubits
            for i_q, q_loc1 in enumerate(qubit_coords):
                    if not timestep in ticktock_dict[q_loc1]:
                        if pi:
                            circuit.append("DEPOLARIZE1", [i_q], pi)
            circuit.append("TICK")
        
        flags_meas_counter = 0
        stab_meas_counter = 0
        #flag and stabilizer measurements
        if flags:
            for i_stab in range(2*num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
                circuit.append("MR", [i_stab+num_qubits], pm)
        else: 
            for i_stab in range(num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
                circuit.append("MR", [i_stab+num_qubits], pm)
        
        circuit.append("TICK")
        
        #flag detectors
        if flags:
            for i_flag in range(num_stabs):
                circuit.append("DETECTOR", [stim.target_rec(-1-i_flag)])
                flags_meas_counter += 1
            
            circuit.append("TICK")
        
        stab_meas_counter = num_stabs

        return circuit, flags_meas_counter, stab_meas_counter

    def _all_stabilizer_generators_Z_basis_state_prep(self, p1=0, p2=0, pm=0, psp=0, pi=0):
        
        """
        Create a circuit for state preparation by measuring all stabilizer generators for the Z basis (measures only the face qubits, justified in the paper).

        Args:
            p1 (float): Probability of depolarization for single-qubit gates.
            p2 (float): Probability of depolarization for two-qubit gates.
            pm (float): Probability of measurement error.
            psp (float): Probability of state preparation error.
            pi (float): Probability of idling error.

        Returns:
            tuple: A tuple containing the circuit, flag measurement counter, and stabilizer measurement counter.
        """
        code = self.panqec_code
        circuit = stim.Circuit()
        qubit_coords = code.get_qubit_coordinates()
        num_qubits = len(qubit_coords)
        stab_coords = code.get_stabilizer_coordinates()
        num_stabs = len(stab_coords)
        ticktock_dict = self._make_ticktock_dict(0, num_stabs)
        timestep_dict = self._make_timestep_dict(0, num_stabs)
        self._stab_indexes_in_order_of_meas = range(num_stabs)

        
        #state prep of ancillas, assume that state prep noise above already includes all state prep noise
        for i_stab in range(num_stabs):
            circuit.append("H", [i_stab+num_qubits])
            if p1:
                circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        
        circuit.append("TICK")

        #adding stabilizer measurement gates in the order of timesteps
        for timestep in timestep_dict.keys():
            
            #skip CZs because vertex qubits are assumed to be in product state already
            if (timestep >= 0 and timestep < 4):
                continue
            
            # iterating through all the qubits involved in this step to get the Paulis in the stab measurements
            for q_loc in timestep_dict[timestep].keys():
                
                gate = timestep_dict[timestep][q_loc][0]
                q_ind = code.qubit_index[q_loc]
                i_stab = timestep_dict[timestep][q_loc][1]
                
                circuit.append("C"+gate, [i_stab+num_qubits, q_ind])
                if p2:
                    circuit.append("DEPOLARIZE2", [i_stab+num_qubits, q_ind], p2)
            
            circuit.append("TICK")
                        
            #idling error on data qubits
            for i_q, q_loc1 in enumerate(qubit_coords):
                    if not timestep in ticktock_dict[q_loc1]:
                        if pi:
                            circuit.append("DEPOLARIZE1", [i_q], pi)
            
            circuit.append("TICK")
            
        flags_meas_counter = 0
        stab_meas_counter = 0
        for i_stab in range(num_stabs):
            circuit.append("H", [i_stab+num_qubits])
            if p1:
                circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
            circuit.append("MR", [i_stab+num_qubits], pm)
            stab_meas_counter += 1
    
        circuit.append("TICK")
        
        return circuit, flags_meas_counter, stab_meas_counter

    def _second_half_stabilizer_generators_state_prep(self, p1=0, p2=0, pm=0, psp=0, pi=0, flags=False):
        
        """
        Create a circuit for state preparation by measuring the second half of the stabilizer generators. Useful for state preparation for Z basis measurements with stabilizer reconstruction.

        Args:
            p1 (float): Probability of depolarization for single-qubit gates.
            p2 (float): Probability of depolarization for two-qubit gates.
            pm (float): Probability of measurement error.
            psp (float): Probability of state preparation error.
            pi (float): Probability of idling error.
            flags (bool): Whether to include flag measurements.

        Return:
            tuple: A tuple containing the circuit, flag measurement counter, and stabilizer measurement counter.

        """

        code = self.panqec_code
        circuit = stim.Circuit()
        qubit_coords = code.get_qubit_coordinates()
        stab_coords = code.get_stabilizer_coordinates()
        stab_coords_first_half = stab_coords[:num_stabs//2]
        num_stabs = len(stab_coords)
        stab_coords = stab_coords[num_stabs//2:]
        num_qubits = len(qubit_coords)
        ticktock_dict = self._make_ticktock_dict(num_stabs//2, num_stabs)
        timestep_dict = self._make_timestep_dict(num_stabs//2, num_stabs)
        num_stabs = len(stab_coords)
        self._stab_indexes_in_order_of_meas = range(num_stabs//2, num_stabs)

        qubit_dict = {}
        for i_stab, stab_loc in enumerate(stab_coords_first_half):
            stabilizer = code.get_stabilizer(stab_loc)
            
            for q_loc in stabilizer.keys():
                gate = stabilizer[q_loc]
                q_ind = code.qubit_index[q_loc]
                
                #check for overlapping measurement with a different gate
                if q_loc in qubit_dict.keys() and not qubit_dict[q_loc] == gate:
                    print("Error creating state prep circuit")
                    return qubit_dict
                
                # adding a measurement and observable based on the gate in the stabilizer
                if not q_loc in qubit_dict.keys():
                    qubit_dict[q_loc] = gate
                    if gate != 'Z':
                        circuit.append("H", [q_ind])
                        if p1:
                            circuit.append('DEPOLARIZE1', [q_ind], p1)
                    if gate == "Y":
                        circuit.append("S", [q_ind])
                        if p1:
                            circuit.append('DEPOLARIZE1', [q_ind], p1)
                
        if flags:
            for i_stab in range(2*num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        else: 
            for i_stab in range(num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        
        circuit.append("TICK")

        #adding stabilizer measurement gates in the order of timesteps
        for timestep in timestep_dict.keys():
            
            # iterating through all the qubits involved in this step to get the Pauli's in the stab measurements
            for q_loc in timestep_dict[timestep].keys():
                
                gate = timestep_dict[timestep][q_loc][0]
                q_ind = code.qubit_index[q_loc]
                i_stab = timestep_dict[timestep][q_loc][1]
                
                circuit.append("C"+gate, [i_stab+num_qubits, q_ind])
                if p2:
                    circuit.append("DEPOLARIZE2", [i_stab+num_qubits, q_ind], p2)
            
            circuit.append("TICK")
            
            #adding CZs for flag
            if flags and (timestep == 0 or timestep == 6):
                for i_stab in range(num_stabs):
                    circuit.append("CZ", [i_stab+num_qubits, i_stab+num_qubits+num_stabs])
                    if p2:
                        circuit.append("DEPOLARIZE2",[i_stab+num_qubits, i_stab+num_qubits+num_stabs], p2)
            
            #idling error on flag ancillas
            elif flags:
                for i_stab in range(num_stabs):
                    if pi:
                        circuit.append("DEPOLARIZE1",[i_stab+num_qubits+num_stabs], pi)
            
            #idling error on data qubits
            for i_q, q_loc1 in enumerate(qubit_coords):
                    if not timestep in ticktock_dict[q_loc1]:
                        if pi:
                            circuit.append("DEPOLARIZE1", [i_q], pi)
            circuit.append("TICK")
        
        flags_meas_counter = 0
        stab_meas_counter = 0
        #flag and stabilizer measurements
        if flags:
            for i_stab in range(2*num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
                circuit.append("MR", [i_stab+num_qubits], pm)
            stab_meas_counter = num_stabs
        else: 
            for i_stab in range(num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
                circuit.append("MR", [i_stab+num_qubits], pm)
                stab_meas_counter += 1
            flags_meas_counter = 0
        circuit.append("TICK")
        
        #flag detectors
        if flags:
            for i_flag in range(num_stabs):
                circuit.append("DETECTOR", [stim.target_rec(-1-i_flag)])
                flags_meas_counter += 1
        circuit.append("TICK")

        return circuit, flags_meas_counter, stab_meas_counter

    def _second_half_stabilizer_generators_Z_basis_state_prep(self, p1=0, p2=0, pm=0, psp=0, pi=0):

        """
        Create a circuit for state preparation by measuring the second half of the stabilizer generators for the Z basis (measures only the face qubits, justified in the paper).

        Args:
            p1 (float): Probability of depolarization for single-qubit gates.
            p2 (float): Probability of depolarization for two-qubit gates.
            pm (float): Probability of measurement error.
            psp (float): Probability of state preparation error.
            pi (float): Probability of idling error.

        Returns:
            tuple: A tuple containing the circuit, flag measurement counter, and stabilizer measurement counter.
        """
        
        code = self.panqec_code
        circuit = stim.Circuit()
        qubit_coords = code.get_qubit_coordinates()
        stab_coords = code.get_stabilizer_coordinates()
        num_stabs = len(stab_coords)
        stab_coords_first_half = stab_coords[:num_stabs//2]
        stab_coords = stab_coords[num_stabs//2:]
        num_qubits = len(qubit_coords)
        ticktock_dict = self._make_ticktock_dict(num_stabs//2, num_stabs)
        timestep_dict = self._make_timestep_dict(num_stabs//2, num_stabs)
        num_stabs = len(stab_coords)
        qubit_dict = {}
        self._stab_indexes_in_order_of_meas = range(num_stabs//2, num_stabs)
        for i_stab, stab_loc in enumerate(stab_coords_first_half):
            stabilizer = code.get_stabilizer(stab_loc)
            
            for q_loc in stabilizer.keys():
                gate = stabilizer[q_loc]
                q_ind = code.qubit_index[q_loc]
                
                #check for overlapping measurement with a different gate
                if q_loc in qubit_dict.keys() and not qubit_dict[q_loc] == gate:
                    print("Error creating state prep circuit")
                    return qubit_dict
                
                # adding a measurement and observable based on the gate in the stabilizer
                if not q_loc in qubit_dict.keys():
                    qubit_dict[q_loc] = gate
                    if gate != 'Z':
                        circuit.append("H", [q_ind])
                        if p1:
                            circuit.append('DEPOLARIZE1', [q_ind], p1)
                    if gate == "Y":
                        circuit.append("S", [q_ind])
                        if p1:
                            circuit.append('DEPOLARIZE1', [q_ind], p1)

        

        for i_stab in range(num_stabs):
            circuit.append("H", [i_stab+num_qubits])
            if p1:
                circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        
        circuit.append("TICK")

        #adding stabilizer measurement gates in the order of timesteps
        for timestep in timestep_dict.keys():
            
            #skip CZs because vertex qubits are assumed to be in product state already
            if (timestep >= 0 and timestep < 4):
                continue
            # iterating through all the qubits involved in this step to get the Paulis in the stab measurements
            for q_loc in timestep_dict[timestep].keys():
                
                gate = timestep_dict[timestep][q_loc][0]
                q_ind = code.qubit_index[q_loc]
                i_stab = timestep_dict[timestep][q_loc][1]
                
                circuit.append("C"+gate, [i_stab+num_qubits, q_ind])
                if p2:
                    circuit.append("DEPOLARIZE2", [i_stab+num_qubits, q_ind], p2)
            
            circuit.append("TICK")
            
            #idling error on data qubits
            if pi:
                for i_q, q_loc1 in enumerate(qubit_coords):
                        if not timestep in ticktock_dict[q_loc1]:
                            circuit.append("DEPOLARIZE1", [i_q], pi)
                circuit.append("TICK")
        
        flags_meas_counter = 0
        stab_meas_counter = 0
        #stabilizer measurements
        for i_stab in range(num_stabs):
            circuit.append("H", [i_stab+num_qubits])
            if p1:
                circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
            circuit.append("MR", [i_stab+num_qubits], pm)
            stab_meas_counter += 1
        
        circuit.append("TICK")

        return circuit, flags_meas_counter, stab_meas_counter
    
    ###LOGICAL ROTATIONS CIRCUIT METHODS###

    def get_efficient_trotter_step(self):
        """
        Get the efficient Trotter step circuit for the encoded Hamiltonian.
        This attempts to add all the rotation in the most space efficient fashion (reducing idling). Assumes open boundary conditions.

        Returns:
            cirq.Circuit: The efficient Trotter step circuit.
        """
        def paulistr_to_cirq(pauli_str):
            if pauli_str is cirq.ops.gate_operation.GateOperation or len(pauli_str) <= 1:
                return cirq.Circuit(cirq.decompose_once(pauli_str))
            ops = cirq.decompose_once(pauli_str ** (3))
            return cirq.Circuit(ops)
        
        def col(q, L):
            return q % L

        def row(q, L):
            return (q // L) % L
        
        cirq_qubits = cirq.LineQubit.range(self.n_qubits)
        circuit = cirq.Circuit()
        
        for i in range(self.L**2):
            circuit.append( cirq.Z(cirq_qubits[i]))

        for i in range(0, self.L**2, 2):
            if (i + 1) % self.L != 0:  # Check if not at the end of a row
                zzpauli = cirq.Z(cirq_qubits[i]) * cirq.Z(cirq_qubits[i + 1])
                circuit += paulistr_to_cirq(zzpauli)
        
        for i in range(1, self.L**2-1, 2):
            if (i + 1) % self.L != 0:  # Check if not at the end of a row
                zzpauli = cirq.Z(cirq_qubits[i]) * cirq.Z(cirq_qubits[i + 1])
                circuit += paulistr_to_cirq(zzpauli)
            
        for i in range(self.L**2 - self.L):
            zzpauli = cirq.Z(cirq_qubits[i]) * cirq.Z(cirq_qubits[i + self.L])
            circuit += paulistr_to_cirq(zzpauli)

        code = self.panqec_code
        logicals = self.panqec_code.get_logicals_x()
        
        for i in range(4):
            for logical in logicals[i::4]:
                pauli1 = cirq.MutablePauliString(cirq.I(cirq_qubits[0]))
                pauli2 = cirq.MutablePauliString(cirq.I(cirq_qubits[0]))
                if code.x_op_orientation(logical) == 'h':
                    for q_coord in logical.keys():
                        if code.qubit_type(q_coord) == 'face':
                            pauli1 *= cirq.Y(cirq_qubits[code.qubit_index[q_coord]])
                            pauli2 *= cirq.Y(cirq_qubits[code.qubit_index[q_coord]])
                        elif logical[q_coord] == 'X':
                            pauli1 *= cirq.X(cirq_qubits[code.qubit_index[q_coord]])
                            pauli2 *= cirq.Y(cirq_qubits[code.qubit_index[q_coord]])
                        elif logical[q_coord] == 'Y':
                            pauli1 *= cirq.X(cirq_qubits[code.qubit_index[q_coord]])
                            pauli2 *= cirq.Y(cirq_qubits[code.qubit_index[q_coord]])
                else:
                    for q_coord in logical.keys():
                        if code.qubit_type(q_coord) == 'face':
                            pauli1 *= cirq.X(cirq_qubits[code.qubit_index[q_coord]])
                            pauli2 *= cirq.X(cirq_qubits[code.qubit_index[q_coord]])
                        elif logical[q_coord] == 'X':
                            pauli1 *= cirq.X(cirq_qubits[code.qubit_index[q_coord]])
                            pauli2 *= cirq.Y(cirq_qubits[code.qubit_index[q_coord]])
                        elif logical[q_coord] == 'Y':
                            pauli1 *= cirq.X(cirq_qubits[code.qubit_index[q_coord]])
                            pauli2 *= cirq.Y(cirq_qubits[code.qubit_index[q_coord]])
                pauli1_qubits = [q.x for q, p in pauli1.items() if p != cirq.I and q.x < self.L**2]
                pauli2_qubits = [q.x for q, p in pauli2.items() if p != cirq.I and q.x < self.L**2]
                
                pauli1_qubits.sort(key=lambda q: (row(q, self.L), col(q, self.L)))
                pauli2_qubits.sort(key=lambda q: (row(q, self.L), col(q, self.L)))
                
                if not ((col(pauli1_qubits[0], self.L) == 0 and col(pauli1_qubits[-1], self.L) == self.L-1) or 
                        (row(pauli1_qubits[0], self.L) == 0 and row(pauli1_qubits[-1], self.L) == self.L-1) or
                        (col(pauli2_qubits[0], self.L) == 0 and col(pauli2_qubits[-1], self.L) == self.L-1) or 
                        (row(pauli2_qubits[0], self.L) == 0 and row(pauli2_qubits[-1], self.L) == self.L-1)):
                    circuit += paulistr_to_cirq(cirq.PauliString(pauli1))
                    circuit += paulistr_to_cirq(cirq.PauliString(pauli2))
        return circuit

    def _get_efficient_trotter_stim_circuit(self, trotter_steps):

        """
        Get the efficient Trotter steps circuit for the encoded Hamiltonian for STIM. 

        Args:
            trotter_steps (int): The number of Trotter steps to add.

        Returns:
            stim.Circuit: The efficient Trotter steps circuit.
        """

        cirq_circuit = self.get_efficient_trotter_step()*trotter_steps
        opt_stim_circuit = self._cirq_to_stim_optimize(cirq_circuit)
        mirror_stim_circuit = self._make_mirror_circuit(opt_stim_circuit)
        return mirror_stim_circuit


    def _get_cirq_circuit(self, vqed=False):
        """
        Get the Cirq circuit representation of the encoded operator (the random logical operators or subset of the Hamiltonian terms).

        Args:
            vqed (bool): Whether to use VQED.

        Returns:
            cirq.Circuit: The Cirq circuit representation.
        """
        def paulistr_to_cirq(pauli_str):
            if pauli_str is cirq.ops.gate_operation.GateOperation or len(pauli_str) <= 1:
                return cirq.Circuit(cirq.decompose_once(pauli_str))
            ops = cirq.decompose_once(pauli_str ** (3))
            return cirq.Circuit(ops)

        ham_paulisum = [of.transforms.qubit_operator_to_pauli_sum(op) for op in self.encoded_subset_operators]
        ham_as_list = []
        for ham_term in ham_paulisum:

            ham_as_list += list(ham_term.__iter__())
        
        if vqed:
            circuits = []
            for ham_term in ham_as_list:
                cirq_qubits = cirq.LineQubit.range(self.n_qubits)
                circuit = cirq.Circuit()

                for cirq_q in cirq_qubits:
                    circuit += cirq.I(cirq_q)

                mutable_term = cirq.MutablePauliString(ham_term)
                mutable_term.coefficient = 1
                circuit += paulistr_to_cirq(cirq.PauliString(mutable_term))
                circuits.append(circuit)
            return circuits
        else:
            cirq_qubits = cirq.LineQubit.range(self.n_qubits)
            circuit = cirq.Circuit()

            for cirq_q in cirq_qubits:
                circuit += cirq.I(cirq_q)
            for ham_term in ham_as_list:
                mutable_term = cirq.MutablePauliString(ham_term)
                mutable_term.coefficient = 1
                circuit += paulistr_to_cirq(cirq.PauliString(mutable_term))
            return circuit
    
    def _cirq_to_stim_optimize(self, cirq_circuit):
        """
        Convert the Cirq circuit to an "optimized" STIM circuit.

        Args:
            cirq_circuit (cirq.Circuit): The Cirq circuit to convert to STIM.

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

    def _make_mirror_circuit(self, stim_circuit):
        """
        Creates a mirror circuit by appending the inverse of the given STIM circuit.

        Args:
            stim_circuit (stim.Circuit): The STIM circuit.

        Returns:
            stim.Circuit: The mirror circuit.
        """
        return stim_circuit + stim_circuit.inverse()

    def _add_noise_to_logical_stim_circ(self, stim_circuit, p1=0, p2=0, pi=0, pm=0, psp=0):
        """
        Adds noise to the given STIM circuit. The circuit is presumed to only contain encoded logical operators (no encoding/stabilizer measurement/etc.).
        The noise in the state prep, stabilizer measurements, and final measurements is added in the functions that create those circuits.

        Args:
            stim_circuit (stim.Circuit): The STIM circuit.
            p1 (float, optional): The depolarization probability for single-qubit gates. Defaults to 0.
            p2 (float, optional): The depolarization probability for two-qubit gates. Defaults to 0.
            pi (float, optional): The depolarization probability for idling. Defaults to 0.

        Returns:
            stim.Circuit: The STIM circuit with added noise.
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
            if (stim_circuit[i].name == 'TICK' and stim_circuit[i + 1].name == 'TICK') or stim_circuit[i].name == 'I':
                continue
            else:
                stim_circuit_clean.append(stim_circuit[i])

        for circ_element in stim_circuit_clean:

            #skip I gates and QUBIT_COORDS
            if circ_element.name in elements_to_skip:
                noisy_circuit.append(circ_element)

            #VQED elements
            elif circ_element.name == 'R':
                noisy_circuit.append(circ_element)
                noisy_circuit.append('X_ERROR', circ_element.targets_copy(), psp)
            elif circ_element.name == 'MR':
                noisy_circuit.append('MR', circ_element.targets_copy(), pm)
            elif circ_element.name == 'DETECTOR':
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
                #idling error
                if pi:
                    for i in range(q_num):
                        if i not in used_qubits_in_timestep:
                            noisy_circuit.append('DEPOLARIZE1', i, pi)

                noisy_circuit.append('TICK')
                used_qubits_in_timestep = set()
        
        return noisy_circuit
    
   
    def _make_vqed_logical_stim_circ(self, cirq_circuits, vqed_rate, flags):

        """
        Create a VQED logical circuit by adding random stabilizer measurements to the given Cirq circuits.

        Args:
            cirq_circuits (list): The Cirq circuits to which VQED measurements will be added.
            vqed_rate (int): The rate of VQED measurements.
            flags (bool): Whether to include flag measurements.

        Returns:
            stim.Circuit: The VQED logical circuit.
            dict: A dictionary mapping measurement indices to stabilizer coordinates or 'f' for flag measurements.
        """

        code = self.panqec_code
        stab_coords = code.get_stabilizer_coordinates()
        new_circuit = stim.Circuit()
        vqed_counter = 0
        measure_counter = 0
        self._measure_jk_dict = {} #dictionary to keep track of the j and k indices for the VQED measurements or 'f' for flag measurements
       
        stim_circuits = []
        stim_inv_circuits = []
        for cirq_circuit in cirq_circuits:
            stim_circuits.append(stimcirq.cirq_circuit_to_stim_circuit(cirq_circuit))
            stim_inv_circuits.append(stimcirq.cirq_circuit_to_stim_circuit(cirq_circuit).inverse())

        effective_length = len(stim_circuits+stim_inv_circuits)
        vqed_rate_int = int(vqed_rate*effective_length)

        if vqed_rate != -1:
            for stim_log_circuit in (stim_circuits+stim_inv_circuits):
                if vqed_counter % vqed_rate_int == 0 and vqed_counter != 0:
                    j = np.random.randint(0, len(stab_coords))
                    k = np.random.randint(0, len(stab_coords))
                    stab_j = code.get_stabilizer(stab_coords[j])
                    stab_k = code.get_stabilizer(stab_coords[k])
                    for q_loc in stab_j.keys():
                        q_ind = code.qubit_index[q_loc]
                        gate = stab_j[q_loc]
                        new_circuit.append(gate, [q_ind])
                    
                    new_circuit.append('R', [self.n_qubits])
                    new_circuit.append('H', [self.n_qubits])

                    if flags:
                        new_circuit.append('R', [self.n_qubits+1])
                        new_circuit.append('H', [self.n_qubits+1])
                    flag_counter = 0
                    for q_loc in stab_k.keys():

                        q_ind = code.qubit_index[q_loc]
                        gate = stab_k[q_loc]
                        if gate == 'X':
                            new_circuit.append('CNOT', [self.n_qubits, q_ind])
                        elif gate == 'Y':
                            new_circuit.append('CY', [self.n_qubits, q_ind])
                        elif gate == 'Z':
                            new_circuit.append('CZ', [self.n_qubits, q_ind])

                        if flags and (flag_counter == 0 or flag_counter == len(stab_k.keys())-2):
                            new_circuit.append('CZ', [self.n_qubits+1, self.n_qubits])
                        flag_counter += 1

                    new_circuit.append('H', [self.n_qubits])
                    
                    new_circuit.append('MR', [self.n_qubits])
                    self._measure_jk_dict[measure_counter] = (j, k)
                    measure_counter += 1

                    if flags:
                        new_circuit.append('H', [self.n_qubits+1])
                        new_circuit.append('MR', [self.n_qubits+1])
                        self._measure_jk_dict[measure_counter] = 'f'
                        measure_counter += 1

                new_circuit += stim_log_circuit
                vqed_counter += 1
        else:
            new_circuit = self._make_mirror_circuit(self._cirq_to_stim_optimize())
            j = np.random.randint(0, len(stab_coords))
            k = np.random.randint(0, len(stab_coords))
            stab_j = code.get_stabilizer(stab_coords[j])
            stab_k = code.get_stabilizer(stab_coords[k])
            
            for q_loc in stab_j.keys():
                q_ind = code.qubit_index[q_loc]
                gate = stab_j[q_loc]
                new_circuit.append(gate, [q_ind])
            
            new_circuit.append('R', [self.n_qubits])
            new_circuit.append('H', [self.n_qubits])
            if flags:
                new_circuit.append('R', [self.n_qubits+1])
                new_circuit.append('H', [self.n_qubits+1])
            flag_counter = 0
            for q_loc in stab_k.keys():
                q_ind = code.qubit_index[q_loc]
                gate = stab_k[q_loc]
                if gate == 'X':
                    new_circuit.append('CNOT', [self.n_qubits, q_ind])
                elif gate == 'Y':
                    new_circuit.append('CY', [self.n_qubits, q_ind])
                elif gate == 'Z':
                    new_circuit.append('CZ', [self.n_qubits, q_ind])
                
                if flags and (flag_counter == 0 or flag_counter == len(stab_k.keys())-2):
                    new_circuit.append('CZ', [self.n_qubits+1, self.n_qubits])
                flag_counter += 1

            new_circuit.append('H', [self.n_qubits])
            
            new_circuit.append('MR', [self.n_qubits])
            self._measure_jk_dict[measure_counter] = (j, k)
            measure_counter += 1

            if flags:
                new_circuit.append('H', [self.n_qubits+1])
                new_circuit.append('MR', [self.n_qubits+1])
                self._measure_jk_dict[measure_counter] = 'f'
                measure_counter += 1
        
        new_circuit.append('TICK')
        return new_circuit, measure_counter

    def make_logical_circuit(self, vqed_rate = 0, p1=0, p2=0, pm=0, pi=0, psp=0, flags=False):
        '''
        Creates a logical circuit with noise.
        
        Args:
            vqed_rate (int, optional): The rate at which VQED measurements are added. Defaults to 0.
            p1 (float, optional): The depolarization probability for single-qubit gates. Defaults to 0.
            p2 (float, optional): The depolarization probability for two-qubit gates. Defaults to 0.
            pm (float, optional): The depolarization probability for measurement. Defaults to 0.
            pi (float, optional): The depolarization probability for idling. Defaults to 0.
            psp (float, optional): The depolarization probability for state preparation. Defaults to 0.
            flags (bool, optional): Whether to include flag measurements. Defaults to False.

        Return:
            stim.Circuit: The logical circuit with noise.        
            
        '''
        logical_circuit = stim.Circuit()
        if vqed_rate:
            logical_circuit, measure_counter = self._make_vqed_logical_stim_circ(self._get_cirq_circuit(vqed = bool(vqed_rate)), vqed_rate, flags)
            noisy_logical_circuit = self._add_noise_to_logical_stim_circ(logical_circuit, p1, p2, pi, pm, psp)

        else:
            noisy_logical_circuit = self._add_noise_to_logical_stim_circ(self._make_mirror_circuit(self._cirq_to_stim_optimize(self._get_cirq_circuit())), p1, p2, pi, pm, psp)
            measure_counter = 0

        return noisy_logical_circuit, measure_counter

    ###STABILIZER GENERATORS POSTSELECTION CIRCUIT METHODS###  

    def _make_ticktock_dict(self, stabilizer_start_index, stabilizer_end_index):
        """
        Creates a dictionary mapping qubit coordinates to timestep in which the qubit should be entangled with an ancilla.
        Takes the stabilizers from the given range.
        {qubit_coordinates: [timestep1, timestep2, ...]}

        Args:
            stabilizer_start_index (int): The starting index of the stabilizers.
            stabilizer_end_index (int): The ending index of the stabilizers.

        Returns:
            dict: The ticktock dictionary.
        """
        code = self.panqec_code
        qubit_coords = code.get_qubit_coordinates()
        stab_coords = code.get_stabilizer_coordinates()[stabilizer_start_index:stabilizer_end_index]
        ticktock_dict = {}

        for qubit_coord in qubit_coords:
            ticktock_dict[qubit_coord] = []

        for i_stab, stab_loc in enumerate(stab_coords):
            stabilizer = code.get_stabilizer(stab_loc)
            stabilizer_array = np.array(list(stabilizer.keys()))
            for i_q, q in enumerate(stabilizer_array):
                q = tuple(q)
                ticktock_dict[q].append(i_q)
        return ticktock_dict
    
    def _make_timestep_dict(self, stabilizer_start_index, stabilizer_end_index):
        """
        Creates a dictionary that maps timestep indices to a dictionary of qubit locations and their corresponding gate and stabilizer index: {timestep_index: {qubit_location : [gate (stabilizer), stabilizer index]}}.
        Takes the stabilizers from the given range.

        Args:
            stabilizer_start_index (int): The starting index of the stabilizers.
            stabilizer_end_index (int): The ending index of the stabilizers.

        Returns:
            dict: The timestep dictionary.
        """
        code = self.panqec_code
        stab_coords = code.get_stabilizer_coordinates()[stabilizer_start_index:stabilizer_end_index]
        timestep_dict = {t: {} for t in range(8)}
        for i_stab, stab_loc in enumerate(stab_coords):
            stabilizer = code.get_stabilizer(stab_loc)
            for i_q, q in enumerate(stabilizer):
                q = tuple(q)
                timestep_dict[i_q][q] = [stabilizer[q], i_stab]
        return timestep_dict
    
    def _full_stabilizer_generators_detection_circuit(self, stim_circ:stim.Circuit, p1=0, p2=0, pm=0, psp=0, pi=0, flags=False, flag_meas_counter=0, stab_meas_counter=0, vqed_measure_counter=0):
        """
        Create a circuit for measuring all stabilizer generators at the end. 
        
        Args:
            stim_circ (stim.Circuit): The STIM circuit to which the stabilizer generators will be added.
            p1 (float): Probability of depolarization for single-qubit gates.
            p2 (float): Probability of depolarization for two-qubit gates.
            pm (float): Probability of measurement error.
            psp (float): Probability of state preparation error.
            pi (float): Probability of idling error.
            flags (bool): Whether to include flag measurements.
            flag_meas_counter (int): The number of flag measurements.
            stab_meas_counter (int): The number of stabilizer measurements.
            vqed_measure_counter (int): The number of VQED measurements.
        
        Return:
        tuple: A tuple containing the circuit, flag measurement counter, and stabilizer measurement counter.
        """

        code = self.panqec_code
        circuit = stim_circ.copy()
        qubit_coords = code.get_qubit_coordinates()
        num_qubits = len(qubit_coords)
        stab_coords = code.get_stabilizer_coordinates()
        num_stabs = len(stab_coords)
        ticktock_dict = self._make_ticktock_dict(0, num_stabs)
        timestep_dict = self._make_timestep_dict(0, num_stabs)

        if flags:
            for i_stab in range(2*num_stabs):
                if psp:
                    circuit.append("X_ERROR", [i_stab+num_qubits], psp)
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        else: 
            for i_stab in range(num_stabs):
                if psp:
                    circuit.append("X_ERROR", [i_stab+num_qubits], psp)
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        
        circuit.append("TICK")

        #adding stabilizer measurement gates in the order of timesteps
        for timestep in timestep_dict.keys():
            
            # iterating through all the qubits involved in this step to get the Pauli's in the stab measurements
            for q_loc in timestep_dict[timestep].keys():
                
                gate = timestep_dict[timestep][q_loc][0]
                q_ind = code.qubit_index[q_loc]
                i_stab = timestep_dict[timestep][q_loc][1]
                
                circuit.append("C"+gate, [i_stab+num_qubits, q_ind])
                if p2:
                    circuit.append("DEPOLARIZE2", [i_stab+num_qubits, q_ind], p2)
            
            circuit.append("TICK")
            
            #adding CZs for flag
            if flags and (timestep == 0 or timestep == 6):
                for i_stab in range(num_stabs):
                    circuit.append("CZ", [i_stab+num_qubits, i_stab+num_qubits+num_stabs])
                    if p2:
                        circuit.append("DEPOLARIZE2",[i_stab+num_qubits, i_stab+num_qubits+num_stabs], p2)
            
            #idling error on flag ancillas
            elif flags:
                for i_stab in range(num_stabs):
                    if pi:
                        circuit.append("DEPOLARIZE1",[i_stab+num_qubits+num_stabs], pi)
            
            #idling error on data qubits
            for i_q, q_loc1 in enumerate(qubit_coords):
                    if not timestep in ticktock_dict[q_loc1]:
                        if pi:
                            circuit.append("DEPOLARIZE1", [i_q], pi)
            circuit.append("TICK")
        
        #this method for full generator set measurement:
        # The encoding circuit could have had (number of measurements doesn't change if only face qubits are entangled):

        flag_meas_counter2 = 0
        stab_meas_counter2 = 0
        # - full generator set measurements with/without flags
        if len(stab_coords) == stab_meas_counter:
            for i_stab in range(num_stabs):
                
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
                circuit.append("MR", [i_stab+num_qubits], pm)
                #same number of measurements here and at encoding
                circuit.append('DETECTOR', [stim.target_rec(-1-(flag_meas_counter+stab_meas_counter+vqed_measure_counter)), stim.target_rec(-1)]) 
                stab_meas_counter2 += 1

        # - second half generator set measurements with/without flags (FT readout case)
        else:
            for i_stab in range(stab_meas_counter):
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
                circuit.append("MR", [i_stab+num_qubits], pm)
                circuit.append('DETECTOR', stim.target_rec(-1))
                stab_meas_counter2 += 1
            for i_stab in range(stab_meas_counter, num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
                circuit.append("MR", [i_stab+num_qubits], pm)
                circuit.append('DETECTOR', [stim.target_rec(-1-(flag_meas_counter+num_stabs+vqed_measure_counter)), stim.target_rec(-1)]) 
                stab_meas_counter2 += 1
        
        if flags:
            for i_flag in range(num_stabs):
                circuit.append("H", [i_flag+num_qubits+num_stabs])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_flag+num_qubits+num_stabs], p1)
                circuit.append("MR", [i_flag+num_qubits+num_stabs], pm)
                circuit.append("DETECTOR", [stim.target_rec(-1)])
                flag_meas_counter2 += 1
               
        circuit.append("TICK")

        return circuit, flag_meas_counter2, stab_meas_counter2

    def _second_half_stabilizer_generators_detection_circuit(self, stim_circ:stim.Circuit, p1=0, p2=0, pm=0, psp=0, pi=0, flags=False, flag_meas_counter=0, stab_meas_counter=0,vqed_measure_counter=0):

        """
        Create a circuit for measuring second half of the stabilizer generators at the end. 
        
        Args:
            stim_circ (stim.Circuit): The STIM circuit to which the stabilizer generators will be added.
            p1 (float): Probability of depolarization for single-qubit gates.
            p2 (float): Probability of depolarization for two-qubit gates.
            pm (float): Probability of measurement error.
            psp (float): Probability of state preparation error.
            pi (float): Probability of idling error.
            flags (bool): Whether to include flag measurements.
            flag_meas_counter (int): The number of flag measurements.
            stab_meas_counter (int): The number of stabilizer measurements.
            vqed_measure_counter (int): The number of VQED measurements.
        
        Return:
        tuple: A tuple containing the circuit, flag measurement counter, and stabilizer measurement counter.
        """
        code = self.panqec_code
        circuit = stim_circ.copy()
        qubit_coords = code.get_qubit_coordinates()
        stab_coords = code.get_stabilizer_coordinates()
        num_stabs = len(stab_coords)
        stab_coords = stab_coords[num_stabs//2:]
        num_qubits = len(qubit_coords)
        ticktock_dict = self._make_ticktock_dict(num_stabs//2, num_stabs)
        timestep_dict = self._make_timestep_dict(num_stabs//2, num_stabs)
        num_stabs = len(stab_coords)

        
        if flags:
            for i_stab in range(2*num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        else: 
            for i_stab in range(num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        
        circuit.append("TICK")

        #adding stabilizer measurement gates in the order of timesteps
        for timestep in timestep_dict.keys():
            
            # iterating through all the qubits involved in this step to get the Pauli's in the stab measurements
            for q_loc in timestep_dict[timestep].keys():
                
                gate = timestep_dict[timestep][q_loc][0]
                q_ind = code.qubit_index[q_loc]
                i_stab = timestep_dict[timestep][q_loc][1]
                
                circuit.append("C"+gate, [i_stab+num_qubits, q_ind])
                if p2:
                    circuit.append("DEPOLARIZE2", [i_stab+num_qubits, q_ind], p2)
            
            circuit.append("TICK")
            
            #adding CZs for flag
            if flags and (timestep == 0 or timestep == 6):
                for i_stab in range(num_stabs):
                    circuit.append("CZ", [i_stab+num_qubits, i_stab+num_qubits+num_stabs])
                    if p2:
                        circuit.append("DEPOLARIZE2",[i_stab+num_qubits, i_stab+num_qubits+num_stabs], p2)
            
            #idling error on flag ancillas
            elif flags:
                for i_stab in range(num_stabs):
                    if pi:
                        circuit.append("DEPOLARIZE1",[i_stab+num_qubits+num_stabs], pi)
            
            #idling error on data qubits
            for i_q, q_loc1 in enumerate(qubit_coords):
                    if not timestep in ticktock_dict[q_loc1]:
                        if pi:
                            circuit.append("DEPOLARIZE1", [i_q], pi)
            circuit.append("TICK")
   
        #this method for full generator set measurement:
        # The encoding circuit could have had (number of measurements doesn't change if only face qubits are entangled):
        stab_meas_counter2 = 0
        flag_meas_counter2 = 0
        # half generators enc and half generators decoding (FT readout)
        if len(stab_coords) == stab_meas_counter:
            for i_stab in range(num_stabs):
                
                circuit.append("H", [i_stab+num_qubits])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
                circuit.append("MR", [i_stab+num_qubits], pm)
                #same number of measurements here and at encoding
                circuit.append('DETECTOR', [stim.target_rec(-1-(flag_meas_counter+stab_meas_counter+vqed_measure_counter)), stim.target_rec(-1)]) 
                stab_meas_counter2 += 1
        else:
            raise NotImplementedError("Implementing non-destructive syndrome readout of half the stabilizers when the encoding measured the entire generating set is either incorrect or the encoding step was not efficient.")
           
        
        if flags:
            for i_flag in range(num_stabs):
                circuit.append("H", [i_flag+num_qubits+num_stabs])
                if p1:
                    circuit.append('DEPOLARIZE1', [i_flag+num_qubits+num_stabs], p1)
                circuit.append("MR", [i_flag+num_qubits+num_stabs], pm)
                circuit.append("DETECTOR", [stim.target_rec(-1)])
                flag_meas_counter2 += 1
        circuit.append("TICK")
        return circuit, flag_meas_counter2, stab_meas_counter2

    ###STATE PREP (BASIS ROTATIONS) AND LOGICAL OBSERVABLES MEASUREMENT CIRCUIT METHODS###

    def _add_X_error_state_prep_error(self, stim_circuit, psp=0):
        """
        Adds X error to the state preparation circuit (only bitflips when preparing all zero state).

        Args:
            stim_circuit (stim.Circuit): The STIM circuit.
            psp (float): The error probability.

        Returns:
            stim.Circuit: The modified STIM circuit with X error.
        """
        stim_prepend_circ = stim.Circuit()
        for i in range(stim_circuit.num_qubits):
            stim_prepend_circ.append("X_ERROR", [i], psp)
        return stim_prepend_circ+stim_circuit

    
    def _add_hopping_measure(self, stim_circuit, pm = 0, p1 = 0):
        """
        Adds hopping terms measurement to the STIM circuit.

        Args:
            stim_circuit (stim.Circuit): The STIM circuit.
            pm (float): Measurement error probability.
            p1 (float): Depolarization rate for single-qubit gates.
        
        Return:
            stim.Circuit: The modified STIM circuit with hopping term measurement.
        """ 

        circuit = stim.Circuit()
        circuit.append("TICK")
        logicals = self.panqec_code.get_logicals_x()[0::4]
        observable_count = 0
        circuit_prepend = stim.Circuit()
        meas_counter = 0
        for logical_op in logicals:
            for q_loc in logical_op.keys():
                q_ind = self.panqec_code.qubit_index[q_loc]
                circuit_prepend.append("H", [q_ind])
                if p1:
                    circuit_prepend.append("DEPOLARIZE1", [q_ind], p1)

                #lazy implementation of the hopping operator instead of edge operator (this ends up being XXX for all observables,
                #but if, for example, logicals = code.get_logicals_x()[1::4] it would be different, because then it would be vertical operators)
                if q_loc[0] % 2 == 0:
                    pauli = "X"
                    circuit.append('H', [q_ind])
                    if p1:
                        circuit.append("DEPOLARIZE1", [q_ind], p1)
                    circuit.append("M", [q_ind], pm)
                    meas_counter += 1
                else:
                    #circuit.append("M"+logical_op[q_loc], [q_ind], pm)
                    meas_counter += 1
                    if logical_op[q_loc] == "X":
                        circuit.append('H', [q_ind])
                        if p1:
                            circuit.append("DEPOLARIZE1", [q_ind], p1)
                        circuit.append("M", [q_ind], pm)
                        
                    elif logical_op[q_loc] == "Y":

                        circuit.append('S_DAG', [q_ind])
                        if p1:
                            circuit.append("DEPOLARIZE1", [q_ind], p1)
                        circuit.append('H', [q_ind])
                        if p1:
                            circuit.append("DEPOLARIZE1", [q_ind], p1)
                        
                        circuit.append("M", [q_ind], pm)
                        circuit_prepend.append("S", [q_ind])
                        if p1:
                            circuit_prepend.append("DEPOLARIZE1", [q_ind], p1)

            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1),stim.target_rec(-2),stim.target_rec(-3)], observable_count)
            observable_count+=1
        return circuit_prepend+stim_circuit+circuit, observable_count, meas_counter
    
    def _add_simple_measurement_stim_circuit(self, stim_circ, pm=0, global_parity_postsel=False):
        """
        Add occupation number (Z observable) measurement gates to the STIM circuit.

        Args:
            stim_circ (stim.Circuit): The STIM circuit.
            pm (float): Measurement error probability.

        Returns:
            stim.Circuit: The modified STIM circuit with measurement gates.
        """
        for i in range(self.L**(2)):
            stim_circ.append("M", [i], pm)
            stim_circ.append("OBSERVABLE_INCLUDE", stim.target_rec(-1), i)
        if global_parity_postsel:
            stim_circ.append('DETECTOR', [stim.target_rec(-i) for i in range(1, self.L**(2)+1)])
        return stim_circ, self.L**(2), self.L**(2) #last thing is number of measurements
    
    def _add_observables_from_dict_Z(self, stim_circuit, qubit_measure_dict, qubit_dict):

        '''
        Adds occupation basis observables to the STIM circuit based on the given dictionary of measurements.
        Therefore, is presumed to be run after _destr_stab_measurement_reconstr.

        Args:
            stim_circuit (STIM Circuit): The STIM circuit to append to.
            qubit_measure_dict (dict): The dictionary mapping qubit locations to measurement indices.
            qubit_dict (dict): The dictionary mapping qubit locations to the measurement basis.

        Returns:
            stim_circuit (STIM Circuit): The STIM circuit with added observables.
        '''

        circuit_append = stim.Circuit()
        observable_count = 0
        #adding observables
        circuit_append.append("TICK")
        observable_count=0
        for q_loc in qubit_dict.keys():
            gate = qubit_dict[q_loc]
            if gate == 'Z':
                circuit_append.append("OBSERVABLE_INCLUDE", stim.target_rec(qubit_measure_dict[q_loc]-len(qubit_measure_dict.keys())), observable_count)
                observable_count += 1

        return stim_circuit + circuit_append, observable_count, len(qubit_dict.keys()) #last thing is number of destructive measurements performed

    def _destr_stab_measurement_reconstr(self, stim_circuit, stab_coords, p1 = 0, p2 = 0, psp = 0, pm = 0, pi = 0, flag = False, reconstruction=False):

        '''
        Adds state preparation and (optional) stabilizer reconstruction to the STIM circuit.
        
        Args:
            stim_circuit (STIM Circuit): The STIM circuit to append to.
            stab_coords (list): The stabilizer coordinates, usual use case sets this to the first half of the stabilizers. PanQEC class' stabilizer ordering is in a way that the first half of the stabilizers is the ones we want.
            p1 (float): Probability of a single-qubit Pauli error.
            p2 (float): Probability of a two-qubit Pauli error.
            pm (float): Probability of a measurement error.
            psp (float): Probability of a single-qubit Pauli error during stabilizer postselection.
            pi (float): Probability of idling error.
            flag (bool): Whether to add flag qubits.
            reconstruction (bool): Whether to perform stabilizer reconstruction.

        Returns:
            stim_circuit (STIM Circuit): The STIM circuit with state preparation and stabilizer reconstruction.
            qubit_measure_dict (dict): The dictionary mapping qubit locations to measurement indices. This is used to add observables in a different function.
            qubit_dict (dict): The dictionary mapping qubit locations to the measurement basis.
        '''

        code = self.panqec_code
        
        circuit_prepend = stim.Circuit()
        circuit_append = stim.Circuit()

        #Adding measurements based on stabilizers in every other row
        qubit_dict = {}
        meas_counter = 0
        qubit_measure_dict = {}
        detectors = []

        for i_stab, stab_loc in enumerate(stab_coords):
            stabilizer = code.get_stabilizer(stab_loc)
            
            for q_loc in stabilizer.keys():
                gate = stabilizer[q_loc]
                q_ind = code.qubit_index[q_loc]
                
                #check for overlapping measurement with a different gate
                if q_loc in qubit_dict.keys() and not qubit_dict[q_loc] == gate:
                    print("Error reconstructing stabilizer measurement")
                    return circuit_prepend + stim_circuit + circuit_append, qubit_measure_dict, qubit_dict
                
                # adding a measurement and observable based on the gate in the stabilizer
                if not q_loc in qubit_dict.keys():
                    qubit_dict[q_loc] = gate
                    if gate == 'Z':
                        circuit_append.append("M", [q_ind], pm)
                        qubit_measure_dict[q_loc] = meas_counter
                        meas_counter += 1
                    elif reconstruction:
                        if gate == 'Y':
                            circuit_append.append("S_DAG", [q_ind])
                            if p1:
                                circuit_append.append("DEPOLARIZE1", [q_ind], p1)
                            circuit_append.append("H", [q_ind])
                            if p1:
                                circuit_append.append("DEPOLARIZE1", [q_ind], p1)
                        elif gate=='X':
                            circuit_append.append("H", [q_ind])
                            if p1:
                                circuit_append.append("DEPOLARIZE1", [q_ind], p1)
                        circuit_append.append("M", [q_ind], pm)
                        qubit_measure_dict[q_loc] = meas_counter
                        meas_counter += 1
                    
        
        #adding detectors for destructive stabilizer reconstruction
        if reconstruction:
            for i_stab, stab_loc in enumerate(stab_coords):
                detector = []
                stabilizer = code.get_stabilizer(stab_loc)
                for q_loc in stabilizer.keys():
                    detector.append(qubit_measure_dict[q_loc]-len(qubit_measure_dict.keys()))
                detectors.append(detector)
            for detector in detectors:
                circuit_append.append("DETECTOR", [stim.target_rec(detector[0]), stim.target_rec(detector[1]), 
                                            stim.target_rec(detector[2]), stim.target_rec(detector[3]),
                                            stim.target_rec(detector[4]), stim.target_rec(detector[5]), 
                                            stim.target_rec(detector[6]), stim.target_rec(detector[7])])

        return circuit_prepend + stim_circuit + circuit_append, qubit_measure_dict, qubit_dict
    
    ###FINAL STIM CIRCUIT METHODS###

    def get_stim_circuit(self,
                          stabilizer_reconstruction: bool,
                          non_destructive_stabilizer_measurement_end: bool,
                          virtual_error_detection_rate:float, #vqed_rate should be 0 if virtual error detection is not used, -1 for symmetry expansion
                          flags_in_synd_extraction: bool,
                          type_of_logical_observables: str,
                          global_parity_postselection: bool,
                          efficient_trotter_steps: int, #0 if not the efficient trotter type circuits
                          p1: float, p2: float, psp: float, pi: float, pm: float):
        '''
        Returns a STIM circuit with the given parameters.

        Args:
            stabilizer_reconstruction (bool): Whether to perform stabilizer reconstruction.
            non_destructive_stabilizer_measurement_end (bool): Whether to perform non-destructive stabilizer measurements at the end of the circuit.
            virtual_error_detection_rate (float): The rate at which virtual error detection measurements are added.
            flags_in_synd_extraction (bool): Whether to add flag qubits during stabilizer measurements.
            type_of_logical_observables (str): The type of logical observables to use. Can be "Z" or "efficient".
            global_parity_postselection (bool): Whether to perform global parity postselection.
            p1 (float): The probability of a single-qubit Pauli error.
            p2 (float): The probability of a two-qubit Pauli error.
            psp (float): The probability of a single-qubit Pauli error during stabilizer postselection.
            pi (float): The probability of idling error.
            pm (float): The probability of a measurement error.

        Returns:
            stim.Circuit: The STIM circuit.

        Raises:
            NotImplementedError: If virtual error detection is requested or if a type of logical observables other than "occupation" or "hopping_subset" is requested.

        '''

        code = self.panqec_code
        stab_coords = code.get_stabilizer_coordinates()
        final_circuit = stim.Circuit()

        flag_meas_counter1 = 0
        stab_meas_counter1 = 0
        vqed_measure_counter = 0
        flag_meas_counter2 = 0
        stab_meas_counter2 = 0
        
        ### CODESPACE STATE PREP

        if type_of_logical_observables == 'Z':
            #if stabilizer_reconstruction: #regardless of stabilizer reconstruction, I can prepare the first half in the product state with single qubit rotations and only measure the second half (TODO confirm ???)
            final_circuit, flag_meas_counter1, stab_meas_counter1 = self._second_half_stabilizer_generators_Z_basis_state_prep(p1, p2, pm, psp, pi)
            #else:
                #   final_circuit, flag_meas_counter, stab_meas_counter = self._second_half_stabilizer_generators_Z_basis_state_prep(p1, p2, pm, psp, pi) #
        elif type_of_logical_observables == 'efficient':
            final_circuit, flag_meas_counter1, stab_meas_counter1 = self._all_stabilizer_generators_state_prep(p1, p2, pm, psp, pi, flags=flags_in_synd_extraction)

        else:
            raise NotImplementedError("Only Z and efficient logical observables are implemented.")

        ### LOGICAL CIRCUIT

        if efficient_trotter_steps == 0:
            noisy_logical_circuit, vqed_measure_counter = self.make_logical_circuit(virtual_error_detection_rate, p1, p2, pm, pi, psp, flags_in_synd_extraction)
        else:
            noisy_logical_circuit = self._add_noise_to_logical_stim_circ(self._get_efficient_trotter_stim_circuit(efficient_trotter_steps), p1, p2, pi, pm, psp)
            vqed_measure_counter = 0

        final_circuit += noisy_logical_circuit
        
        
        ### POSTSELECTION AND MEASUREMENTS

        if type_of_logical_observables == "Z":

            if non_destructive_stabilizer_measurement_end:

                if stabilizer_reconstruction:

                    #stabilizer reconstruction for half the stabilizers and non-destructive measurement for the other half
                    
                    final_circuit, flag_meas_counter2, stab_meas_counter2 = self._second_half_stabilizer_generators_detection_circuit(final_circuit, p1, p2, pm, psp, pi, flags_in_synd_extraction, flag_meas_counter1, stab_meas_counter1, vqed_measure_counter)
                    final_circuit, qubit_measure_dict, qubit_dict = self._destr_stab_measurement_reconstr(final_circuit, stab_coords[:(len(stab_coords)//2)], p1, p2, pm, psp, pi, flags_in_synd_extraction, stabilizer_reconstruction)

                    final_circuit, obs_count, destr_meas_count = self._add_observables_from_dict_Z(final_circuit, qubit_measure_dict, qubit_dict)
                
                else:
                    
                    #non-destructive measurement for all stabilizers

                    final_circuit, flag_meas_counter2, stab_meas_counter2 = self._full_stabilizer_generators_detection_circuit(final_circuit, p1, p2, pm, psp, pi, flags_in_synd_extraction, flag_meas_counter1, stab_meas_counter1, vqed_measure_counter)

                    final_circuit, obs_count, destr_meas_count = self._add_simple_measurement_stim_circuit(final_circuit, pm, global_parity_postsel=global_parity_postselection)
                
            elif stabilizer_reconstruction:
                #only stabilizer reconstruction for half stabilizers
                final_circuit, qubit_measure_dict, qubit_dict = self._destr_stab_measurement_reconstr(final_circuit, stab_coords[:(len(stab_coords)//2)], p1, p2, pm, psp, pi, flags_in_synd_extraction, stabilizer_reconstruction)
                final_circuit, obs_count, destr_meas_count = self._add_observables_from_dict_Z(final_circuit, qubit_measure_dict, qubit_dict) #in principle this could also have global parity postselection TODO
            
            else:
                final_circuit, obs_count, destr_meas_count = self._add_simple_measurement_stim_circuit(final_circuit, pm, global_parity_postsel=global_parity_postselection)

        elif type_of_logical_observables == "efficient":

            if non_destructive_stabilizer_measurement_end:
                final_circuit, flag_meas_counter2, stab_meas_counter2 = self._full_stabilizer_generators_detection_circuit(final_circuit, p1, p2, pm, psp, pi, flags_in_synd_extraction, flag_meas_counter1, stab_meas_counter1, vqed_measure_counter)
            
            final_circuit, obs_count, destr_meas_count = self._add_hopping_measure(final_circuit, pm, p1)

        else:
            raise NotImplementedError("Parameters invalid.")

        final_circuit = self._add_X_error_state_prep_error(final_circuit, psp)
        return final_circuit, obs_count, destr_meas_count, flag_meas_counter1, stab_meas_counter1, vqed_measure_counter, flag_meas_counter2, stab_meas_counter2
        
        