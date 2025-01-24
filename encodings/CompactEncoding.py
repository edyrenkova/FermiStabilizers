from FermionSquare import FermionSquare
import panqec
from FermionHamiltonian import FermionHamiltonian
import cirq
import copy
import stim
import stimcirq
import numpy as np
from AbstractEncoding import AbstractEncoding
from openfermion.ops import FermionOperator, QubitOperator
import openfermion as of
from QubitHamiltonian import QubitHamiltonian


class CompactEncoding(AbstractEncoding):
    """
    This class provides methods to encode a fermion Hamiltonian using DK encoding

    Args:
        fermion_hamiltonian (FermionHamiltonian, optional): The fermion Hamiltonian to be encoded. Defaults to None.

    Attributes:
        L (int): The size of the fermion Hamiltonian.
        panqec_code (FermionSquare): The PanQEC code for the fermion Hamiltonian.
        n_qubits (int): The number of qubits in the PanQEC code.
        fermion_ham (FermionHamiltonian): The fermion Hamiltonian.
        encoded_subset_operators (list): The encoded subset operators.
        full_encoded_ham (QubitOperator): The full encoded Hamiltonian.
        num_tensor_bases (None): Placeholder for the number of tensor bases.
        measure_jk_dict (dict): Dictionary for measurement JK.
        stab_indexes_in_order_of_meas (list): List of stabilizer indexes in order of measurement.
    """
   
    ###INITIALIZATION###
    def __init__(self, fermion_hamiltonian: FermionHamiltonian = None):
        
        self.L = int(fermion_hamiltonian.L)
        self.panqec_code = FermionSquare(self.L)
        self.n_qubits = len(self.panqec_code.qubit_index.keys())

        self.fermion_ham = fermion_hamiltonian

        compact_ham_dicts = [self._encode_compact_ham_terms(op) for op in fermion_hamiltonian.hamiltonian_subset]

        self.encoded_subset_operators = [self._get_qubit_operator(op) for op in compact_ham_dicts]

        self.full_encoded_ham = self._encode_ham(fermion_hamiltonian.full_hamiltonian)
        self.full_encoded_ham.compress()
        self.num_tensor_bases = None
        self.measure_jk_dict = {}
        self.stab_indexes_in_order_of_meas = []
    
    ###ENCODED HAMILTONIAN METHODS###

    def get_relevant_observable_counts(self):
        """
        Get the number of Z observables in the encoded Hamiltonian.

        Returns:
            int: The number of Z observables.
            int: The number of X observables (efficient).
            int: number of Ham terms diagonalized in the best tensor basis
        """
        full_encoded_ham = self.full_encoded_ham
        num_Z_terms = np.sum([all([p[1] == 'Z' for p in op]) for op in list(full_encoded_ham.terms)])
        
        return num_Z_terms, None, None
    
    def _encode_ham(self, f_operator: FermionOperator = None):
        """
        Encodes the fermion Hamiltonian with Compact Encoding.

        Ignores coefficients in the Hamiltonian.

        Args:
            f_operator (FermionOperator, optional): The fermion operator to be encoded. Defaults to None.

        Returns:
            QubitOperator: The encoded Hamiltonian.
        """
        compact_ham_dict = self._encode_compact_ham_terms(f_operator)
        return self._get_qubit_operator(compact_ham_dict)

    def _encode_compact_ham_terms(self, f_operator: FermionOperator = None):
        """
        Encodes the fermion Hamiltonian with Compact Encoding.

        Ignores coefficients in the Hamiltonian.

        Args:
            f_operator (FermionOperator, optional): The fermion operator to be encoded. Defaults to None.

        Returns:
            list: The encoded compact operators.
        """
        f_ham_terms = f_operator.terms
        compact_terms = []
        for f_key in f_ham_terms.keys():
            compact_terms.append(self._encode_compact_term(f_key))
        
        # Flattening the list and deleting duplicates, 
        # because encoding individual terms can result in common terms that would be combined in the encoded Hamiltonian.
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
            # Coulomb interaction
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
            # Hopping or chemical potential
            q_coords = dict()
            for f_index, dag in f_term:
                q_coord = self.panqec_code.get_qubit_coordinates()[f_index]
                q_coords[q_coord] = dag

            if len(q_coords.keys()) == 1:
                # Chemical potential (single Z)
                compact_op = {}
                q_coord = list(q_coords.keys())[0]
                compact_op[q_coord] = 'Z'
                compact_ops.append(compact_op)

            # Half of the hopping term
            else:
                q_logicals_dict = self.panqec_code.get_qubit_logicals_dict()

                q_coord1 = list(q_coords.keys())[0]
                q_coord2 = list(q_coords.keys())[1]

                logicals1 = q_logicals_dict[q_coord1]
                logicals2 = q_logicals_dict[q_coord2]

                # Finds the logical operator that contains both qubits
                overlap_dict = [d1 for d1 in logicals1 for d2 in logicals2 if d1 == d2][0]

                compact_op = copy.deepcopy(overlap_dict)
                
                # Decides between E_ij*V_j and E_ij*V_i based on the dag value
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
        Converts encoded operators to QubitOperator.

        Args:
            encoded_ops (list, optional): The encoded operators. Defaults to None.

        Returns:
            QubitOperator: The qubit operator.
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
    
    def get_full_encoded_ham_length(self):
        """
        Get the length of the full encoded Hamiltonian.

        Returns:
            int: The length of the full encoded Hamiltonian.
        """
        return len(list(self.full_encoded_ham.terms.items()))
    
    def _get_num_Z_obs(self):
        """
        Get the number of Z observables in the encoded Hamiltonian.

        Returns:
            int: The number of Z observables.
        """
        num_Z_terms = np.sum([all([p[1] == 'Z' for p in op]) for op in list(self.full_encoded_ham.terms)])
        return num_Z_terms

    def get_best_tensor_meas_basis(self):
        """
        Get the best measurement basis for the given FermionOperator.

        Maximum is in terms of the number of Hamiltonian terms this basis diagonalizes.

        Args:
            full_fermion_hamiltonian (FermionOperator): The FermionOperator to measure (full Hamiltonian).
            This is needed when Hamiltonian subset is provided to the encoding.

        Returns:
            Tuple[str, QubitOperator]: The best measurement basis and the corresponding QubitOperator.
        """
        full_fermion_hamiltonian = self.fermion_ham.full_hamiltonian
        full_encoded_ham = self._encode_ham(full_fermion_hamiltonian)
        full_encoded_ham.compress()
        full_q_ham = QubitHamiltonian(full_encoded_ham)
        meas_bases = full_q_ham.get_meas_bases_sorted(self.fermion_ham.seed)

        # Attempting to avoid all Z measurement here
        for basis, _ in meas_bases:
            if not all(op == 'Z' for op in basis):
                return basis, _

        return meas_bases[0][0], meas_bases[0][1]
    ###NON-UNITARY STATE PREP CIRCUIT METHODS###
    """
    These methods return circuits for non-unitary codespace preparation as well as counters for flag and stabilizer measurements.
    The counts are to be used when creating detectors in the stabilizer generators postselection circuits.
    """
    def _all_stabilizer_generators_state_prep(self, p1=0, p2=0, pm=0, psp=0, pi=0, flags=False):
        """
        Prepare the state for all stabilizer generators.

        Args:
            p1 (float): Probability of single-qubit depolarizing error.
            p2 (float): Probability of two-qubit depolarizing error.
            pm (float): Probability of measurement error.
            psp (float): Probability of state preparation error.
            pi (float): Probability of idling error.
            flags (bool): Whether to include flag qubits.

        Returns:
            stim.Circuit: The prepared circuit.
            int: Number of flag measurements.
            int: Number of stabilizer measurements.
        """
        code = self.panqec_code
        circuit = stim.Circuit()
        qubit_coords = code.get_qubit_coordinates()
        num_qubits = len(qubit_coords)
        stab_coords = code.get_stabilizer_coordinates()
        num_stabs = len(stab_coords)
        ticktock_dict = self._make_ticktock_dict(0, num_stabs)
        timestep_dict = self._make_timestep_dict(0, num_stabs)
        self.stab_indexes_in_order_of_meas = range(num_stabs)

        if flags:
            for i_stab in range(2*num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        else: 
            for i_stab in range(num_stabs):
                circuit.append("H", [i_stab+num_qubits])
                circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        
        circuit.append("TICK")

        for timestep in timestep_dict.keys():
            for q_loc in timestep_dict[timestep].keys():
                gate = timestep_dict[timestep][q_loc][0]
                q_ind = code.qubit_index[q_loc]
                i_stab = timestep_dict[timestep][q_loc][1]
                circuit.append("C"+gate, [i_stab+num_qubits, q_ind])
                if p2:
                    circuit.append("DEPOLARIZE2", [i_stab+num_qubits, q_ind], p2)
            circuit.append("TICK")
            
            if flags and (timestep == 0 or timestep == 6):
                for i_stab in range(num_stabs):
                    circuit.append("CZ", [i_stab+num_qubits, i_stab+num_qubits+num_stabs])
                    if p2:
                        circuit.append("DEPOLARIZE2",[i_stab+num_qubits, i_stab+num_qubits+num_stabs], p2)
            elif flags:
                for i_stab in range(num_stabs):
                    if pi:
                        circuit.append("DEPOLARIZE1",[i_stab+num_qubits+num_stabs], pi)
            for i_q, q_loc1 in enumerate(qubit_coords):
                if not timestep in ticktock_dict[q_loc1]:
                    if pi:
                        circuit.append("DEPOLARIZE1", [i_q], pi)
            circuit.append("TICK")
        
        flags_meas_counter = 0
        stab_meas_counter = 0
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
        
        if flags:
            for i_flag in range(num_stabs):
                circuit.append("DETECTOR", [stim.target_rec(-1-i_flag)])
                flags_meas_counter += 1
            circuit.append("TICK")
        
        stab_meas_counter = num_stabs

        return circuit, flags_meas_counter, stab_meas_counter

    def _all_stabilizer_generators_Z_basis_state_prep(self, p1=0, p2=0, pm=0, psp=0, pi=0):
        """
        Prepare the state for all stabilizer generators in the Z basis.

        Args:
            p1 (float): Probability of single-qubit depolarizing error.
            p2 (float): Probability of two-qubit depolarizing error.
            pm (float): Probability of measurement error.
            psp (float): Probability of state preparation error.
            pi (float): Probability of idling error.

        Returns:
            stim.Circuit: The prepared circuit.
            int: Number of flag measurements.
            int: Number of stabilizer measurements.
        """
        code = self.panqec_code
        circuit = stim.Circuit()
        qubit_coords = code.get_qubit_coordinates()
        num_qubits = len(qubit_coords)
        stab_coords = code.get_stabilizer_coordinates()
        num_stabs = len(stab_coords)
        ticktock_dict = self._make_ticktock_dict(0, num_stabs)
        timestep_dict = self._make_timestep_dict(0, num_stabs)
        self.stab_indexes_in_order_of_meas = range(num_stabs)

        for i_stab in range(num_stabs):
            circuit.append("H", [i_stab+num_qubits])
            if p1:
                circuit.append('DEPOLARIZE1', [i_stab+num_qubits], p1)
        
        circuit.append("TICK")

        for timestep in timestep_dict.keys():
            if (timestep >= 0 and timestep < 4):
                continue
            for q_loc in timestep_dict[timestep].keys():
                gate = timestep_dict[timestep][q_loc][0]
                q_ind = code.qubit_index[q_loc]
                i_stab = timestep_dict[timestep][q_loc][1]
                circuit.append("C"+gate, [i_stab+num_qubits, q_ind])
                if p2:
                    circuit.append("DEPOLARIZE2", [i_stab+num_qubits, q_ind], p2)
            circuit.append("TICK")
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
        Prepare the state for the second half of the stabilizer generators.

        Args:
            p1 (float): Probability of single-qubit depolarizing error.
            p2 (float): Probability of two-qubit depolarizing error.
            pm (float): Probability of measurement error.
            psp (float): Probability of state preparation error.
            pi (float): Probability of idling error.
            flags (bool): Whether to include flag qubits.

        Returns:
            stim.Circuit: The prepared circuit.
            int: Number of flag measurements.
            int: Number of stabilizer measurements.
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
        self.stab_indexes_in_order_of_meas = range(num_stabs//2, num_stabs)

        qubit_dict = {}
        for i_stab, stab_loc in enumerate(stab_coords_first_half):
            stabilizer = code.get_stabilizer(stab_loc)
            for q_loc in stabilizer.keys():
                gate = stabilizer[q_loc]
                q_ind = code.qubit_index[q_loc]
                if q_loc in qubit_dict.keys() and not qubit_dict[q_loc] == gate:
                    print("Error creating state prep circuit")
                    return qubit_dict
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

        for timestep in timestep_dict.keys():
            for q_loc in timestep_dict[timestep].keys():
                gate = timestep_dict[timestep][q_loc][0]
                q_ind = code.qubit_index[q_loc]
                i_stab = timestep_dict[timestep][q_loc][1]
                circuit.append("C"+gate, [i_stab+num_qubits, q_ind])
                if p2:
                    circuit.append("DEPOLARIZE2", [i_stab+num_qubits, q_ind], p2)
            circuit.append("TICK")
            if flags and (timestep == 0 or timestep == 6):
                for i_stab in range(num_stabs):
                    circuit.append("CZ", [i_stab+num_qubits, i_stab+num_qubits+num_stabs])
                    if p2:
                        circuit.append("DEPOLARIZE2",[i_stab+num_qubits, i_stab+num_qubits+num_stabs], p2)
            elif flags:
                for i_stab in range(num_stabs):
                    if pi:
                        circuit.append("DEPOLARIZE1",[i_stab+num_qubits+num_stabs], pi)
            for i_q, q_loc1 in enumerate(qubit_coords):
                if not timestep in ticktock_dict[q_loc1]:
                    if pi:
                        circuit.append("DEPOLARIZE1", [i_q], pi)
            circuit.append("TICK")
        
        flags_meas_counter = 0
        stab_meas_counter = 0
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
        
        if flags:
            for i_flag in range(num_stabs):
                circuit.append("DETECTOR", [stim.target_rec(-1-i_flag)])
                flags_meas_counter += 1
        circuit.append("TICK")

        return circuit, flags_meas_counter, stab_meas_counter

    def _second_half_stabilizer_generators_Z_basis_state_prep(self, p1=0, p2=0, pm=0, psp=0, pi=0):
        """
        Prepare the state for the second half of the stabilizer generators in the Z basis.

        Args:
            p1 (float): Probability of single-qubit depolarizing error.
            p2 (float): Probability of two-qubit depolarizing error.
            pm (float): Probability of measurement error.
            psp (float): Probability of state preparation error.
            pi (float): Probability of idling error.

        Returns:
            stim.Circuit: The prepared circuit.
            int: Number of flag measurements.
            int: Number of stabilizer measurements.
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
        self.stab_indexes_in_order_of_meas = range(num_stabs//2, num_stabs)
        for i_stab, stab_loc in enumerate(stab_coords_first_half):
            stabilizer = code.get_stabilizer(stab_loc)
            for q_loc in stabilizer.keys():
                gate = stabilizer[q_loc]
                q_ind = code.qubit_index[q_loc]
                if q_loc in qubit_dict.keys() and not qubit_dict[q_loc] == gate:
                    print("Error creating state prep circuit")
                    return qubit_dict
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

        for timestep in timestep_dict.keys():
            if (timestep >= 0 and timestep < 4):
                continue
            for q_loc in timestep_dict[timestep].keys():
                gate = timestep_dict[timestep][q_loc][0]
                q_ind = code.qubit_index[q_loc]
                i_stab = timestep_dict[timestep][q_loc][1]
                circuit.append("C"+gate, [i_stab+num_qubits, q_ind])
                if p2:
                    circuit.append("DEPOLARIZE2", [i_stab+num_qubits, q_ind], p2)
            circuit.append("TICK")
            if pi:
                for i_q, q_loc1 in enumerate(qubit_coords):
                    if not timestep in ticktock_dict[q_loc1]:
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
