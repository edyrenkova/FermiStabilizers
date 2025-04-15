import os
import git
import pandas as pd
from encoding.CompactEncoding import CompactEncoding
from hamiltonians.FermionHamiltonian import FermionHamiltonian
from encoding.TernaryEncoding import TernaryEncoding
from encoding.JWEncoding import JWEncoding
import numpy as np
import stim
import time
from scipy.stats import bootstrap
import copy

class Simulation:
    def __init__(self, data_frame: pd.Series, output_folder: str):
        """
        Initializes a Simulation object.

        Args:
            data_frame (pd.DataFrame): The input data frame containing simulation parameters.
            rand_seeds (list[int]): The random seeds, passed from run_simulation where it's read from an .npy file.
            output_folder (str): The output folder.
        """


        self.lattice_size = int(data_frame['lattice_size'])
        self.fermion_hamiltonian_descr = data_frame['fermion_hamiltonian_descr']#.iloc[0]
        self.boundary_conditions = data_frame['boundary_conditions']#.iloc[0]
        self.rand_seed = int(data_frame['rand_seed'])

        
        self.logical_operators_depth = data_frame['logical_operators_depth']#.iloc[0]
        self.efficient_trotter_steps = data_frame['efficient_trotter_steps']#.iloc[0]

        self.encoding = data_frame['encoding']#.iloc[0]
        self.virtual_error_detection_rate = data_frame['virtual_error_detection_rate']#.iloc[0]
        self.stabilizer_reconstruction = data_frame['stabilizer_reconstruction']#.iloc[0]
        self.flags_in_synd_extraction = data_frame['flags_in_synd_extraction']#.iloc[0]
        self.type_of_logical_observables = data_frame['type_of_logical_observables']#.iloc[0]
        self.global_parity_postselection = data_frame['global_parity_postselection']#.iloc[0]
        self.non_destructive_stabilizer_measurement_end = data_frame['non_destructive_stabilizer_measurement_end']#.iloc[0]

        self.pm = data_frame['pm']#.iloc[0]
        self.psp = data_frame['psp']#.iloc[0]
        self.p1 = data_frame['p1']#.iloc[0]
        self.p2 = data_frame['p2']#.iloc[0]
        self.pi = data_frame['pi']#.iloc[0]

        self.error_model = str(data_frame['error_model'])#.iloc[0])

        self.n_shots = data_frame['n_shots']#.iloc[0]
        self.output_folder = output_folder

        self.bootstrap_resamples = data_frame['bootstrap_resamples']#.iloc[0]

        self.filename_prefix = f"{self.encoding}{self.lattice_size}{int(self.logical_operators_depth*100)}{int(self.efficient_trotter_steps*100)}{int(self.virtual_error_detection_rate*100)}{int(self.stabilizer_reconstruction)}{int(self.flags_in_synd_extraction)}{int(self.global_parity_postselection)}{int(self.non_destructive_stabilizer_measurement_end)}{self.type_of_logical_observables}{str(self.error_model).replace(', ', '').replace('(', 'm').replace(')','m')}{int(self.p2*10000)}"
        self.output_data_folder = os.path.join(output_folder, self.filename_prefix)
    
    def run(self):
        """
        Runs the simulation.
        """
        rand_seed = self.rand_seed
        tic = time.time()
        fh = self._initialize_fermion_hamiltonian(rand_seed)
        toc = time.time()
        self.time_to_initialize_fermion_hamiltonian = toc-tic
        print(f"Time to initialize Fermion Hamiltonian: {toc-tic} seconds")
        
        tic = time.time()
        circuit_params = self._make_stim_circuit(fh)

        stim_circuit, obs_count = circuit_params[0], circuit_params[1]
        detector_number, destr_meas_count= circuit_params[2], circuit_params[3]
        flag_meas_counter1, stab_meas_counter1 = circuit_params[4], circuit_params[5]
        vqed_measure_counter, flag_meas_counter2, stab_meas_counter2 = circuit_params[6], circuit_params[7], circuit_params[8]
        toc = time.time()
        print(f"Time to make stim circuit: {toc-tic} seconds")
        self.time_to_make_stim_circuit = toc-tic

        tic = time.time()

        if self.virtual_error_detection_rate == 0:
            postsel_any_observable_arr, any_detection_arr, obs_data_for_triv_syndrome = self._sample_stim_circuit_detector_compiler(stim_circuit)
            bootstrapped_any_logical_error_rate, raw_any_logical_error_rate, bootstrapped_local_logical_error_rates, raw_local_logical_error_rates, bootstrapped_detection_rates, raw_any_detection_rate = self._bootstrap_no_vqed(any_detection_arr, 
                                                                                                    postsel_any_observable_arr, obs_data_for_triv_syndrome)
        else:
            obs_est_samples, v_s, any_detection_arr = self._sample_stim_circuit_vqed(self.q_hamiltonian)
            bootstrapped_any_logical_error_rate, raw_any_logical_error_rate, bootstrapped_local_logical_error_rates, raw_local_logical_error_rates, bootstrapped_detection_rates, raw_any_detection_rate = self._bootstrap_vqed(obs_est_samples, v_s, any_detection_arr)
            

        toc = time.time()
        self.time_to_sample_stim_circuit = toc-tic
        print(f"Time to sample stim circuit: {toc-tic} seconds")
        
        repo = git.Repo(search_parent_directories=True)

        # Save results to numpy files
        os.makedirs(self.output_data_folder, exist_ok=True)

        # Save raw and bootstrapped logical error rates
        np.save(os.path.join(self.output_data_folder, f"raw_local_logical_error_rates_{rand_seed}.npy"), raw_local_logical_error_rates)
        np.save(os.path.join(self.output_data_folder, f"raw_any_logical_error_rate_{rand_seed}.npy"), raw_any_logical_error_rate)
        np.save(os.path.join(self.output_data_folder, f"raw_any_detection_rate_{rand_seed}.npy"), raw_any_detection_rate)

        # Save additional statistics for bootstrapping
        if bootstrapped_local_logical_error_rates is not None:
            np.save(os.path.join(self.output_data_folder, f"bootstrapped_local_logical_error_rates_confidence_interval_{rand_seed}.npy"), [res.confidence_interval for res in bootstrapped_local_logical_error_rates])
            np.save(os.path.join(self.output_data_folder, f"bootstrapped_local_logical_error_rates_standard_error_{rand_seed}.npy"), [res.standard_error for res in bootstrapped_local_logical_error_rates])

        if bootstrapped_any_logical_error_rate is not None:
            np.save(os.path.join(self.output_data_folder, f"bootstrapped_any_logical_error_rate_confidence_interval_{rand_seed}.npy"), bootstrapped_any_logical_error_rate.confidence_interval)
            np.save(os.path.join(self.output_data_folder, f"bootstrapped_any_logical_error_rate_standard_error_{rand_seed}.npy"), bootstrapped_any_logical_error_rate.standard_error)

        if bootstrapped_detection_rates is not None:
            np.save(os.path.join(self.output_data_folder, f"bootstrapped_detection_rates_confidence_interval_{rand_seed}.npy"), bootstrapped_detection_rates.confidence_interval)
            np.save(os.path.join(self.output_data_folder, f"bootstrapped_detection_rates_standard_error_{rand_seed}.npy"), bootstrapped_detection_rates.standard_error)

        output_data = {
            'lattice_size': [self.lattice_size],
            'fermion_hamiltonian_descr': [self.fermion_hamiltonian_descr],
            'boundary_conditions': [self.boundary_conditions],
            'logical_operators_depth': [self.logical_operators_depth],
            'efficient_trotter_steps': [self.efficient_trotter_steps],
            'encoding': [self.encoding],
            'virtual_error_detection_rate': [self.virtual_error_detection_rate],
            'stabilizer_reconstruction': [self.stabilizer_reconstruction],
            'flags_in_synd_extraction': [self.flags_in_synd_extraction],
            'type_of_logical_observables': [self.type_of_logical_observables],
            'global_parity_postselection': [self.global_parity_postselection],
            'non_destructive_stabilizer_measurement_end': [self.non_destructive_stabilizer_measurement_end],
            'p1': [self.p1],
            'p2': [self.p2],
            'pm': [self.pm],
            'psp': [self.psp],
            'pi': [self.pi],
            'error_model': [self.error_model],

            'n_shots': [self.n_shots],

            'rand_seed': [rand_seed],
            'bootstrap_resamples': [self.bootstrap_resamples],  

            'git_commit': [repo.head.object.hexsha],
            'timestamp': [pd.Timestamp.now()],

            'circuit_file': [self.circuit_file],

            'num_qubits': [self.n_qubits],
            'num_ticks': [self.n_ticks],
            'fermion_hamiltonian_length': [self.hamiltonian_length],
            'full_encoded_ham_length': [self.full_encoded_ham_length],
            'circuit_obs_count': [obs_count],
            'circuit_detector_count': [detector_number],

            'Z_observable_count': [self.Z_obs_count], 
            'efficient_observable_count': [self.efficient_obs_count], 
            'time_to_initialize_fermion_hamiltonian': [self.time_to_initialize_fermion_hamiltonian], 
            'time_to_make_stim_circuit': [self.time_to_make_stim_circuit],
            'time_to_sample_stim_circuit': [self.time_to_sample_stim_circuit],  

            'detection_rate_raw': [os.path.join(self.output_data_folder, f"raw_any_detection_rate_{rand_seed}.npy")],
            'detection_rate_CI': [os.path.join(self.output_data_folder, f"bootstrapped_detection_rates_confidence_interval_{rand_seed}.npy")],
            'detection_rate_bootstrap_std': [os.path.join(self.output_data_folder, f"bootstrapped_detection_rates_standard_error_{rand_seed}.npy")],

            'any_obs_rate_raw': [os.path.join(self.output_data_folder, f"raw_any_logical_error_rate_{rand_seed}.npy")],
            'any_obs_CI': [os.path.join(self.output_data_folder, f"bootstrapped_any_logical_error_rate_confidence_interval_{rand_seed}.npy")],
            'any_obs_bootstrap_std': [os.path.join(self.output_data_folder, f"bootstrapped_any_logical_error_rate_standard_error_{rand_seed}.npy")],

            'local_obs_rate_raw': [os.path.join(self.output_data_folder, f"raw_local_logical_error_rates_{rand_seed}.npy")],
            'local_obs_CI': [os.path.join(self.output_data_folder, f"bootstrapped_local_logical_error_rates_confidence_interval_{rand_seed}.npy")],
            'local_obs_bootstrap_std': [os.path.join(self.output_data_folder, f"bootstrapped_local_logical_error_rates_standard_error_{rand_seed}.npy")]
            }
        
        self.output_data_frame = pd.DataFrame(output_data)
        return self.output_data_frame
            
    def record_results(self, output_filename:str, mode='a'):
        """
        Records and compares the simulation results.
        """
        output_filename = self.output_folder + "/" +output_filename
        # Check if file exists
        if not os.path.exists(output_filename):
        # Create the file and write the column names
            with open(output_filename, 'w') as f:
                column_names = ['lattice_size', 'fermion_hamiltonian_descr',
                    'boundary_conditions',
                    'logical_operators_depth', 'efficient_trotter_steps',
                    'encoding',
                    'virtual_error_detection_rate',
                    'stabilizer_reconstruction',
                    'flags_in_synd_extraction', 'type_of_logical_observables',
                    'global_parity_postselection', 'non_destructive_stabilizer_measurement_end',
                    'p1', 'p2', 'pm', 'psp', 'pi', 'error_model', 'n_shots',

                    'rand_seed', 
                    'bootstrap_num_resamples',

                    'git_commit', 'timestamp',

                    'circuit_file',

                    'num_qubits', 'num_ticks',
                    'fermion_hamiltonian_length', 'full_encoded_ham_length',
                    'circuit_obs_count', 'circuit_detector_count',
                    'Z_observable_count', 'efficient_observable_count',

                    'time_to_initialize_fermion_hamiltonian', 'time_to_make_stim_circuit', 'time_to_sample_stim_circuit',

                    'detection_rate_raw', 'detection_rate_CI', 'detection_rate_bootstrap_std',
                    'any_obs_rate_raw', 'any_obs_CI', 'any_obs_bootstrap_std',
                    'local_obs_rate_raw', 'local_obs_CI', 'local_obs_bootstrap_std'
                    
                    ]

                f.write(','.join(column_names) + '\n')
        
        self.output_data_frame.to_csv(output_filename, mode=mode, header=False, index=False)
        return output_filename

    def _initialize_fermion_hamiltonian(self, rand_seed: int): 
        """
        Initializes a FermionHamiltonian object based on the provided description and seed.

        FermionHamiltonian object now includes both the full Hamiltonian and a subset of terms.
        """
        
        if self.fermion_hamiltonian_descr == 'fermi_hubbard' and self.boundary_conditions == 'periodic':
            fh = FermionHamiltonian(True, self.lattice_size, True, rand_seed, self.logical_operators_depth)
            self.hamiltonian_length = fh.full_ham_length

            #vqed interval (how often vqed is done) provided in input parameters is a fraction, so we need to multiply it by the effective logical depth (number of logical operators applied in the circuit)
            # if self.virtual_error_detection_rate == -1:
            #     self.vqed_interval = -1
            # else:
            #     self.effective_logical_depth = fh.rand_op_num*2 #times 2 for the mirror ciruit
            #     self.vqed_interval = int(self.virtual_error_detection_rate*self.effective_logical_depth) 
        elif self.fermion_hamiltonian_descr == 'fermi_hubbard' and self.boundary_conditions == 'closed':
            fh = FermionHamiltonian(True, self.lattice_size, True, rand_seed, self.logical_operators_depth, periodic=False)
            self.hamiltonian_length = fh.full_ham_length
        else:
            raise ValueError(f"Fermion Hamiltonian description {self.fermion_hamiltonian_descr} are not supported.")
        return fh
    
    def _make_stim_circuit(self, fh: FermionHamiltonian):
        
        if self.encoding == 'JW':
            q_hamiltonian = JWEncoding(fh)
        elif self.encoding == 'C':
            q_hamiltonian = CompactEncoding(fh)
        elif self.encoding == 'Ternary':
            q_hamiltonian = TernaryEncoding(fh)
        else:
            raise ValueError(f"Encoding {self.encoding} not supported.")
        self.q_hamiltonian = q_hamiltonian
        self.Z_obs_count, self.efficient_obs_count =\
             q_hamiltonian._get_relevant_observable_counts()
        
        self.full_encoded_ham_length = q_hamiltonian._get_full_encoded_ham_length()

                
        stim_circuit, obs_count, destr_meas_count, flag_meas_counter1, stab_meas_counter1, vqed_measure_counter, flag_meas_counter2, stab_meas_counter2 = q_hamiltonian.get_stim_circuit(
                                 stabilizer_reconstruction=self.stabilizer_reconstruction,
                                 non_destructive_stabilizer_measurement_end=self.non_destructive_stabilizer_measurement_end,
                                 virtual_error_detection_rate=self.virtual_error_detection_rate,
                                 flags_in_synd_extraction=self.flags_in_synd_extraction,
                                 type_of_logical_observables=self.type_of_logical_observables,
                                 global_parity_postselection=self.global_parity_postselection,
                                 efficient_trotter_steps=self.efficient_trotter_steps,
                                 p1=self.p1, p2=self.p2,
                                 psp=self.psp, pi=self.pi,
                                 pm=self.pm)
        if self.encoding == 'C' and self.virtual_error_detection_rate:
            self.measure_jk_dict = q_hamiltonian.measure_jk_dict #{measure_counter: (j, k)} - j, k are indexes of j and k stabilizers for vqed
            self.stab_indexes_in_order_of_meas = q_hamiltonian.stab_indexes_in_order_of_meas #indexes of stabilizers in the order they are measured in the state prep
        if self.encoding == 'C' and self.type_of_logical_observables == 'efficient':
            self.efficient_obs_count = obs_count
        
        self.circuit_obs_count = obs_count

        detector_number = stim_circuit.num_detectors
        #this should error if there are non-deterministic detectors/observables
        stim_circuit.detector_error_model()

        self.circuit_file = self._save_circuit_file(stim_circuit, fh.seed)

        self.n_qubits = stim_circuit.num_qubits
        self.n_ticks = 0 #stim_circuit.num_ticks

        return stim_circuit, obs_count, detector_number, destr_meas_count, flag_meas_counter1, stab_meas_counter1, vqed_measure_counter, flag_meas_counter2, stab_meas_counter2
        
    def _save_circuit_file(self, stim_circuit, rand_seed):
        os.makedirs(self.output_data_folder, exist_ok=True)
        circuit_file = f"{self.output_data_folder}/circuit_{rand_seed}.stim"
        stim_circuit.to_file(circuit_file)
        return circuit_file


    def _sample_stim_circuit_detector_compiler(self, stim_circuit:stim.Circuit):
        """
        Samples the stim circuit and returns the logical errors and syndrome count.
        
        Returns:
            np.ndarray: Postselected shot array where True means at least one logical observable was flipped, False - none were flipped.
            np.ndarray: Shot array where True means at least one detector was flipped, False - none were flipped.
        """
        
        sampler = stim_circuit.compile_detector_sampler()
        
        det_data, obs_data = sampler.sample(shots=self.n_shots,separate_observables=True)
        
        any_detection_arr = np.any(det_data, axis = 1)

        obs_data_for_triv_syndrome = obs_data[~any_detection_arr]
        
        postsel_any_observable_arr = np.any(obs_data_for_triv_syndrome, axis = 1)

        return postsel_any_observable_arr, any_detection_arr, obs_data_for_triv_syndrome
    

    def _sample_stim_circuit_vqed(self, q_hamiltonian):
        """
        
        Samples the stim circuit and returns the logical errors and syndrome count.
        
        Returns:
            obs_est_samples: Per shot observable estimates (either -1 or 1) based on the observable measurements and random stabilizer measuremetns
            v_s: Per shot product of random stabilizer measurements.
            any_detection_arr: Per shot array where True means at least one detector was flipped, False - none were flipped.
        """
        v_s = []
        obs_est_samples = []
        any_detection_arr = []
        for i in range(self.n_shots):
            stim_circuit, obs_count, destr_meas_count, flag_meas_counter1, stab_meas_counter1, vqed_measure_counter, flag_meas_counter2, stab_meas_counter2 = q_hamiltonian.get_stim_circuit(
                                    stabilizer_reconstruction=self.stabilizer_reconstruction,
                                    non_destructive_stabilizer_measurement_end=self.non_destructive_stabilizer_measurement_end,
                                    virtual_error_detection_rate=self.virtual_error_detection_rate,
                                    flags_in_synd_extraction=self.flags_in_synd_extraction,
                                    type_of_logical_observables=self.type_of_logical_observables,
                                    global_parity_postselection=self.global_parity_postselection,
                                    p1=self.p1, p2=self.p2,
                                    psp=self.psp, pi=self.pi,
                                    pm=self.pm)
            
            if self.encoding == 'C' and self.virtual_error_detection_rate:
                self.measure_jk_dict = q_hamiltonian.measure_jk_dict #{measure_counter: (j, k)} - j, k are indexes of j and k stabilizers for vqed
                
                self.stab_indexes_in_order_of_meas = q_hamiltonian.stab_indexes_in_order_of_meas #indexes of stabilizers in the order they are measured in the state prep
            
            meas_sampler = stim_circuit.compile_sampler(skip_reference_sample=True)

            sample = meas_sampler.sample(shots=1)[0]


            #defining partitions of measurements in the samples array
            start_stab_meas_start = 0
            start_stab_meas_end = stab_meas_counter1
            start_flag_meas_start = stab_meas_counter1
            start_flag_meas_end = stab_meas_counter1+flag_meas_counter1
            vqed_meas_start = stab_meas_counter1+flag_meas_counter1
            vqed_meas_end = vqed_meas_start + vqed_measure_counter
            end_stab_meas_start = vqed_meas_end
            end_stab_meas_end = vqed_meas_end + stab_meas_counter2
            end_flag_meas_start = end_stab_meas_end
            end_flag_meas_end = end_stab_meas_end + flag_meas_counter2
            

        
            vqed_m_results = sample[vqed_meas_start:vqed_meas_end]

            #####################POSTSELECTION########################
            if stab_meas_counter1:
                start_stab_m_results = sample[start_stab_meas_start:start_stab_meas_end]

            if stab_meas_counter2:
                end_stab_m_results = sample[end_stab_meas_start:end_stab_meas_end]
                if len(end_stab_m_results) == len(start_stab_m_results):
                    if np.any(np.bitwise_xor(end_stab_m_results, start_stab_m_results)):
                        any_detection_arr.append(True)
                        continue
                elif len(end_stab_m_results) == len(start_stab_m_results)*2:
                    if np.any(np.bitwise_xor(end_stab_m_results[len(start_stab_m_results):], start_stab_m_results)) or np.any(end_stab_m_results[:len(start_stab_m_results)]):
                        any_detection_arr.append(True)
                        continue

            if flag_meas_counter1:
                start_flag_m_results = sample[start_flag_meas_start:start_flag_meas_end]
                if np.any(start_flag_m_results):
                    any_detection_arr.append(True)
                    continue
            if flag_meas_counter2:
                end_flag_m_results = sample[end_flag_meas_start:end_flag_meas_end]
                if np.any(end_flag_m_results):
                    any_detection_arr.append(True)
                    continue
            
            vqed_flag_detected = False
            for j in range(vqed_measure_counter):
                if vqed_m_results[j] and self.measure_jk_dict[j] == 'f':
                    vqed_flag_detected = True
                    break
            if vqed_flag_detected:
                any_detection_arr.append(True)
                continue
            
            obs_meas_results = sample[end_flag_meas_end:]

            if self.global_parity_postselection:
                if np.sum(obs_meas_results) % 2 != 0:
                    any_detection_arr.append(True)
                    continue

            #####################END POSTSELECTION########################

            any_detection_arr.append(False)

            if len(obs_meas_results) != destr_meas_count:
                raise ValueError(f"Number of observables measured {len(obs_meas_results)} does not match the expected number {destr_meas_count}")
            
            #combining triples of "edge operators" together
            if self.type_of_logical_observables == 'efficient':
                obs_meas_results_efficient_combined = []
                for i in range(0, len(obs_meas_results), 3):
                    obs_meas_results_efficient_combined.append(obs_meas_results[i] and obs_meas_results[i+1] and obs_meas_results[i+2])
                obs_meas_results = obs_meas_results_efficient_combined
            

            #add up results from the vqed measurements, a_j in the paper where j counts samples
            #multiply by -1 if the corresponding stabilizer measurements are different
            v_s_i = 1
            
            for v_i in range(len(vqed_m_results)):
                if self.measure_jk_dict[v_i] != 'f':
                    v_s_i *= 1 if not vqed_m_results[v_i] else (-1)
                    if self.measure_jk_dict[v_i][0] in self.stab_indexes_in_order_of_meas:                    
                        j_state_prep_res = 1 if not start_stab_m_results[self.stab_indexes_in_order_of_meas.index(self.measure_jk_dict[v_i][0])] else -1
                    else:
                        j_state_prep_res = 1
                    if self.measure_jk_dict[v_i][1] in self.stab_indexes_in_order_of_meas:
                        k_state_prep_res = 1 if not start_stab_m_results[self.stab_indexes_in_order_of_meas.index(self.measure_jk_dict[v_i][1])] else -1
                    else:
                        k_state_prep_res = 1
                    v_s_i *= (j_state_prep_res*k_state_prep_res)
                        
            v_s.append(v_s_i)
                
            obs_est_sample = []
            for obs_res in obs_meas_results:
                obs_est_sample.append((1 if not obs_res else -1)*v_s_i)
            #in our circuits we expect obs_est_sample values to be close to +1 -> we expect to return to |0> state

            obs_est_samples.append(obs_est_sample)


        return obs_est_samples, v_s, any_detection_arr

    
    def _bootstrap_no_vqed(self, any_detection_arr, any_obs_arr, obs_data_for_triv_syndrome):
        """
        Estimates the logical error rate using bootstrap.

        """
        
        bootstrapped_local_logical_error_rates = [] #local obs error rates
        bootstrapped_any_logical_error_rate = []
        raw_logical_error_rates = [] #local obs error rates
        

        if len(any_obs_arr) < 10:
            bootstrapped_local_logical_error_rates = None
            bootstrapped_any_logical_error_rate = None
            raw_any_logical_error_rate = 0
        else:
            bootstrapped_any_logical_error_rate = bootstrap(data=(any_obs_arr,), statistic=np.mean, n_resamples=self.bootstrap_resamples, method='basic')
            raw_any_logical_error_rate = np.mean(any_obs_arr)
            
            for i in range(len(obs_data_for_triv_syndrome[0])):
                bootstrapped_local_logical_error_rates.append(bootstrap(data=(obs_data_for_triv_syndrome[:,i],), statistic=np.mean, n_resamples=self.bootstrap_resamples, method='basic', vectorized=False))
                raw_logical_error_rates.append(np.mean(obs_data_for_triv_syndrome[:,i]))

        if len(any_detection_arr) == 0:
            bootstrapped_detection_rates = None
            raw_any_detection_rate = 0
        else:
            bootstrapped_detection_rates = bootstrap(data=(any_detection_arr,), statistic=np.mean, n_resamples=self.bootstrap_resamples, method='basic')
            raw_any_detection_rate = np.mean(any_detection_arr)
        
        return bootstrapped_any_logical_error_rate, raw_any_logical_error_rate, bootstrapped_local_logical_error_rates, raw_logical_error_rates, bootstrapped_detection_rates, raw_any_detection_rate

    def _bootstrap_vqed(self, obs_est_samples, v_s, any_detection_arr):
        """
        Estimates the logical error rate using bootstrap.

        """
        def vqed_estimate(obs_est_samples, v_s, axis=-1):
            """
            Estimates the logical error rate using VQED.

            Return:
                obs_est_error: 
            """
            #obs_est_sums_norm = (np.sum(obs_est_samples)/np.sum(v_s)+1)/2
            #obs_est_error = 1 - obs_est_sums_norm
            b_s = np.sum(obs_est_samples,axis)
            a_s = np.sum(v_s,axis)
            if a_s == 0:
                obs_est_error = 0
            else:
                obs_est_error = 1 - (float(b_s)/float(a_s)+1)/2
            return obs_est_error
        

        
        bootstrapped_local_logical_error_rates = []
        raw_local_logical_error_rates = []
        if len(obs_est_samples) < 10:
            bootstrapped_local_logical_error_rates = None
        else:
            for i in range(len(obs_est_samples[0])):
                bootstrapped_local_logical_error_rates.append(bootstrap(data=(np.array(obs_est_samples)[:,i], v_s), statistic=vqed_estimate, n_resamples=self.bootstrap_resamples, method='basic', paired=True, vectorized=False))
                raw_local_logical_error_rates.append(vqed_estimate(np.array(obs_est_samples)[:,i], v_s))
        
        if len(any_detection_arr) == 0:
            bootstrapped_detection_rates = None
            raw_any_detection_rate = 0
        else:
            bootstrapped_detection_rates = bootstrap(data=(any_detection_arr,), statistic=np.mean, n_resamples=self.bootstrap_resamples, method='basic')
            raw_any_detection_rate = np.mean(any_detection_arr)

        bootstrapped_any_logical_error_rate = None
        raw_any_logical_error_rate = 0
        return bootstrapped_any_logical_error_rate, raw_any_logical_error_rate, bootstrapped_local_logical_error_rates, raw_local_logical_error_rates, bootstrapped_detection_rates, raw_any_detection_rate



   
def sample_shot(i, q_hamiltonian, stabilizer_reconstruction, non_destructive_stabilizer_measurement_end, virtual_error_detection_rate, flags_in_synd_extraction, type_of_logical_observables, global_parity_postselection, p1, p2, psp, pi, pm):
    stim_circuit, obs_count, destr_meas_count, flag_meas_counter1, stab_meas_counter1, vqed_measure_counter, flag_meas_counter2, stab_meas_counter2 = q_hamiltonian.get_stim_circuit(
                            stabilizer_reconstruction=stabilizer_reconstruction,
                            non_destructive_stabilizer_measurement_end=non_destructive_stabilizer_measurement_end,
                            virtual_error_detection_rate=virtual_error_detection_rate,
                            flags_in_synd_extraction=flags_in_synd_extraction,
                            type_of_logical_observables=type_of_logical_observables,
                            global_parity_postselection=global_parity_postselection,
                            p1=p1, p2=p2,
                            psp=psp, pi=pi,
                            pm=pm)
    
    measure_jk_dict = copy.deepcopy(q_hamiltonian.measure_jk_dict) #{measure_counter: (j, k)} - j, k are indexes of j and k stabilizers for vqed
        
    stab_indexes_in_order_of_meas = copy.deepcopy(q_hamiltonian.stab_indexes_in_order_of_meas) #indexes of stabilizers in the order they are measured in the state prep
    
    meas_sampler = stim_circuit.compile_sampler(skip_reference_sample=True)

    sample = meas_sampler.sample(shots=1)[0]


    #defining partitions of measurements in the samples array
    start_stab_meas_start = 0
    start_stab_meas_end = stab_meas_counter1
    start_flag_meas_start = stab_meas_counter1
    start_flag_meas_end = stab_meas_counter1+flag_meas_counter1
    vqed_meas_start = stab_meas_counter1+flag_meas_counter1
    vqed_meas_end = vqed_meas_start + vqed_measure_counter
    end_stab_meas_start = vqed_meas_end
    end_stab_meas_end = vqed_meas_end + stab_meas_counter2
    end_flag_meas_start = end_stab_meas_end
    end_flag_meas_end = end_stab_meas_end + flag_meas_counter2
    


    vqed_m_results = sample[vqed_meas_start:vqed_meas_end]

    #####################POSTSELECTION########################
    if stab_meas_counter1:
        start_stab_m_results = sample[start_stab_meas_start:start_stab_meas_end]

    if stab_meas_counter2:
        end_stab_m_results = sample[end_stab_meas_start:end_stab_meas_end]
        if len(end_stab_m_results) == len(start_stab_m_results):
            if np.any(np.bitwise_xor(end_stab_m_results, start_stab_m_results)):
                any_detection = True
                return any_detection, None, None
        elif len(end_stab_m_results) == len(start_stab_m_results)*2:
            if np.any(np.bitwise_xor(end_stab_m_results[len(start_stab_m_results):], start_stab_m_results)) or np.any(end_stab_m_results[:len(start_stab_m_results)]):
                any_detection = True
                return any_detection, None, None

    if flag_meas_counter1:
        start_flag_m_results = sample[start_flag_meas_start:start_flag_meas_end]
        if np.any(start_flag_m_results):
            any_detection = True
            return any_detection, None, None
    if flag_meas_counter2:
        end_flag_m_results = sample[end_flag_meas_start:end_flag_meas_end]
        if np.any(end_flag_m_results):
            any_detection = True
            return any_detection, None, None
    
    vqed_flag_detected = False
    for j in range(vqed_measure_counter):
        
        if vqed_m_results[j] and measure_jk_dict[j] == 'f':
            vqed_flag_detected = True
            break
    if vqed_flag_detected:
        any_detection = True
        return any_detection, None, None
    
    obs_meas_results = sample[end_flag_meas_end:]

    if global_parity_postselection:
        if np.sum(obs_meas_results) % 2 != 0:
            any_detection = True
            return any_detection, None, None
        
    #####################END POSTSELECTION########################

    any_detection = False

    if len(obs_meas_results) != destr_meas_count:
        raise ValueError(f"Number of observables measured {len(obs_meas_results)} does not match the expected number {destr_meas_count}")
    
    #combining triples of "edge operators" together
    if type_of_logical_observables == 'efficient':
        obs_meas_results_efficient_combined = []
        for i in range(0, len(obs_meas_results), 3):
            obs_meas_results_efficient_combined.append(obs_meas_results[i] and obs_meas_results[i+1] and obs_meas_results[i+2])
        obs_meas_results = obs_meas_results_efficient_combined
    

    #add up results from the vqed measurements, a_j in the paper where j counts samples
    #multiply by -1 if the corresponding stabilizer measurements are different
    v_s_i = 1
    
    for v_i in range(len(vqed_m_results)):
        if measure_jk_dict[v_i] != 'f':
            v_s_i *= 1 if not vqed_m_results[v_i] else (-1)
            if measure_jk_dict[v_i][0] in stab_indexes_in_order_of_meas:                    
                j_state_prep_res = 1 if not start_stab_m_results[stab_indexes_in_order_of_meas.index(measure_jk_dict[v_i][0])] else -1
            else:
                j_state_prep_res = 1
            if measure_jk_dict[v_i][1] in stab_indexes_in_order_of_meas:
                k_state_prep_res = 1 if not start_stab_m_results[stab_indexes_in_order_of_meas.index(measure_jk_dict[v_i][1])] else -1
            else:
                k_state_prep_res = 1
            v_s_i *= (j_state_prep_res*k_state_prep_res)
                
    v_s = v_s_i
        
    obs_est_sample = []
    for obs_res in obs_meas_results:
        obs_est_sample.append((1 if not obs_res else -1)*v_s_i)
    #in our circuits we expect obs_est_sample values to be close to +1 -> we expect to return to |0> state

    #obs_est_samples.append(obs_est_sample)
    return v_s, obs_est_sample, any_detection