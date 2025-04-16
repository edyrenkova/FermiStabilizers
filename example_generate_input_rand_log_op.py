import pandas as pd
import numpy as np
import os



fermion_hamiltonian_descr = 'fermi_hubbard'
boundary_conditions = 'periodic'
lattice_size_arr = [4]
efficient_trotter_steps_arr = None #[1,2,3,4,5]
logical_operators_depth_arr = [0.05]
encodings = ['C', 'JW', 'Ternary']
virtual_error_detection_rates = [0] 
stabilizer_reconstruction_options = [False]
flags_in_synd_extraction_options = [False]
type_of_logical_observables_arr = ['Z']
global_parity_postselection_options = [False]
non_destructive_stabilizer_measurement_end_options = [False]
#logical_operators_depth_arr=[0]


p_arr = [0.001]
error_models = [(1, 1, 1, 1, 0)] 
n_shots = 1000
num_rand_ham_subsets = 1
bootstrap_resamples = 100

FOLDER = '.'
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)
filename = f'{FOLDER}/input.csv'
RAND_SEEDS_INPUT_FILE = FOLDER+'/input.csv'


def get_rand_seeds_from_csv(filename):
    '''Gets the random seeds from another input csv file.
    This is useful for creating input file with different parameters but the same random seeds.'''
    df = pd.read_csv(filename)
    rand_seeds = df['rand_seed'].unique()
    return rand_seeds

def generate_error_model(p, error_model: tuple):
    '''Generates the error model based on the error model tuple and the error rate p.'''
    p1_factor, p2_factor, psp_factor, pm_factor, pi_factor = error_model
    p1 = p1_factor * p
    p2 = p2_factor * p
    psp = psp_factor * p
    pm = pm_factor * p
    pi = pi_factor * p
    return p1, p2, psp, pm, pi

def generate_metadata(filename):
    
    '''Generates the metadata (description of the input file) file.'''
    metadata = {'lattice_size': [str(lattice_size_arr)],
                'boundary_conditions': [str(boundary_conditions)],
                'logical_operators_depth_arr': [str(logical_operators_depth_arr)],
                'efficient_trotter_steps': [str(efficient_trotter_steps_arr)],
                'encodings': [str(encodings)],
                'global_parity_postselection_options': [str(global_parity_postselection_options)],
                'virtual_error_detection_rate': [str(virtual_error_detection_rates)],
                'stabilizer_reconstruction_options': [str(stabilizer_reconstruction_options)],
                'flags_in_synd_extraction_options': [str(flags_in_synd_extraction_options)],
                'non_destructive_stabilizer_measurement_end_options': [str(non_destructive_stabilizer_measurement_end_options)],
                'type_of_logical_observables': [str(type_of_logical_observables_arr)],
                'p_arr': [str(p_arr)],
                'error_models': [str(error_models)],
                'n_shots': [str(n_shots)], 'bootstrap_resamples': [str(bootstrap_resamples)],
                'num_rand_ham_subsets': [str(num_rand_ham_subsets)]
                }
    
    df = pd.DataFrame(metadata)
    df.to_csv(filename, index=False)

def generate_input(lattice_size_arr:list, fermion_hamiltonian_descr:str,
                   boundary_conditions:str,
                   logical_operators_depth_arr:list, efficient_trotter_steps_arr:list, encodings:list, global_parity_postselection_options:list,
                   virtual_error_detection_rates:list, stabilizer_reconstruction_options:list, flags_in_synd_extraction_options:list, 
                   non_destructive_stabilizer_measurement_end_options:list,
                   type_of_logical_observables_arr:list, 
                   p_arr:list, error_models:list, n_shots:int, bootstrap_resamples:int, num_rand_ham_subsets:int, filename:str):
    '''Generates the input file for the simulation.'''
    # Check if file exists
    if not os.path.exists(filename):
        # Create the file and write the column names
        with open(filename, 'w') as f:
            column_names = ['lattice_size', 'fermion_hamiltonian_descr', 'boundary_conditions', 'rand_seed',
                            'logical_operators_depth', 'efficient_trotter_steps', 'encoding', 'global_parity_postselection', 
                            'virtual_error_detection_rate', 
                            'stabilizer_reconstruction', 'flags_in_synd_extraction', 'non_destructive_stabilizer_measurement_end',
                            'type_of_logical_observables', 
                            'pm', 'psp', 'p1', 'p2', 'pi', 'error_model', 'n_shots', 'bootstrap_resamples']
            f.write(','.join(column_names) + '\n')
    
    #rand_seeds = get_rand_seeds_from_csv(RAND_SEEDS_INPUT_FILE)
    # One can also generate random seeds:
    rand_seeds = [np.random.randint(0, 100000) for i in range(num_rand_ham_subsets)]

    if efficient_trotter_steps_arr:
        for lattice_size in lattice_size_arr:
            for efficient_trotter_steps in efficient_trotter_steps_arr:
                for error_model in error_models:
                    for p in p_arr:
                        p1, p2, psp, pm, pi = generate_error_model(p, error_model)
                        for type_of_logical_observables in type_of_logical_observables_arr:
                            for enc in encodings:
                                if enc == 'JW':
                                    if type_of_logical_observables == 'Z':
                                        for global_parity_postselection in global_parity_postselection_options:
                                            for rand_seed in rand_seeds:
                                                data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                        'boundary_conditions': [boundary_conditions], 'rand_seed': [0],
                                                        'logical_operators_depth': [0], 'efficient_trotter_steps': [efficient_trotter_steps],
                                                        'encoding': [enc], 'global_parity_postselection': [global_parity_postselection],
                                                        'virtual_error_detection_rate': [0],
                                                        'stabilizer_reconstruction': [False],
                                                        'flags_in_synd_extraction': [False], 'non_destructive_stabilizer_measurement_end': [False],
                                                        'type_of_logical_observables': [type_of_logical_observables],
                                                        'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                                df = pd.DataFrame(data)
                                                df.to_csv(filename, mode='a', header=False, index=False)
                                    elif type_of_logical_observables == 'efficient':
                                        for rand_seed in rand_seeds:
                                            data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                    'boundary_conditions': [boundary_conditions], 'rand_seed': [0],
                                                    'logical_operators_depth': [0], 'efficient_trotter_steps': [efficient_trotter_steps],
                                                    'encoding': [enc], 'global_parity_postselection': [False],
                                                    'virtual_error_detection_rate': [0],
                                                    'stabilizer_reconstruction': [False],
                                                    'flags_in_synd_extraction': [False], 'non_destructive_stabilizer_measurement_end': [False],
                                                    'type_of_logical_observables': [type_of_logical_observables],
                                                    'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                            df = pd.DataFrame(data)
                                            df.to_csv(filename, mode='a', header=False, index=False)
                                    else:
                                        continue
                                elif enc == 'C':
                                    
                                    if type_of_logical_observables == 'efficient':
                                        for non_destructive_stabilizer_measurement_end_option in non_destructive_stabilizer_measurement_end_options:
                                            for flags_in_synd_extraction_option in flags_in_synd_extraction_options:
                                                data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                        'boundary_conditions': [boundary_conditions], 'rand_seed': [0],
                                                        'logical_operators_depth': [0], 'efficient_trotter_steps': [efficient_trotter_steps],
                                                        'encoding': [enc], 'global_parity_postselection': [False],
                                                        'virtual_error_detection_rate': [0],
                                                        'stabilizer_reconstruction': [False],
                                                        'flags_in_synd_extraction': [flags_in_synd_extraction_option], 'non_destructive_stabilizer_measurement_end': [non_destructive_stabilizer_measurement_end_option],
                                                        'type_of_logical_observables': [type_of_logical_observables],
                                                        'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                                df = pd.DataFrame(data)
                                                df.to_csv(filename, mode='a', header=False, index=False)
                                    elif type_of_logical_observables == 'Z':
                                        for stabilizer_reconstruction in stabilizer_reconstruction_options:
                                            data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                    'boundary_conditions': [boundary_conditions], 'rand_seed': [0],
                                                    'logical_operators_depth': [0], 'efficient_trotter_steps': [efficient_trotter_steps],
                                                    'encoding': [enc], 'global_parity_postselection': [False],
                                                    'virtual_error_detection_rate': [0],
                                                    'stabilizer_reconstruction': [stabilizer_reconstruction],
                                                    'flags_in_synd_extraction': [False], 'non_destructive_stabilizer_measurement_end': [False],
                                                    'type_of_logical_observables': [type_of_logical_observables],
                                                    'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                            df = pd.DataFrame(data)
                                            df.to_csv(filename, mode='a', header=False, index=False)
                                elif enc == 'Ternary':
                                    if type_of_logical_observables=='efficient':
                                        continue
                                    for rand_seed in rand_seeds:
                                        data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                'boundary_conditions': [boundary_conditions], 'rand_seed': [0],
                                                'logical_operators_depth': [0], 'efficient_trotter_steps': [efficient_trotter_steps],
                                                'encoding': [enc], 'global_parity_postselection': [False],
                                                'virtual_error_detection_rate': [0],
                                                'stabilizer_reconstruction': [False],
                                                'flags_in_synd_extraction': [False], 'non_destructive_stabilizer_measurement_end': [False],
                                                'type_of_logical_observables': [type_of_logical_observables],
                                                'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                        df = pd.DataFrame(data)
                                        df.to_csv(filename, mode='a', header=False, index=False)
                                else:
                                    print(enc, type_of_logical_observables, virtual_error_detection_rate, non_destructive_stabilizer_measurement_end_option)
                                    raise NotImplementedError('Invalid type of logical observables.')
        return

                                                    

    for lattice_size in lattice_size_arr:
        
        for error_model in error_models:

            for p in p_arr:
                p1, p2, psp, pm, pi = generate_error_model(p, error_model)

                for logical_operators_depth in logical_operators_depth_arr:
                    
                    for type_of_logical_observables in type_of_logical_observables_arr:

                        for enc in encodings:
                            if (enc == 'Ternary' and type_of_logical_observables == 'efficient'):
                                continue

                            if enc == 'Ternary':
                                for rand_seed in rand_seeds:
                                    data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr], 
                                        'boundary_conditions': [boundary_conditions], 'rand_seed': [rand_seed],
                                        'logical_operators_depth': [logical_operators_depth], 
                                        'efficient_trotter_steps':[0],
                                        'encoding': [enc], 'global_parity_postselection': [False], #only JW and Compact have global parity postselection
                                        'virtual_error_detection_rate': [0], 
                                        'stabilizer_reconstruction': [False], 
                                        'flags_in_synd_extraction': [False], 'non_destructive_stabilizer_measurement_end': [False],
                                        'type_of_logical_observables': [type_of_logical_observables], 
                                        'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                    df = pd.DataFrame(data)
                                    df.to_csv(filename, mode='a', header=False, index=False)
                            elif enc == 'JW':
                                if type_of_logical_observables == 'Z':
                                    for global_parity_postselection in global_parity_postselection_options:
                                        for rand_seed in rand_seeds:
                                            data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                    'boundary_conditions': [boundary_conditions], 'rand_seed': [rand_seed],
                                                    'logical_operators_depth': [logical_operators_depth], 
                                                    'efficient_trotter_steps':[0],
                                                    'encoding': [enc], 'global_parity_postselection': [global_parity_postselection], 
                                                    'virtual_error_detection_rate': [0], 
                                                    'stabilizer_reconstruction': [False], 
                                                    'flags_in_synd_extraction': [False], 'non_destructive_stabilizer_measurement_end': [False],
                                                    'type_of_logical_observables': [type_of_logical_observables], 
                                                    'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                            df = pd.DataFrame(data)
                                            df.to_csv(filename, mode='a', header=False, index=False)
                                else:
                                    for rand_seed in rand_seeds:
                                        data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                'boundary_conditions': [boundary_conditions], 'rand_seed': [rand_seed],
                                                'logical_operators_depth': [logical_operators_depth], 
                                                'efficient_trotter_steps':[0],
                                                'encoding': [enc], 'global_parity_postselection': [False], 
                                                'virtual_error_detection_rate': [0], 
                                                'stabilizer_reconstruction': [False], 
                                                'flags_in_synd_extraction': [False], 'non_destructive_stabilizer_measurement_end': [False],
                                                'type_of_logical_observables': [type_of_logical_observables], 
                                                'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                        df = pd.DataFrame(data)
                                        df.to_csv(filename, mode='a', header=False, index=False)
                            elif enc == 'C':
                                if type_of_logical_observables == 'efficient':
                                    for virtual_error_detection_rate in virtual_error_detection_rates:
                                        for non_destructive_stabilizer_measurement_end_option in non_destructive_stabilizer_measurement_end_options:
                                            for flags_in_synd_extraction_option in flags_in_synd_extraction_options:
                                                if virtual_error_detection_rate == 0:
                                                    for rand_seed in rand_seeds:
                                                        data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                                'boundary_conditions': [boundary_conditions], 'rand_seed': [rand_seed],
                                                                'logical_operators_depth': [logical_operators_depth], 
                                                                'efficient_trotter_steps':[0],
                                                                'encoding': [enc], 'global_parity_postselection': [False], 
                                                                'virtual_error_detection_rate': [virtual_error_detection_rate], 
                                                                'stabilizer_reconstruction': [False], 
                                                                'flags_in_synd_extraction': [flags_in_synd_extraction_option], 'non_destructive_stabilizer_measurement_end': [non_destructive_stabilizer_measurement_end_option],
                                                                'type_of_logical_observables': [type_of_logical_observables], 
                                                                'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                                        df = pd.DataFrame(data)
                                                        df.to_csv(filename, mode='a', header=False, index=False)
                                                else:
                                                    for rand_seed in rand_seeds:
                                                        data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                                'boundary_conditions': [boundary_conditions], 'rand_seed': [rand_seed],
                                                                'logical_operators_depth': [logical_operators_depth], 
                                                                'efficient_trotter_steps':[0],
                                                                'encoding': [enc], 'global_parity_postselection': [False], 
                                                                'virtual_error_detection_rate': [virtual_error_detection_rate], 
                                                                'stabilizer_reconstruction': [False], 
                                                                'flags_in_synd_extraction': [flags_in_synd_extraction_option], 'non_destructive_stabilizer_measurement_end': [non_destructive_stabilizer_measurement_end_option],
                                                                'type_of_logical_observables': [type_of_logical_observables], 
                                                                'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                                        df = pd.DataFrame(data)
                                                        df.to_csv(filename, mode='a', header=False, index=False)
                                elif type_of_logical_observables == 'Z':
                                    for virtual_error_detection_rate in virtual_error_detection_rates:
                                        if virtual_error_detection_rate != 0:
                                            for non_destructive_stabilizer_measurement_end_option in non_destructive_stabilizer_measurement_end_options:
                                                for flags_in_synd_extraction_option in flags_in_synd_extraction_options: 
                                                    for global_parity_postselection in global_parity_postselection_options:
                                                        if flags_in_synd_extraction_option and not non_destructive_stabilizer_measurement_end_option:
                                                            continue
                                                        for rand_seed in rand_seeds:
                                                            data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                                    'boundary_conditions': [boundary_conditions], 'rand_seed': [rand_seed],
                                                                    'logical_operators_depth': [logical_operators_depth],
                                                                    'efficient_trotter_steps':[0], 
                                                                    'encoding': [enc], 'global_parity_postselection': [global_parity_postselection], 
                                                                    'virtual_error_detection_rate': [virtual_error_detection_rate], 
                                                                    'stabilizer_reconstruction': [False], 
                                                                    'flags_in_synd_extraction': [flags_in_synd_extraction_option], 'non_destructive_stabilizer_measurement_end': [non_destructive_stabilizer_measurement_end_option],
                                                                    'type_of_logical_observables': [type_of_logical_observables], 
                                                                    'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                                            df = pd.DataFrame(data)
                                                            df.to_csv(filename, mode='a', header=False, index=False)

                                        else:
                                            for non_destructive_stabilizer_measurement_end_option in non_destructive_stabilizer_measurement_end_options:
                                                for flags_in_synd_extraction_option in flags_in_synd_extraction_options: 
                                                    for global_parity_postselection in global_parity_postselection_options:
                                                        for rand_seed in rand_seeds:
                                                            data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                                    'boundary_conditions': [boundary_conditions], 'rand_seed': [rand_seed],
                                                                    'logical_operators_depth': [logical_operators_depth], 
                                                                    'efficient_trotter_steps':[0],
                                                                    'encoding': [enc], 'global_parity_postselection': [global_parity_postselection], 
                                                                    'virtual_error_detection_rate': [virtual_error_detection_rate],
                                                                    'stabilizer_reconstruction': [False], 
                                                                    'flags_in_synd_extraction': [flags_in_synd_extraction_option], 'non_destructive_stabilizer_measurement_end': [non_destructive_stabilizer_measurement_end_option],
                                                                    'type_of_logical_observables': [type_of_logical_observables], 
                                                                    'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                                            df = pd.DataFrame(data)
                                                            df.to_csv(filename, mode='a', header=False, index=False)
                                            for stabilizer_reconstruction in stabilizer_reconstruction_options:
                                                if stabilizer_reconstruction:
                                                    for rand_seed in rand_seeds:
                                                            data = {'lattice_size': [lattice_size], 'fermion_hamiltonian_descr': [fermion_hamiltonian_descr],
                                                                    'boundary_conditions': [boundary_conditions], 'rand_seed': [rand_seed],
                                                                    'logical_operators_depth': [logical_operators_depth], 
                                                                    'efficient_trotter_steps':[0],
                                                                    'encoding': [enc], 'global_parity_postselection': [False], 
                                                                    'virtual_error_detection_rate': [virtual_error_detection_rate],
                                                                    'stabilizer_reconstruction': [stabilizer_reconstruction], 
                                                                    'flags_in_synd_extraction': [False], 'non_destructive_stabilizer_measurement_end': [False],
                                                                    'type_of_logical_observables': [type_of_logical_observables], 
                                                                    'pm': [pm], 'psp': [psp], 'p1': [p1], 'p2': [p2], 'pi': [pi], 'error_model':[error_model], 'n_shots': [n_shots], 'bootstrap_resamples': [bootstrap_resamples]}
                                                            df = pd.DataFrame(data)
                                                            df.to_csv(filename, mode='a', header=False, index=False)
                                            
                                           
                                else:
                                    print(enc, type_of_logical_observables, virtual_error_detection_rate, non_destructive_stabilizer_measurement_end_option)
                                    raise NotImplementedError('Invalid type of logical observables.')
                                
                            
                            else:
                                raise NotImplementedError('Invalid encoding.')


def sort_by_lattice_size(filename):
    '''Sorts the input file by lattice size.'''
    df = pd.read_csv(filename)
    df = df.sort_values(by='lattice_size')
    df.to_csv(filename, index=False)

generate_input(lattice_size_arr, fermion_hamiltonian_descr, boundary_conditions, logical_operators_depth_arr, efficient_trotter_steps_arr, encodings, global_parity_postselection_options,
              virtual_error_detection_rates, stabilizer_reconstruction_options, flags_in_synd_extraction_options, non_destructive_stabilizer_measurement_end_options, type_of_logical_observables_arr, 
               p_arr, error_models, n_shots, bootstrap_resamples, num_rand_ham_subsets, filename) 
generate_metadata(f'{FOLDER}/metadata.csv')
sort_by_lattice_size(filename)