import typer
import pandas as pd
import numpy as np
from Simulation import Simulation


"""
Run a simulation for a given rows in the input file.
Useful for HPC simulations or testing with a subset of the data.
"""
def run_simulation(input_file: str, start_row: int, end_row: int, output_folder: str, debug: bool = False):
    df = pd.read_csv(input_file)
    j = 1
    end_row = min(end_row, len(df) - 1)
    for i in range(start_row, end_row+1):
        row = df.iloc[[i]].squeeze()
        if debug:
            print(row)

        simulation = Simulation(row, output_folder)
        simulation.run()
        simulation.record_results(f'output_{start_row}_{end_row}.csv')

        print(f'Job {j} of {end_row-start_row+1} complete.')
        j += 1


"""
Run intervals of simulations.
Useful for running sequentially in chunks.
"""
def run_simulations(input_file: str, start_row:int, interval: int, output_folder: str, debug: bool = False):
    df = pd.read_csv(input_file)
    num_rows = len(df)
    end_row = start_row + interval - 1
    start_row_counter = start_row

    while end_row < num_rows:

        run_simulation(input_file, start_row_counter, end_row, output_folder, debug)
        start_row_counter += interval
        end_row += interval

    # Handle remaining rows if the total number of rows is not divisible by the interval
    if start_row_counter < num_rows:
        run_simulation(input_file, start_row_counter, num_rows - 1, output_folder, debug)

if __name__ == "__main__":
    typer.run(run_simulation)