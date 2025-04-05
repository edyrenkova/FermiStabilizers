from openfermion.ops import FermionOperator
from openfermion import hermitian_conjugated
from openfermion.hamiltonians import fermi_hubbard
import random

class FermionHamiltonian:
    """
    Represents a fermionic Hamiltonian.

    In the same object one can store both a full hamiltonian as well as a random subset of the terms.
    Useful for creating circuits with tensor product basis measurements.

    Attributes:
        fermion_operator (FermionOperator): The fermionic operator representing the Hamiltonian.
        L (int): The size of the system.
    """

    def __init__(self, fermi_hubbard:bool=True, L=None, add_hermitian_conjugate:bool=True, seed = 0, subset_size = 0, periodic=True):
        """
        Initializes a FermionHamiltonian object.

        Automatically generates fermi-hubbard model as well as generates a subset of terms.

        Args:
            fermi_hubbard (bool, optional): Whether to generate a Fermi-Hubbard Hamiltonian. Defaults to True.
            L (int, optional): The size of the system. Defaults to None.
            add_hermitian_conjugate (bool, optional): Whether to add the hermitian conjugate of the terms in the hamiltonian_subset. Defaults to True. Decreases the number of encoded terms for all encodings due to the hopping terms.            
            seed (int, optional): The seed for the random number generator. Defaults to 0.
            subset_size (int, optional): The size of the subset of terms. Defaults to 0.
            fermion_operator (FermionOperator, optional): The fermionic operator representing the Hamiltonian. Defaults to None
        """
        self.full_hamiltonian = FermionOperator()
        self.L = L if L is not None else 0
        self.is_fermi_hubbard = fermi_hubbard
        if not self.is_fermi_hubbard:
            raise ValueError("Only Fermi-Hubbard model is supported at the moment.")
        self.seed = seed
        self.subset_size = subset_size
        self.generate_fermi_hubbard(self.L, periodic=periodic)
        self.hamiltonian_subset = self.get_rand_ham_subset(self.subset_size, add_hermitian_conjugate)
        self.full_ham_length = len(list(self.full_hamiltonian.terms.items()))

    
    def generate_fermi_hubbard(self, L, tunneling=2., coulomb=1., magnetic_field=0.5, chemical_potential=0.25, periodic=True, spinless=True):
        """
        Generates a Fermi-Hubbard Hamiltonian.

        Args:
            L (int): The size of the system.
            tunneling (float, optional): The tunneling parameter. Defaults to 2.0.
            coulomb (float, optional): The Coulomb interaction parameter. Defaults to 1.0.
            magnetic_field (float, optional): The magnetic field parameter. Defaults to 0.5.
            chemical_potential (float, optional): The chemical potential parameter. Defaults to 0.25.
            periodic (bool, optional): Whether the system is periodic. Defaults to True.
            spinless (bool, optional): Whether the system is spinless. Defaults to True.
        """
        hubbard_terms = fermi_hubbard(x_dimension=L, y_dimension=L, 
                                  tunneling=tunneling, coulomb=coulomb, chemical_potential=chemical_potential,
                                  magnetic_field=magnetic_field, periodic=periodic, spinless=spinless).terms
        self.L = L
        for term, coefficient in hubbard_terms.items():
            self._add_term(term, coefficient)
        self.is_fermi_hubbard = True

    
    def _add_term(self, term, coefficient):
        """
        Adds a term to the fermionic operator.

        Args:
            term (str): The term to be added.
            coefficient (complex): The coefficient of the term.
        """
        self.full_hamiltonian += FermionOperator(term) * coefficient

    def get_rand_ham_subset(self, op_num_p, add_hermitian_conjugate=True):
        """
        Returns a random subset of the Hamiltonian. Supports hermitian conjugate and op_num larger than the number of terms.

        When the number of terms in the subset is larger than the number of terms in the Hamiltonian, the terms are repeated, but in a different random order. The seed used is incremented deterministically.

        Args:
            op_num_p (int): The percentage of terms in the subset. (rounded down)
            add_hermitian_conjugate (bool, optional): Whether to add the hermitian conjugate of each term.
                Defaults to True.

        Returns:
            list[FermionOperator]: if op_num < total number of terms, then returns a list with just one FermionOperator. Otherwise, returns a list of FermionOperators.
                                    This is needed to make sure the Hamiltonian is simplified but the repeated operators are kept separate for the cases when op_num > total number of terms.
        """

        #need to make sure to add the correct number of terms but also avoid duplicates....
        all_terms = list(self.full_hamiltonian.terms.items())
        random.seed(self.seed)
        random.shuffle(all_terms)
        
        rand_ham_subset = []
        num_terms = len(all_terms)
        fermion_op = FermionOperator()
        op_num = int(op_num_p * num_terms)
        self.rand_op_num = op_num
        for i in range(op_num):
            term, coefficient = all_terms[i % num_terms]
            fermion_op += FermionOperator(term) * coefficient
            
            if i % num_terms == num_terms - 1:
                if add_hermitian_conjugate:
                    fermion_op += hermitian_conjugated(fermion_op)
                rand_ham_subset.append(fermion_op)
                fermion_op = FermionOperator()
                random.seed(self.seed + i + 1)
                random.shuffle(all_terms)
        if add_hermitian_conjugate:
            rand_ham_subset.append(fermion_op+hermitian_conjugated(fermion_op))
        else:
            rand_ham_subset.append(fermion_op)
        return rand_ham_subset
        
    def __str__(self):
        """
        Returns a string representation of the fermionic operator.

        Returns:
            str: The string representation.
        """
        return str(self.full_hamiltonian)
