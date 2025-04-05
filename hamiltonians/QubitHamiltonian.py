from openfermion.ops import QubitOperator
from openfermion.measurements import group_into_tensor_product_basis_sets

class QubitHamiltonian:

    def __init__(self, qubit_operator: QubitOperator):
        self.qubit_operator = qubit_operator
    
    def get_qubit_operator(self):
        return self.qubit_operator
    
    def get_many_body_order(self):
        """
        Get the non-locality measure of the qubit Hamiltonian.
        """
        return self.qubit_operator.many_body_order()
    
    def get_hamiltonian_length(self):
        """
        Get the length of the qubit Hamiltonian.
        """
        return len(self.qubit_operator.terms)
    
    def get_measurement_bases(self, seed):
        """
        Get the measurement bases of the qubit Hamiltonian.
        """
        return group_into_tensor_product_basis_sets(self.qubit_operator, seed=seed)
    
    def get_meas_bases_sorted(self, seed):
        """
        Get the measurement basis with the most operators.
        """
        meas_bases = group_into_tensor_product_basis_sets(self.qubit_operator, seed=seed)
        
        sorted_bases = sorted(meas_bases.items(), key=lambda x: len(list(x[1].terms.keys())), reverse=True)

        return sorted_bases
