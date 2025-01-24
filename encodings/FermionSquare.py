from panqec.codes import StabilizerCode
from typing import Tuple, Dict, List
import numpy as np

Operator = Dict[Tuple, str]  # Keys - qubit location, values - Pauli Operator('X', 'Y' or 'Z')
Coordinates = List[Tuple]  # List of locations (x,y) (could be qubits or stabilizers)

class FermionSquare(StabilizerCode):
    """
    Represents a periodic boundary square fermionic lattice encoded with DK with PanQEC library.

    Stores coordinate positions of qubits and stabilizer generators, as well as Pauli strings for stabilizer generators and logical operators.

    Not all functions available in panQEC are implemented here. However, this can still be used for panQEC GUI functionality.

    References:
    - [arXiv:2003.06939](https://arxiv.org/abs/2003.06939)

    Args:
        L_x (int): The size of the code in the x-direction.
        L_y (int, optional): The size of the code in the y-direction. Defaults to None. If none - then the code is square.
        deformed_axis (str, optional):  deformation not implemented for this code.

    Attributes:
        dimension (int): The dimension of the code.
        _qsmap (Dict[int, List]): A dictionary mapping qubit indices to stabilizer indices.
        _face_q_ids (List): A list of face qubit indices.
    """

    dimension = 2

    def __init__(self, L_x: int, L_y=None, deformed_axis=None):
       
        super().__init__(L_x, L_y, deformed_axis)
        self._qsmap = {}
        self._face_q_ids = []

    @property
    def label(self) -> str:
        """
        Returns the label of the Fermionic Square code.

        Returns:
            str: The label of the code.
        """
        return 'Fermionic Square {}x{}'.format(*self.size)

    @property
    def face_q_ids(self) -> List[int]:
        """
        Returns the indices of the face qubits in the code.

        Returns:
            List[int]: The indices of the face qubits.
        """
        if len(self._face_q_ids) == 0:
            for i, coord in enumerate(self.qubit_coordinates):
                if self.qubit_type(coord) == 'face':
                    self._face_q_ids.append(i)
        return self._face_q_ids

    @property
    def qsmap(self) -> Dict[int, List[int]]:
        """
        Returns the qubit-stabilizers map.

        Returns:
            Dict[int, List[int]]: A dictionary mapping qubit indices to stabilizer indices.
        """
        if len(self._qsmap.keys()) == 0:
            for qubit_location in self.qubit_index.keys():
                q_ind = self.qubit_index[qubit_location]
                self._qsmap[q_ind] = []
                if self.qubit_type(qubit_location) == 'face':
                    delta = [(0, 2), (0, -2), (2, 0), (-2, 0)]  # first 2 - X, last 2 - Y
                else:
                    delta = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
                for d in delta:
                    stab_location = tuple(np.add(qubit_location, d) %
                                          (2 * np.array(self.size)))
                    if self.is_stabilizer(stab_location):
                        self._qsmap[q_ind].append(self.stabilizer_index[stab_location])
        return self._qsmap

    def get_qubit_coordinates(self) -> Coordinates:
        """
        Returns the coordinates of all qubits in the code.

        The ordering is such that vertex qubits come first, followed by face qubits.

        Returns:
            Coordinates: A list of qubit coordinates.
        """
        coordinates: Coordinates = []
        Lx, Ly = self.size

        # vertex qubits
        for x in range(0, 2 * Lx, 2):
            for y in range(0, 2 * Ly, 2):
                coordinates.append((x, y))

        # Face Qubits
        for x in range(1, 2 * Lx, 4):
            for y in range(1, 2 * Ly, 4):
                coordinates.append((x, y))

        # more face Qubits
        for x in range(3, 2 * Lx, 4):
            for y in range(3, 2 * Ly, 4):
                coordinates.append((x, y))

        return coordinates

    def get_stabilizer_coordinates(self) -> Coordinates:
        """
        Returns the coordinates of all stabilizers in the code.

        The stabilizer coordinate is a point in the center of a square of 4 vertex qubits. No qubits are at the stabilizer coordinates.

        Returns:
            Coordinates: A list of stabilizer coordinates.
        """
        coordinates: Coordinates = []
        Lx, Ly = self.size

        # Faces
        for x in range(3, 2 * Lx, 4):
            for y in range(1, 2 * Ly, 4):
                coordinates.append((x, y))

        for x in range(1, 2 * Lx, 4):
            for y in range(3, 2 * Ly, 4):
                coordinates.append((x, y))

        return coordinates

    def stabilizer_type(self, location: Tuple) -> str:
        """
        Returns the type of stabilizer at the given location.

        Note:
            Not implemented for this code.

        Args:
            location (Tuple): The location of the stabilizer.

        Returns:
            str: The type of stabilizer.

        Raises:
            ValueError: If the coordinate is not a valid stabilizer coordinate.
        """
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        return 'face'

    def get_stabilizer(self, location: Tuple, deformed_axis=None) -> Operator:
        """
        Returns the stabilizer operator at the given location.

        Args:
            location (Tuple): The location of the stabilizer.
            deformed_axis (str, optional): The axis of deformation. Defaults to None.

        Returns:
            Operator: The stabilizer operator. Stores the Pauli operator at each qubit location.

        Raises:
            ValueError: If the coordinate is not a valid stabilizer coordinate.
        """
        if not self.is_stabilizer(location):
            raise ValueError(f"Invalid coordinate {location} for a stabilizer")

        def get_pauli(relative_location):
            if relative_location == (0, 2) or relative_location == (0, -2):
                return 'Y'
            if relative_location == (2, 0) or relative_location == (-2, 0):
                return 'X'
            else:
                return 'Z'

        delta = [(-1, 1), (-1, -1), (1, -1), (1, 1), (0, 2), (2, 0), (-2, 0), (0, -2)]

        operator = dict()
        for d in delta:
            qubit_location = tuple(np.add(location, d) %
                                   (2 * np.array(self.size)))
            if self.is_qubit(qubit_location):
                operator[qubit_location] = get_pauli(d)
            else:
                operator[qubit_location] = ''

        return operator

    def qubit_type(self, location: Tuple) -> str:
        """
        Returns the type of qubit at the given location.

        Args:
            location (Tuple): The location of the qubit.

        Returns:
            str: The type of qubit.
        """
        if location[0] % 2 != 0 and location[1] % 2 != 0:
            return 'face'
        else:
            return 'vertex'

    def qubit_axis(self, location: Tuple) -> str:
        """
        Returns the axis of the qubit at the given location.

        Note:
            Not implemented for this code.

        Args:
            location (Tuple): The location of the qubit.

        Returns:
            str: The axis of the qubit.

        """
        return 'x'

    def get_logicals_x(self) -> List[Operator]:
        """
        Returns a list of logical operators in the order of clockwise rotation around each stabilizer.

        Ignores phases of the logical operators.

        Returns:
            List[Operator]: A list of logical X operators.
        """
        logicals = []
        Lx, Ly = self.size
        relative_op_locations = [[(-1, 1), (-1, -1), (-2, 0)],  # up XYX
                                 [(-1, -1), (1, -1), (0, -2)],  # side XYY
                                 [(1, -1), (1, 1), (2, 0)],  # down XYX
                                 [(1, 1), (-1, 1), (0, 2)]]  # side XYY

        for stab in self.get_stabilizer_coordinates():
            for i, relative_op_location in enumerate(relative_op_locations):
                operator = dict()

                qubit_location = tuple(np.add(stab, relative_op_location[0]) %
                                       (2 * np.array(self.size)))
                operator[qubit_location] = 'X'
                qubit_location = tuple(np.add(stab, relative_op_location[1]) %
                                       (2 * np.array(self.size)))
                operator[qubit_location] = 'Y'
                if i % 2 == 0:
                    qubit_location = tuple(np.add(stab, relative_op_location[2]) %
                                           (2 * np.array(self.size)))
                    operator[qubit_location] = 'X'
                else:
                    qubit_location = tuple(np.add(stab, relative_op_location[2]) %
                                           (2 * np.array(self.size)))
                    operator[qubit_location] = 'Y'
                logicals.append(operator)

        return logicals

    def get_qubit_logicals_dict(self) -> Dict[Tuple, List[Operator]]:
        """
        Returns a dictionary mapping qubit coordinates to logical operators.

        Returns:
            Dict[Tuple, List[Operator]]: A dictionary mapping qubit coordinates to logical operators.
        """
        x_logicals = self.get_logicals_x()
        logical_dict = dict()
        for x_l in x_logicals:
            for q_coord in x_l.keys():
                if q_coord not in logical_dict.keys():
                    logical_dict[q_coord] = list()
                logical_dict[q_coord].append(x_l)

        return logical_dict

    def x_op_orientation(self, op: Operator) -> str:
        """
        Given an edge logical operator, returns "h" for horizontally oriented operators and 
        "v" for vertically oriented operators.

        Args:
            op (Operator): The edge logical operator.

        Returns:
            str: The orientation of the operator.
        """
        for q_coord in op.keys():
            if self.qubit_type(q_coord) == "face":
                if op[q_coord] == 'Y':
                    return "h"
        return "v"

    def get_logicals_z(self) -> List[Operator]:
        """
        Returns the logical Z operators on all vertex qubits.

        Returns:
            List[Operator]: A list of logical Z operators.
        """
        Lx, Ly = self.size
        logicals = []

        # vertex qubits
        for x in range(0, 2 * Lx, 2):
            for y in range(0, 2 * Ly, 2):
                operator = dict()
                operator[(x, y)] = 'Z'
                logicals.append(operator)

        return logicals

    def stabilizer_representation(self, location: Tuple, rotated_picture=False) -> str:
        """
        Represents the stabilizer at the given location.

        Args:
            location (Tuple): The location of the stabilizer.
            rotated_picture (bool, optional): Whether to rotate the picture. Defaults to False.

        Returns:
            str: The stabilizer representation at the given location.
        """
        representation = super().stabilizer_representation(location, rotated_picture, json_file='FermionSquare.json')
        return representation

    def qubit_representation(self, location: Tuple, rotated_picture=False) -> str:
        """
        Represents the qubit at the given location.

        Args:
            location (Tuple): The location of the qubit.
            rotated_picture (bool, optional): Whether to rotate the picture. Defaults to False.

        Returns:
            str: The qubit representation at the given location.
        """
        representation = super().qubit_representation(location, rotated_picture, json_file='FermionSquare.json')
        return representation
    
