from abc import ABC
from discopy.quantum import Id, H, qubit, Box, Discard, Bra, Ket

def make_density_matrix(data, n_qubits):
    """ Create Box from density matrix. """
    box = Box(name='Density Matrix',
              dom=qubit ** 0,
              cod=qubit ** n_qubits,
              is_mixed=True)
    box.array = data
    return box

def make_state_vector(data, n_qubits):
    """ Create Box from state vector. """
    box = Box(name='State Vector',
              dom=qubit ** 0,
              cod=qubit ** n_qubits,
              is_mixed=False)
    box.array = data
    return box

def apply_box(circ, box, idx):
    circ >>= Id(idx) @ box @ Id(len(circ.cod) - len(box.dom) - idx)
    return circ


class BaseAnsatz(ABC):
    def __init__(self, n_layers, discard):
        self.n_layers = n_layers
        self.discard = discard

    def __call__(self, dom, cod, params):
        n_qubits = max(dom, cod)
        circ = self.make_circ(n_qubits, params)

        if dom < cod:
            circ = Ket(*[0] * (cod - dom)) >> circ
        elif dom > cod:
            effect = Discard() if self.discard else Bra(0)
            if dom == 2 * cod:
                boxes = [effect if i % 2 else Id(1) for i in range(n_qubits)]
            else:
                boxes = [effect if i < cod else Id(1) for i in range(n_qubits)]
            circ >>= Id.tensor(*boxes)
        return circ

    def make_circ(self, n_qubits, params):
        ...

    def n_qubits(self, n_qubits):
        ...


class IQPAnsatz(BaseAnsatz):
    def make_circ(self, n_qubits, params):
        n_layers = self.n_layers
        circ = Id(n_qubits)
        if n_qubits == 1:
            assert len(params) == 3
            circ = circ.Rx(params[0], 0)
            circ = circ.Rz(params[1], 0)
            circ = circ.Rx(params[2], 0)
        else:
            assert len(params) == (n_qubits-1) * n_layers
            lay_params = (n_qubits-1) 
            
            for n in range(n_layers):
                hadamards = Id(0).tensor(*(n_qubits * [H]))
                circ = circ >> hadamards
                for i in range(n_qubits-1):
                    tgt = i
                    src = i+1
                    circ = circ.CRz(params[i+(n*lay_params)], src, tgt)
        return circ

    def n_params(self, n_qubits):
        if n_qubits == 1:
            return 3
        return (n_qubits - 1) * self.n_layers

class Ansatz9(BaseAnsatz):
    def make_circ(self, n_qubits, n_layers, params):
        n_layers = self.n_layers
        circ = Id(n_qubits)

        if n_qubits == 1:
            assert len(params) == 3
            circ = circ.Rx(params[0], 0)
            circ = circ.Rz(params[1], 0)
            circ = circ.Rx(params[2], 0)
        else:
            assert len(params) == n_qubits * n_layers
            lay_params = n_qubits
            for n in range(n_layers):
                
                hadamards = Id.tensor(*(n_qubits * [H]))
                circ = circ >> hadamards

                for i in range(n_qubits - 1):
                    circ = circ.CZ(i, i + 1)

                for i in range(n_qubits):
                    param = params[i+(n * lay_params)]
                    circ = circ.Rx(param, i)

        return circ

    def n_params(self, n_qubits):
        if n_qubits == 1:
            return 3
        return n_qubits * self.n_layers

class Ansatz14(BaseAnsatz):
    def make_circ(self, n_qubits, params):
        n_layers = self.n_layers
        circ = Id(n_qubits)

        if n_qubits == 1:
            assert len(params) == 3
            circ = circ.Rx(params[0], 0)
            circ = circ.Rz(params[1], 0)
            circ = circ.Rx(params[2], 0)
        else:
            assert len(params) == 4 * n_qubits * n_layers
            lay_params = 4 * n_qubits
            
            for n in range(n_layers):
                # single qubit rotation wall 1
                for i in range(n_qubits):
                    param = params[i + (n * lay_params)]
                    circ = circ.Ry(param, i)

                # entangling ladder 1
                for i in range(n_qubits):
                    src = (n_qubits - 1 + i) % n_qubits
                    tgt = (n_qubits - 1 + i + 1) % n_qubits
                    param = params[i + n_qubits + (n * lay_params)]
                    circ = circ.CRx(param, src, tgt)

                # single qubit rotation wall 2
                for i in range(n_qubits):
                    param = params[i  + 2 * n_qubits+(n * lay_params)]
                    circ = circ.Ry(param, i)

                # entangling ladder 2
                for i in range(n_qubits):
                    src = (n_qubits - 1 + i) % n_qubits
                    tgt = (n_qubits - 1 + i - 1) % n_qubits
                    param = params[i + 3  * n_qubits + (n * lay_params)]
                    circ = circ.CRx(param, src, tgt)
        return circ

    def n_params(self, n_qubits):
        if n_qubits == 1:
            return 3
        return n_qubits * 4 * self.n_layers