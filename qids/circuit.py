"""Small differentiable statevector circuits with explicit bit ordering."""

import numpy as np
import torch
from torch import nn


class Circuit:
    def __init__(self, qubits):
        self.qubits = qubits
        index = torch.arange(1 << qubits)
        bit = (index[:, None] >> torch.arange(qubits)) & 1
        self.z = (1 - 2*bit).float()
        self.flips = [index ^ (1 << i) for i in range(qubits)]
        self.controls = [index ^ (((index >> i) & 1) << ((i+1) % qubits)) for i in range(qubits)]

    def zero(self, x):
        state = torch.zeros((len(x), 1 << self.qubits), dtype=torch.complex64)
        state[:, 0] = 1
        return state

    def ry(self, state, angle, wire):
        c, s = (angle/2).cos().reshape(-1,1), (angle/2).sin().reshape(-1,1)
        return c*state - s*state[:,self.flips[wire]]*self.z[:,wire]

    def rz(self, state, angle, wire):
        return state*torch.exp(-.5j * angle.reshape(-1,1) * self.z[:,wire])

    def ry_layer(self, state, angles):
        for wire in range(self.qubits):
            state = self.ry(state, angles[:,wire], wire)
        return state

    def encode(self, x, method):
        state = self.ry_layer(self.zero(x), x)
        if method == "phase":
            # The commuting single and two-body systems can use one diagonal pass.
            pair = self.z * self.z.roll(-1,dims=1)
            interaction = x*x.roll(-1,dims=1)
            state = state*torch.exp(1j*(.45*x@self.z.T + .25*interaction@pair.T))
            state = self.ry_layer(state, .6*x)
        return state

    def measure(self, state):
        return state.abs().square()@self.z


class QuantumNet(nn.Module):
    """RY/RZ trainable blocks, re-uploaded features and a ring of CNOT gates."""

    def __init__(self, qubits, hybrid=False, layers=2):
        super().__init__()
        self.circuit = Circuit(qubits)
        self.hybrid = hybrid
        self.angles = nn.Parameter(torch.randn(layers,qubits,2)*.15)
        self.head = nn.Linear(2*qubits if hybrid else 1,1)

    def states(self, x):
        state = self.circuit.zero(x)
        for block in self.angles:
            state = self.circuit.ry_layer(state, x+block[:,0])
            for i in range(self.circuit.qubits):
                state = self.circuit.rz(state,block[i,1],i)
            for perm in self.circuit.controls:
                state = state[:,perm]
        return state

    def logits(self,x,observations):
        fields = torch.cat([x,observations],dim=1) if self.hybrid else observations[:,:1]
        return self.head(fields).ravel()

    def forward(self,x):
        return self.logits(x,self.circuit.measure(self.states(x)))


def fidelity(a,b):
    return np.abs(a@b.conj().T).astype(np.float64)**2


def noisy_overlap(matrix, qubits, mix, shots, rng, *, symmetric=False):
    prob = ((1-mix)*matrix + mix/(1 << qubits)).clip(0,1)
    if not shots:
        return prob
    if symmetric:
        ix = np.triu_indices(len(matrix))
        out = np.zeros_like(matrix)
        out[ix] = rng.binomial(shots,prob[ix])/shots
        out[(ix[1],ix[0])] = out[ix]
        return out
    return rng.binomial(shots,prob)/shots


def measure_shots(states, signs, shots, mix, rng):
    distributions = (1-mix)*np.abs(states)**2 + mix/states.shape[1]
    distributions = distributions/distributions.sum(axis=1,keepdims=True)
    if not shots:
        return distributions@signs
    counts = np.array([rng.multinomial(shots,p.astype(float)/p.astype(float).sum())
                       for p in distributions])
    return counts@signs/shots
