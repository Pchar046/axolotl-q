"""Modular agents: sense -> integrate -> decide.

The integrate module is either classical (dense layer) or quantum
(a small circuit simulated with NumPy, cross-checked against Qiskit).
Parameters live in one flat vector; a mask marks damaged (missing) parts.
"""
import numpy as np

N_IN, N_H, N_ACT = 10, 4, 4


# ---------- tiny statevector simulator for the quantum module ----------
def _apply_1q(state, gate, q, n):
    s = state.reshape([2] * n)
    s = np.moveaxis(s, q, 0)
    s = np.tensordot(gate, s, axes=([1], [0]))
    return np.moveaxis(s, 0, q).reshape(-1)


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def _rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]])


def _cnot(state, c, t, n):
    s = state.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[c] = 1
    sub = s[tuple(idx)]
    tt = t if t < c else t - 1
    s[tuple(idx)] = np.flip(sub, axis=tt)
    return s.reshape(-1)


Z_SIGNS = None


def circuit_unitary(theta, n=N_H, layers=2):
    """The trainable part of the circuit as one 2^n x 2^n matrix (built once per policy)."""
    U = np.eye(2 ** n, dtype=complex)
    cols = []
    for j in range(2 ** n):
        state = U[:, j].copy()
        k = 0
        for _ in range(layers):
            for q in range(n):
                state = _apply_1q(state, _ry(theta[k]), q, n); k += 1
                state = _apply_1q(state, _rz(theta[k]), q, n); k += 1
            for q in range(n):
                state = _cnot(state, q, (q + 1) % n, n)
        cols.append(state)
    return np.array(cols).T


def _zsigns(n):
    global Z_SIGNS
    if Z_SIGNS is None or Z_SIGNS.shape[1] != 2 ** n:
        bits = (np.arange(2 ** n)[None, :] >> (n - 1 - np.arange(n))[:, None]) & 1
        Z_SIGNS = 1 - 2 * bits
    return Z_SIGNS


def quantum_layer_fast(x, U, n=N_H):
    """Encode x as a product state, apply the prebuilt circuit U, return <Z_i>."""
    a = np.pi * np.tanh(x[:n]) / 2
    state = np.array([1.0])
    for q in range(n):
        state = np.kron(state, np.array([np.cos(a[q]), np.sin(a[q])]))
    p = np.abs(U @ state) ** 2
    return _zsigns(n) @ p


def quantum_layer(x, theta, n=N_H, layers=2):
    """Angle-encode x into n qubits, apply `layers` of RY/RZ + CNOT ring, return <Z_i>."""
    global Z_SIGNS
    if Z_SIGNS is None or Z_SIGNS.shape[1] != 2 ** n:
        bits = (np.arange(2 ** n)[None, :] >> (n - 1 - np.arange(n))[:, None]) & 1
        Z_SIGNS = 1 - 2 * bits
    state = np.zeros(2 ** n, complex)
    state[0] = 1
    for q in range(n):
        state = _apply_1q(state, _ry(np.pi * np.tanh(x[q])), q, n)
    k = 0
    for _ in range(layers):
        for q in range(n):
            state = _apply_1q(state, _ry(theta[k]), q, n); k += 1
            state = _apply_1q(state, _rz(theta[k]), q, n); k += 1
        for q in range(n):
            state = _cnot(state, q, (q + 1) % n, n)
    p = np.abs(state) ** 2
    return Z_SIGNS @ p


# ---------- agent ----------
class Agent:
    """quantum=False: model A/B.  quantum=True: model C/D."""

    def __init__(self, quantum, seed):
        self.quantum = quantum
        rng = np.random.default_rng(seed)
        self.sizes = {"sense": N_IN * N_H, "integrate": 16, "decide": N_H * N_ACT}
        self.slices, start = {}, 0
        for k, s in self.sizes.items():
            self.slices[k] = slice(start, start + s)
            start += s
        self.theta = rng.normal(0, 0.5, start)
        self.mask = np.ones(start)  # 0 = damaged / missing

    @property
    def n_params(self):
        return int(self.mask.sum())

    def forward(self, o, theta=None, U=None):
        th = (self.theta if theta is None else theta) * self.mask
        W1 = th[self.slices["sense"]].reshape(N_IN, N_H)
        h = np.tanh(o @ W1)
        if self.quantum:
            if U is None:
                U = circuit_unitary(th[self.slices["integrate"]])
            h = quantum_layer_fast(h, U)
        else:
            W2 = th[self.slices["integrate"]].reshape(N_H, N_H)
            h = np.tanh(h @ W2)
        W3 = th[self.slices["decide"]].reshape(N_H, N_ACT)
        return h @ W3

    def policy(self, theta=None):
        U = None
        if self.quantum:
            th = (self.theta if theta is None else theta) * self.mask
            U = circuit_unitary(th[self.slices["integrate"]])
        return lambda o: self.forward(o, theta, U)

    def copy(self):
        a = Agent.__new__(Agent)
        a.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return a
