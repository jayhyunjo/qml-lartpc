"""Configurable quanvolution circuit.

One parameterised circuit family covers all three reference papers (see
docs/primary-source-findings.md section I).  Switches:

    encoding    gate(s) + function applied to each pixel
    ansatz      trainable gate set, params per qubit per layer
    entangler   CNOT/CX pattern
    depth       n_layers (ansatz repeats) and n_reuploads (re-encode data)
    measurement observable + which wires

The number of qubits equals the number of pixels in the window, following all
three papers ("the number of qubits equals to the square of kernel (filter)
size" -- Chen et al. Fig. 5 caption).
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pennylane as qml

# --------------------------------------------------------------------------
# encoding functions: pixel value -> rotation angle(s)
# --------------------------------------------------------------------------

ENCODING_FUNCS: dict[str, Sequence[Callable]] = {
    # Chen et al. 2012.12177 Fig. 5: two angles per pixel
    "arctan_arctan_sq": (np.arctan, lambda x: np.arctan(x**2)),
    # Tutorial 2009.09423 Fig. 3 / PRD Eq. (4): one angle per pixel
    "identity": (lambda x: x,),
    "arctan": (np.arctan,),
    "pi_scaled": (lambda x: np.pi * x,),
}


@dataclass(frozen=True)
class QuanvConfig:
    """A point in the circuit-family design space."""

    name: str
    n_qubits: int
    encode_gates: tuple[str, ...]          # e.g. ("RY", "RZ")
    encode_func: str                       # key into ENCODING_FUNCS
    ansatz_gates: tuple[str, ...]          # e.g. ("Rot",) or ("RY", "RZ")
    entangler: str = "chain"               # chain | ring | none
    n_layers: int = 1                      # ansatz repeats within one block
    n_reuploads: int = 1                   # (U_theta U_F)^R  -- PRD Eq. (3)
    measure_wires: tuple[int, ...] = (0,)  # one output scalar per wire
    n_channels: int = 1                    # independent parameter sets
    entangle_before_rotation: bool = True  # Chen: CNOTs then rotations

    @property
    def params_per_qubit(self) -> int:
        return sum(3 if g == "Rot" else 1 for g in self.ansatz_gates)

    @property
    def n_params(self) -> int:
        """Trainable parameters in this quanvolution layer."""
        return (self.n_channels * self.n_reuploads * self.n_layers
                * self.n_qubits * self.params_per_qubit)

    @property
    def param_shape(self) -> tuple[int, ...]:
        return (self.n_channels, self.n_reuploads, self.n_layers,
                self.n_qubits, self.params_per_qubit)

    @property
    def n_outputs(self) -> int:
        return self.n_channels * len(self.measure_wires)


def _apply_encoding(cfg: QuanvConfig, x):
    """Encode one window of pixel values into rotation angles."""
    funcs = ENCODING_FUNCS[cfg.encode_func]
    if len(funcs) != len(cfg.encode_gates):
        raise ValueError(
            f"{cfg.name}: {len(cfg.encode_gates)} encode gates but "
            f"{len(funcs)} encode funcs for '{cfg.encode_func}'"
        )
    for gate_name, f in zip(cfg.encode_gates, funcs):
        gate = getattr(qml, gate_name)
        for w in range(cfg.n_qubits):
            gate(f(x[w]), wires=w)


def _apply_entangler(cfg: QuanvConfig):
    if cfg.entangler == "none" or cfg.n_qubits < 2:
        return
    for i in range(cfg.n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    if cfg.entangler == "ring":
        qml.CNOT(wires=[cfg.n_qubits - 1, 0])


def _apply_ansatz(cfg: QuanvConfig, theta_layer):
    """theta_layer: (n_qubits, params_per_qubit)"""
    for w in range(cfg.n_qubits):
        j = 0
        for gate_name in cfg.ansatz_gates:
            if gate_name == "Rot":
                qml.Rot(theta_layer[w][j], theta_layer[w][j + 1],
                        theta_layer[w][j + 2], wires=w)
                j += 3
            else:
                getattr(qml, gate_name)(theta_layer[w][j], wires=w)
                j += 1


def build_circuit(cfg: QuanvConfig, device: str = "default.qubit",
                  diff_method: str = "backprop"):
    """Return a QNode mapping (window, params) -> list of expectation values.

    window : (n_qubits,)      pixel values of one filter-sized patch
    params : cfg.param_shape
    """
    dev = qml.device(device, wires=cfg.n_qubits)

    @qml.qnode(dev, diff_method=diff_method)
    def _channel(x, params_ch):
        """One channel = one independent circuit, as one conv filter is one kernel."""
        for r in range(cfg.n_reuploads):
            _apply_encoding(cfg, x)
            for l in range(cfg.n_layers):
                if cfg.entangle_before_rotation:
                    _apply_entangler(cfg)
                    _apply_ansatz(cfg, params_ch[r][l])
                else:
                    _apply_ansatz(cfg, params_ch[r][l])
                    _apply_entangler(cfg)
        return [qml.expval(qml.PauliZ(w)) for w in cfg.measure_wires]

    def circuit(x, params):
        """Returns n_channels * len(measure_wires) values, channel-major."""
        outs = []
        for c in range(cfg.n_channels):
            outs.extend(np.atleast_1d(np.asarray(_channel(x, params[c]), dtype=float)).ravel())
        return outs

    circuit.channel_qnode = _channel
    return circuit


def init_params(cfg: QuanvConfig, rng: np.random.Generator, scale: float = np.pi):
    """Uniform in [-scale, scale].  PRD Table II sweeps this scale."""
    return rng.uniform(-scale, scale, size=cfg.param_shape)
