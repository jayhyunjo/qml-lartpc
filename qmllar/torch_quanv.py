"""Differentiable quanvolution (milestone A1).

The third implementation of the same circuit family:
    circuits.py   PennyLane, readable            (reference #1)
    fast.py       numpy state vectors, fast      (reference #2, validated vs #1)
    THIS FILE     torch state vectors            -> autograd gives dL/dtheta, runs on GPU

Validated against fast.py in tests/test_torch_matches_fast.py, including a
finite-difference gradient check.  Same conventions as fast.py throughout:
wire 0 = most significant bit; Rot(a,b,c) = RZ(c) RY(b) RZ(a); independent
parameters per reupload block.
"""
import numpy as np
import torch

from .circuits import QuanvConfig
from . import fast as _fast

# torch versions of the pixel-value -> angle encodings in circuits.ENCODING_FUNCS
ENCODE_TORCH = {
    "arctan_arctan_sq": lambda x: torch.stack([torch.atan(x), torch.atan(x * x)], -1),
    "identity":         lambda x: x.unsqueeze(-1),
    "arctan":           lambda x: torch.atan(x).unsqueeze(-1),
    "pi_scaled":        lambda x: (torch.pi * x).unsqueeze(-1),
}


def _rz(t):
    z = torch.zeros_like(t)
    return torch.stack([
        torch.stack([torch.exp(-0.5j * t), z + 0j], -1),
        torch.stack([z + 0j, torch.exp(0.5j * t)], -1)], -2)


def _ry(t):
    c, s = torch.cos(t / 2) + 0j, torch.sin(t / 2) + 0j
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def _rx(t):
    c, s = torch.cos(t / 2) + 0j, torch.sin(t / 2) + 0j
    return torch.stack([torch.stack([c, -1j * s], -1), torch.stack([-1j * s, c], -1)], -2)


_GATES = {"RX": _rx, "RY": _ry, "RZ": _rz}


def _embed(gate, q, nq, cdtype):
    """Single-qubit gate on wire q of nq -> (2^nq, 2^nq), via kron with identities."""
    out = None
    eye = torch.eye(2, dtype=cdtype, device=gate.device)
    for k in range(nq):
        m = gate if k == q else eye
        out = m if out is None else torch.kron(out, m)
    return out


def _entangler_matrix(cfg, device, cdtype):
    """The fixed CNOT pattern -- no parameters, so precompute with fast.py's numpy code."""
    nq = cfg.n_qubits
    ent = np.eye(2 ** nq, dtype=np.complex128)
    if cfg.entangler != "none" and nq > 1:
        for i in range(nq - 1):
            ent = _fast._cnot(i, i + 1, nq) @ ent
        if cfg.entangler == "ring":
            ent = _fast._cnot(nq - 1, 0, nq) @ ent
    return torch.as_tensor(ent, dtype=cdtype, device=device)


def ansatz_matrix(cfg, params_r, ent, cdtype):
    """U_theta for one channel+reupload: params_r (layers, nq, ppq) -> (2^nq, 2^nq).

    Built from torch ops on the parameter tensor, so autograd flows through it.
    """
    nq = cfg.n_qubits
    U = torch.eye(2 ** nq, dtype=cdtype, device=params_r.device)
    for l in range(cfg.n_layers):
        R = torch.eye(2 ** nq, dtype=cdtype, device=params_r.device)
        for q in range(nq):
            j = 0
            th = params_r[l, q]
            for g in cfg.ansatz_gates:
                if g == "Rot":                       # Rot(a,b,c) = RZ(c) RY(b) RZ(a)
                    gate = (_rz(th[j + 2]) @ _ry(th[j + 1]) @ _rz(th[j])).to(cdtype)
                    j += 3
                else:
                    gate = _GATES[g](th[j]).to(cdtype)
                    j += 1
                # embed EACH gate as it comes -- leaving this outside the loop
                # silently applies only the last gate of a multi-gate ansatz
                # (the bug this very test caught on first run)
                R = _embed(gate, q, nq, cdtype) @ R
        U = (ent @ R if cfg.entangle_before_rotation is False else R @ ent) @ U
    return U


def _qubit_amps(cfg, angles, cdtype):
    """Per-qubit state after encoding |0>: angles (B, nq, n_enc) -> (B, nq, 2)."""
    B, nq = angles.shape[0], cfg.n_qubits
    a0 = torch.ones(B, nq, dtype=cdtype, device=angles.device)
    a1 = torch.zeros(B, nq, dtype=cdtype, device=angles.device)
    for gi, g in enumerate(cfg.encode_gates):
        t = angles[:, :, gi]
        c, s = torch.cos(t / 2).to(cdtype), torch.sin(t / 2).to(cdtype)
        if g == "RX":
            a0, a1 = c * a0 - 1j * s * a1, -1j * s * a0 + c * a1
        elif g == "RY":
            a0, a1 = c * a0 - s * a1, s * a0 + c * a1
        else:  # RZ
            e = torch.exp(-0.5j * t).to(cdtype)
            a0, a1 = e * a0, torch.conj(e) * a1
    return torch.stack([a0, a1], -1)


def encode_product_state(cfg, angles, cdtype):
    """angles (B, nq, n_enc) -> product state (B, 2^nq)."""
    amps = _qubit_amps(cfg, angles, cdtype)
    state = amps[:, 0]
    for q in range(1, cfg.n_qubits):
        state = (state.unsqueeze(2) * amps[:, q].unsqueeze(1)).reshape(angles.shape[0], -1)
    return state


def apply_encoding_to_state(cfg, state, angles, cdtype):
    """Re-apply the (product) encoding unitary to an arbitrary state -- for reuploads."""
    B, nq = state.shape[0], cfg.n_qubits
    s = state.reshape((B,) + (2,) * nq)
    for q in range(nq):
        for gi, g in enumerate(cfg.encode_gates):
            t = angles[:, q, gi]
            sh = (B,) + (1,) * (nq - 1)
            c = torch.cos(t / 2).to(cdtype).reshape(sh)
            si = torch.sin(t / 2).to(cdtype).reshape(sh)
            s = torch.movedim(s, 1 + q, -1)
            s0, s1 = s[..., 0], s[..., 1]
            if g == "RX":
                n0, n1 = c * s0 - 1j * si * s1, -1j * si * s0 + c * s1
            elif g == "RY":
                n0, n1 = c * s0 - si * s1, si * s0 + c * s1
            else:
                e = torch.exp(-0.5j * t).to(cdtype).reshape(sh)
                n0, n1 = e * s0, torch.conj(e) * s1
            s = torch.movedim(torch.stack([n0, n1], -1), -1, 1 + q)
    return s.reshape(B, -1)


def quanv_forward(cfg, angles, params, ent=None, cdtype=torch.complex64):
    """angles (B, nq, n_enc) real, params cfg.param_shape -> (B, n_outputs) real."""
    nq = cfg.n_qubits
    if ent is None:
        ent = _entangler_matrix(cfg, params.device, cdtype)
    signs = [torch.as_tensor(
        [1.0 if ((i >> (nq - 1 - w)) & 1) == 0 else -1.0 for i in range(2 ** nq)],
        dtype=angles.dtype, device=angles.device) for w in cfg.measure_wires]
    outs = []
    for ch in range(cfg.n_channels):
        Us = [ansatz_matrix(cfg, params[ch, r], ent, cdtype) for r in range(cfg.n_reuploads)]
        s = encode_product_state(cfg, angles, cdtype) @ Us[0].T
        for r in range(1, cfg.n_reuploads):
            s = apply_encoding_to_state(cfg, s, angles, cdtype) @ Us[r].T
        prob = (s.real ** 2 + s.imag ** 2)
        for sg in signs:
            outs.append(prob @ sg)
    return torch.stack(outs, 1)


class QuanvLayer(torch.nn.Module):
    """Sliding-window quanvolution: (B, 1, H, W) -> (B, n_outputs, H-f+1, W-f+1).

    trainable=False freezes the circuit at its random initialisation -- the
    frozen-random control arm (plan section 2.3, arm 4) is this one flag.
    """

    def __init__(self, cfg: QuanvConfig, seed=0, trainable=True, cdtype=torch.complex64):
        super().__init__()
        self.cfg, self.cdtype = cfg, cdtype
        self.filt = int(np.sqrt(cfg.n_qubits))
        init = _fast.np.random.default_rng(seed).uniform(-np.pi, np.pi, cfg.param_shape)
        p = torch.as_tensor(init, dtype=torch.float32)
        self.params = torch.nn.Parameter(p, requires_grad=trainable)
        self.register_buffer("ent", _entangler_matrix(cfg, p.device, cdtype))
        self.encode = ENCODE_TORCH[cfg.encode_func]

    def forward(self, x):
        B, _, H, W = x.shape
        f = self.filt
        # unfold: every f x f window as a column -> (B, f*f, L), L = (H-f+1)*(W-f+1)
        wins = torch.nn.functional.unfold(x, kernel_size=f)
        L = wins.shape[-1]
        wins = wins.permute(0, 2, 1).reshape(B * L, f * f)       # one row per window
        out = quanv_forward(self.cfg, self.encode(wins), self.params,
                            ent=self.ent, cdtype=self.cdtype)
        oh = H - f + 1
        return out.reshape(B, L, -1).permute(0, 2, 1).reshape(B, -1, oh, W - f + 1)
