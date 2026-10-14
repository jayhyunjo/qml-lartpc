"""Vectorised statevector quanvolution.

Same circuit family as qmllar.circuits, but evaluated as explicit batched tensor
operations instead of one QNode call per window.  A 4-qubit state is a 16-vector,
so N windows is an (N, 16) complex array and the whole layer is a handful of
matmuls.  Validated against PennyLane in tests/test_fast_matches_pennylane.py.

Backend-agnostic: pass numpy for forward-only, or torch for autodiff.
"""
import numpy as np

I2 = np.eye(2, dtype=np.complex128)


def rx(t): 
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)


def ry(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(t):
    return np.array([[np.exp(-1j*t/2), 0], [0, np.exp(1j*t/2)]], dtype=np.complex128)


def rot(a, b, c):
    """PennyLane qml.Rot(a,b,c) = RZ(c) RY(b) RZ(a)."""
    return rz(c) @ ry(b) @ rz(a)


def _embed(gate, q, nq):
    """Single-qubit gate on wire q of nq (wire 0 = most significant)."""
    ops = [I2]*nq
    ops[q] = gate
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def _cnot(ctrl, targ, nq):
    dim = 2**nq
    M = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        bits = [(i >> (nq-1-k)) & 1 for k in range(nq)]
        if bits[ctrl] == 1:
            bits[targ] ^= 1
        j = sum(b << (nq-1-k) for k, b in enumerate(bits))
        M[j, i] = 1.0
    return M


def ansatz_matrix(cfg, params_ch):
    """Fixed U_theta for one channel: (n_layers, n_qubits, params_per_qubit) -> (2^nq, 2^nq)."""
    nq = cfg.n_qubits
    U = np.eye(2**nq, dtype=np.complex128)
    ent = np.eye(2**nq, dtype=np.complex128)
    if cfg.entangler != "none" and nq > 1:
        for i in range(nq-1):
            ent = _cnot(i, i+1, nq) @ ent
        if cfg.entangler == "ring":
            ent = _cnot(nq-1, 0, nq) @ ent

    for l in range(cfg.n_layers):
        R = np.eye(2**nq, dtype=np.complex128)
        for q in range(nq):
            j = 0
            th = params_ch[l][q]
            for g in cfg.ansatz_gates:
                if g == "Rot":
                    R = _embed(rot(th[j], th[j+1], th[j+2]), q, nq) @ R
                    j += 3
                else:
                    R = _embed({"RX": rx, "RY": ry, "RZ": rz}[g](th[j]), q, nq) @ R
                    j += 1
        U = (ent @ R if cfg.entangle_before_rotation is False else R @ ent) @ U
    return U


def encode_product_state(cfg, angles, xp=np):
    """angles: (B, nq, n_encode_gates) -> (B, 2^nq) product state from |0...0>."""
    nq = cfg.n_qubits
    B = angles.shape[0]
    state = xp.ones((B, 1), dtype=xp.complex128)
    for q in range(nq):
        amp = xp.ones((B, 2), dtype=xp.complex128)
        amp[:, 0] = 1.0
        amp[:, 1] = 0.0
        for gi, g in enumerate(cfg.encode_gates):
            t = angles[:, q, gi]
            c, s = xp.cos(t/2), xp.sin(t/2)
            if g == "RX":
                a0 = c*amp[:, 0] - 1j*s*amp[:, 1]
                a1 = -1j*s*amp[:, 0] + c*amp[:, 1]
            elif g == "RY":
                a0 = c*amp[:, 0] - s*amp[:, 1]
                a1 = s*amp[:, 0] + c*amp[:, 1]
            else:  # RZ
                a0 = xp.exp(-1j*t/2)*amp[:, 0]
                a1 = xp.exp(1j*t/2)*amp[:, 1]
            amp = xp.stack([a0, a1], axis=1)
        state = (state[:, :, None] * amp[:, None, :]).reshape(B, -1)
    return state


def apply_encoding_to_state(cfg, state, angles, xp=np):
    """Re-apply the (product) encoding unitary to an arbitrary batched state."""
    nq = cfg.n_qubits
    B = state.shape[0]
    s = state.reshape((B,) + (2,)*nq)
    for q in range(nq):
        for gi, g in enumerate(cfg.encode_gates):
            t = angles[:, q, gi]
            c, si = xp.cos(t/2), xp.sin(t/2)
            s = xp.moveaxis(s, 1+q, -1)
            s0, s1 = s[..., 0], s[..., 1]
            sh = (B,) + (1,)*(nq-1)
            c_ = c.reshape(sh); si_ = si.reshape(sh)
            if g == "RX":
                n0 = c_*s0 - 1j*si_*s1
                n1 = -1j*si_*s0 + c_*s1
            elif g == "RY":
                n0 = c_*s0 - si_*s1
                n1 = si_*s0 + c_*s1
            else:
                e = xp.exp(-1j*t/2).reshape(sh)
                n0 = e*s0; n1 = xp.conj(e)*s1
            s = xp.stack([n0, n1], axis=-1)
            s = xp.moveaxis(s, -1, 1+q)
    return s.reshape(B, -1)


def quanv_forward(cfg, angles, params, xp=np):
    """angles: (B, nq, n_enc) -> (B, n_channels) expectation values of Z on measure_wires."""
    nq = cfg.n_qubits
    outs = []
    for ch in range(cfg.n_channels):
        # NOTE: independent parameters per reupload, matching qmllar.circuits.
        # The PRD's Eq. (3) writes (U_theta U_F)^R, whose literal reading is
        # *shared* weights across reuploads.  Both are defensible; this is one of
        # the things to pin down with the authors.  See docs/primary-source-findings.md.
        Us = [ansatz_matrix(cfg, params[ch][r]) for r in range(cfg.n_reuploads)]
        s = encode_product_state(cfg, angles, xp)
        s = s @ Us[0].T
        for r in range(1, cfg.n_reuploads):
            s = apply_encoding_to_state(cfg, s, angles, xp)
            s = s @ Us[r].T
        prob = xp.abs(s)**2
        for w in cfg.measure_wires:
            sign = np.array([1.0 if ((i >> (nq-1-w)) & 1) == 0 else -1.0
                             for i in range(2**nq)])
            outs.append(prob @ sign)
    return xp.stack(outs, axis=1)
