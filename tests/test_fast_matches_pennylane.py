"""The vectorised path must agree with PennyLane to numerical precision."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qmllar import configs, build_circuit, init_params
from qmllar.fast import quanv_forward

def check(cfg, n=32, tol=1e-9):
    rng = np.random.default_rng(1)
    params = init_params(cfg, rng)
    n_enc = len(cfg.encode_gates)
    raw = rng.normal(size=(n, cfg.n_qubits))
    from qmllar.circuits import ENCODING_FUNCS
    funcs = ENCODING_FUNCS[cfg.encode_func]
    angles = np.stack([np.stack([f(raw[b]) for f in funcs], axis=-1) for b in range(n)])
    fast = quanv_forward(cfg, angles, params)
    circ = build_circuit(cfg)
    ref = np.array([np.atleast_1d(np.array(circ(raw[b], params), dtype=float)).ravel()
                    for b in range(n)])
    ref = ref.reshape(n, -1)
    err = np.abs(fast.real - ref).max()
    status = "OK  " if err < tol else "FAIL"
    print(f"  [{status}] {cfg.name:28s} shape {tuple(fast.shape)} vs {tuple(ref.shape)}  max|err| = {err:.2e}")
    return err < tol

if __name__ == "__main__":
    print("validating vectorised quanvolution against PennyLane:")
    ok = all(check(c) for c in configs.ALL_CONFIGS.values())
    print("ALL PASS" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
