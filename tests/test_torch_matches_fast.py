"""The torch implementation must agree with fast.py (itself validated vs PennyLane),
and its autograd gradients must agree with finite differences."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from qmllar import configs
from qmllar.fast import quanv_forward as np_forward
from qmllar.circuits import ENCODING_FUNCS
from qmllar.torch_quanv import quanv_forward as th_forward, ENCODE_TORCH

def check_forward(cfg, n=64, tol=1e-10):
    rng = np.random.default_rng(7)
    params = rng.uniform(-np.pi, np.pi, cfg.param_shape)
    raw = rng.normal(size=(n, cfg.n_qubits))
    funcs = ENCODING_FUNCS[cfg.encode_func]
    ang_np = np.stack([f(raw) for f in funcs], axis=-1)
    ref = np_forward(cfg, ang_np, params).real
    got = th_forward(cfg, ENCODE_TORCH[cfg.encode_func](torch.as_tensor(raw, dtype=torch.float64)),
                     torch.as_tensor(params, dtype=torch.float64),
                     cdtype=torch.complex128).numpy()
    err = np.abs(ref - got).max()
    ok = err < tol
    print(f"  [{'OK  ' if ok else 'FAIL'}] forward {cfg.name:28s} max|err| = {err:.2e}")
    return ok

def check_grad(cfg, tol=1e-5):
    """Autograd dL/dtheta vs central finite differences, double precision."""
    rng = np.random.default_rng(3)
    raw = torch.as_tensor(rng.normal(size=(8, cfg.n_qubits)), dtype=torch.float64)
    ang = ENCODE_TORCH[cfg.encode_func](raw)
    p = torch.as_tensor(rng.uniform(-1, 1, cfg.param_shape), dtype=torch.float64,
                        ).requires_grad_(True)
    loss = th_forward(cfg, ang, p, cdtype=torch.complex128).sum()
    loss.backward()
    auto = p.grad.detach().numpy()
    # finite differences on 5 randomly chosen parameters
    idx = [np.unravel_index(i, cfg.param_shape)
           for i in rng.choice(int(np.prod(cfg.param_shape)), 5, replace=False)]
    worst = 0.0
    for ix in idx:
        eps = 1e-6
        for sgn in (+1, -1):
            q = p.detach().clone(); q[ix] += sgn * eps
            val = th_forward(cfg, ang, q, cdtype=torch.complex128).sum().item()
            if sgn > 0: up = val
            else: dn = val
        fd = (up - dn) / (2 * eps)
        worst = max(worst, abs(fd - auto[ix]))
    ok = worst < tol
    print(f"  [{'OK  ' if ok else 'FAIL'}] grad    {cfg.name:28s} max|fd - autograd| = {worst:.2e}")
    return ok

if __name__ == "__main__":
    print("torch vs numpy forward, and autograd vs finite differences:")
    ok = True
    for cfg in configs.ALL_CONFIGS.values():
        ok &= check_forward(cfg)
        ok &= check_grad(cfg)
    print("ALL PASS" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
