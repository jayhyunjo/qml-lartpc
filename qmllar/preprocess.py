"""Charge normalisation — a swept configuration choice, not a fixed assumption.

Raw campaign charge spans 0 to ~2.4e5 (nonzero median ~720, p99 ~1.6e4).  None of
the three paper encodings is defined on that range:

    Chen      arctan(x), arctan(x^2)   saturates at pi/2 for x >~ 10, so every
                                        pixel above ~10 encodes to the same angle
    PRD       identity -> RX(x)         RX(720) wraps the Bloch sphere ~115 times
    Tutorial  pi_scaled -> RX(pi*x)     worse

Chen's inputs were pre-scaled 30x30 images, not raw ADC.  So normalisation is part
of the model definition and must be swept alongside the encoding, never assumed --
`arctan` on log1p-scaled charge is a different model from `arctan` on max-scaled
charge.  Raw charge is kept in the patch file; normalisation is applied here.
"""
import numpy as np

# Global scale reference measured on data/patches_31.npz (nonzero p99).
DEFAULT_P99 = 15610.0


def normalise(patches, mode="per_patch_max", p99=DEFAULT_P99, eps=1e-6):
    """patches: (N, H, W) raw charge -> (N, H, W) in roughly [0, 1].

    mode:
      per_patch_max  x / max(patch)                 scale-free, loses absolute dE/dx
      global_p99     clip(x / p99, 0, 1)            keeps absolute scale, clips tails
      log1p          log1p(x) / log1p(p99)          compresses the 4-decade range
      arcsinh        arcsinh(x / p99)               like log but smooth through 0
      none           raw                            reference only; encodings saturate
    """
    x = np.asarray(patches, dtype=np.float32)
    # np.float32(...) on the scalars below keeps the output float32: a float32
    # array divided by a float64 *numpy* scalar silently promotes to float64,
    # which later breaks torch layers initialised in float32.
    if mode == "none":
        out = x
    elif mode == "per_patch_max":
        m = x.reshape(len(x), -1).max(axis=1).reshape(-1, 1, 1)
        out = x / np.maximum(m, np.float32(eps))
    elif mode == "global_p99":
        out = np.clip(x / np.float32(p99), 0.0, 1.0)
    elif mode == "log1p":
        out = np.log1p(np.maximum(x, 0.0)) / np.float32(np.log1p(p99))
    elif mode == "arcsinh":
        out = np.arcsinh(x / np.float32(p99))
    else:
        raise ValueError(f"unknown normalisation mode: {mode}")
    return out.astype(np.float32, copy=False)


NORM_MODES = ("per_patch_max", "global_p99", "log1p", "arcsinh", "none")
