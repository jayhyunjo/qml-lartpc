"""The three reference-paper circuits as configurations of one module.

Sources are quoted in docs/primary-source-findings.md section I.  Each config
carries the parameter count the paper reports, so the module is checked against
published numbers rather than against my reading of the figures.
"""

from .circuits import QuanvConfig

# --------------------------------------------------------------------------
# Chen, Wei, Zhang, Yu, Yoo -- arXiv:2012.12177, Fig. 5
#
#   "Ry(arctan(x_i)) and Rz(arctan(x_i^2))" encoding, not optimised.
#   "CNOT gates between each pair of neighbouring qubits" then
#   "general single qubit unitary gates R(alpha, beta, gamma)".
#   "we only consider the measurement on the first qubit"
#   "the number of qubits equals to the square of kernel (filter) size"
#
#   Reported: conv1 (3x3 filter) = 9 x 3 x 2 = 54 params
#             conv2 (2x2 filter) = 4 x 3 x 2 = 24 params
#   -> 3 params/qubit (Rot), 2 repeats of the grouped variational block.
# --------------------------------------------------------------------------

CHEN_CONV1 = QuanvConfig(
    name="chen2020_conv1_3x3",
    n_qubits=9,
    encode_gates=("RY", "RZ"),
    encode_func="arctan_arctan_sq",
    ansatz_gates=("Rot",),
    entangler="chain",
    n_layers=2,            # "the grouped box ... may repeat several times"
    n_reuploads=1,
    measure_wires=(0,),
    n_channels=1,
    entangle_before_rotation=True,
)

CHEN_CONV2 = QuanvConfig(
    name="chen2020_conv2_2x2",
    n_qubits=4,
    encode_gates=("RY", "RZ"),
    encode_func="arctan_arctan_sq",
    ansatz_gates=("Rot",),
    entangler="chain",
    n_layers=2,
    n_reuploads=1,
    measure_wires=(0,),
    n_channels=1,
    entangle_before_rotation=True,
)

# --------------------------------------------------------------------------
# Oh, Choi, Kim -- arXiv:2009.09423, Fig. 3
#
#   4 qubits for a 2x2 filter, RX(a_ij) encoding, RZ/RX learnable gates
#   with controlled gates between.  Feature map returned with 8 channels.
# --------------------------------------------------------------------------

TUTORIAL_FIG3 = QuanvConfig(
    name="tutorial2020_fig3_2x2",
    n_qubits=4,
    encode_gates=("RX",),
    encode_func="pi_scaled",
    ansatz_gates=("RZ", "RX"),
    entangler="chain",
    n_layers=2,
    n_reuploads=1,
    measure_wires=(0,),
    n_channels=8,          # "returns a feature map with 8 channels"
    entangle_before_rotation=False,
)

# --------------------------------------------------------------------------
# Duffy et al. -- Phys. Rev. D 112, 092006 (2025), Eqs. (4)-(5)
#
#   RX rotation embedding; "nearest-neighbour entangling ansatz with two
#   trainable parameters per qubit, per entanglement block layer" (RY, RZ);
#   (U_theta U_F)^R reuploads.  2x2 windows, 3 output channels.
#   Table II: n_layers in {1,2}, n_filters in {1,2,3}, n_reuploads in {1,2}.
# --------------------------------------------------------------------------

PRD_QUANV = QuanvConfig(
    name="prd2025_quanv_2x2",
    n_qubits=4,
    encode_gates=("RX",),
    encode_func="identity",
    ansatz_gates=("RY", "RZ"),
    entangler="chain",
    n_layers=2,
    n_reuploads=2,
    measure_wires=(0,),
    n_channels=3,
    entangle_before_rotation=False,
)

ALL_CONFIGS = {c.name: c for c in
               (CHEN_CONV1, CHEN_CONV2, TUTORIAL_FIG3, PRD_QUANV)}

# Parameter counts the papers report, for validation.
PUBLISHED_PARAM_COUNTS = {
    "chen2020_conv1_3x3": 54,
    "chen2020_conv2_2x2": 24,
}
