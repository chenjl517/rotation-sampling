"""Parameter loader for the parity-mixing demo (TFIM chain, arXiv:2603.18148 Fig. 5a)."""
from __future__ import annotations

import argparse
import os
import sys
import tomllib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path() -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    args, _ = pre.parse_known_args(sys.argv[1:])
    path = args.config or os.environ.get("LOCAL_ROT_CONFIG") or os.path.join(SCRIPT_DIR, "config.toml")
    return os.path.abspath(os.path.expanduser(path))


CONFIG_PATH = _resolve_path()
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"config file {CONFIG_PATH} not found: pass --config PATH or set LOCAL_ROT_CONFIG")
with open(CONFIG_PATH, "rb") as _f:
    _cfg = tomllib.load(_f)

_lat, _ham, _ansatz = _cfg["lattice"], _cfg["hamiltonian"], _cfg["ansatz"]
_rot, _samp, _tvmc = _cfg.get("rotation", {}), _cfg["sampling"], _cfg["tvmc"]
_ed, _io = _cfg.get("ed", {}), _cfg.get("io", {})

LX, LY = int(_lat["lx"]), int(_lat["ly"])
N = LX * LY
PBC = bool(_lat["pbc"])

J = float(_ham["J"])
H_FIELD = float(_ham["h"])

RBM_ALPHA = int(_ansatz.get("rbm_alpha", 4))
RBM_VISIBLE_BIAS = bool(_ansatz.get("rbm_visible_bias", True))
RBM_HIDDEN_BIAS = bool(_ansatz.get("rbm_hidden_bias", True))
PERTURB = float(_ansatz.get("perturb", 1e-5))
PERTURB_DEAD = float(_ansatz.get("perturb_dead", 0.0))
PROJECTOR_UNITS = int(_ansatz.get("projector_units", 1))
if not RBM_HIDDEN_BIAS:
    raise ValueError("the analytic even-parity RBM construction needs rbm_hidden_bias = true")

ROT_XY = tuple(int(v) for v in _rot.get("site", [0, 0]))

SEED = int(_samp["seed"])
EVOL_SAMPLES = int(_samp["n_samples"])
EVOL_CHAINS = int(_samp["n_chains"])
MACHINE_POW = int(_samp.get("machine_pow", 2))

DT = float(_tvmc["dt"])
ROT_SWEEPS = int(_tvmc.get("rot_sweeps", 0))
STD_SWEEPS = int(_tvmc.get("std_sweeps", 0))
EVOL_DIAG_SHIFT = float(_tvmc.get("diag_shift", 0.0))
KERNEL = str(_tvmc.get("kernel", "qgt"))
SOLVER = str(_tvmc.get("solver", "pinv_smooth")).lower()
RTOL = float(_tvmc.get("rtol", 1e-14))
RTOL_SMOOTH = float(_tvmc.get("rtol_smooth", 1e-10))
E_GS_SHIFT = bool(_tvmc.get("e_gs_shift", True))
INTEGRATOR = str(_tvmc.get("integrator", "rk4")).lower()
if INTEGRATOR not in ("rk4", "rk45"):
    raise ValueError(f"[tvmc] integrator must be 'rk4' or 'rk45', got {INTEGRATOR!r}")
RK45_RTOL = float(_tvmc.get("rk45_rtol", 1e-4))
RK45_ATOL = float(_tvmc.get("rk45_atol", 1e-8))
DT_MIN = float(_tvmc.get("dt_min", 1e-5))
DT_MAX = float(_tvmc.get("dt_max", 1e-2))
T_MAX = float(_tvmc.get("t_max", 0.0))
MAX_STEPS = int(_tvmc.get("max_steps", 100000))
SAMPLE_POLICY = str(_tvmc.get("sample_policy", "fresh")).lower()
RK45_NORM = str(_tvmc.get("rk45_norm", "l2")).lower()
if SAMPLE_POLICY not in ("fresh", "cached") or RK45_NORM not in ("l2", "qgt"):
    raise ValueError(f"[tvmc] sample_policy must be fresh|cached and rk45_norm l2|qgt, got {SAMPLE_POLICY!r}, {RK45_NORM!r}")

ED_DT = float(_ed.get("dt", 0.01))
ED_T_MAX = float(_ed.get("t_max", 2.0))

_cfg_dir = os.path.dirname(CONFIG_PATH)
OUTPUT_DIR = os.path.abspath(os.path.join(_cfg_dir, str(_io.get("output_dir", "results"))))
CKPT_DIR = os.path.abspath(os.path.join(_cfg_dir, str(_io.get("ckpt_dir", "ckpts"))))
ED_DIR = os.path.abspath(os.path.join(_cfg_dir, str(_io.get("ed_dir", "ed_reference"))))
RUN_TAG = str(_io.get("run_tag", "")).strip()

GEOM_TAG = f"{LX}x{LY}_{'pbc' if PBC else 'obc'}"
_INIT_TAG = f"alpha{RBM_ALPHA}_proj{PROJECTOR_UNITS}_pd{PERTURB_DEAD:g}" \
    + ("" if RBM_VISIBLE_BIAS else "_nova") + ("" if RBM_HIDDEN_BIAS else "_nohb")
_TAG = f"tfim_{GEOM_TAG}_J{J}_h{H_FIELD}_{_INIT_TAG}"
HISTORY_TAG = _TAG if not RUN_TAG else f"{_TAG}_{RUN_TAG}"

ED_DYN_PATH = os.path.join(ED_DIR, f"ed_dynamics_{_TAG}.npz")
ROT_CKPT_FMT = os.path.join(CKPT_DIR, f"rot_ckpt_{HISTORY_TAG}_{{:06d}}.pkl")
STD_CKPT_FMT = os.path.join(CKPT_DIR, f"std_ckpt_{HISTORY_TAG}_{{:06d}}.pkl")
ROT_HISTORY_PATH = os.path.join(OUTPUT_DIR, f"history_rot_{HISTORY_TAG}.npz")
STD_HISTORY_PATH = os.path.join(OUTPUT_DIR, f"history_std_{HISTORY_TAG}.npz")
