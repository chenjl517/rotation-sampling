"""Parameter loader for the lookup-ansatz ground-state demo (run_std.py / run_rot.py)."""
from __future__ import annotations

import argparse
import os
import sys
import tomllib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def _resolve_path() -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    args, _ = pre.parse_known_args(sys.argv[1:])
    path = args.config or os.environ.get("LOCAL_ROT_GS_CONFIG") or os.path.join(SCRIPT_DIR, "config.toml")
    return os.path.abspath(os.path.expanduser(path))


CONFIG_PATH = _resolve_path()
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"config file not found: {CONFIG_PATH}")
with open(CONFIG_PATH, "rb") as _f:
    _cfg = tomllib.load(_f)

_lat, _ham, _ansatz = _cfg["lattice"], _cfg["hamiltonian"], _cfg["ansatz"]
_samp, _opt, _rot, _io = _cfg["sampling"], _cfg["optimization"], _cfg.get("rotation", {}), _cfg.get("io", {})

LX, LY = int(_lat["lx"]), int(_lat["ly"])
if LX != LY:
    raise ValueError("this demo uses nk.graph.Square and requires lx == ly")
N = LX * LY
PBC = bool(_lat["pbc"])
J1 = float(_ham["j1"])
J2 = float(_ham["j2"])

TOTAL_SZ = float(_ansatz.get("total_sz", 0.0))
INIT_STDDEV = float(_ansatz.get("init_stddev", 0.5))

SEED = int(_samp["seed"])
N_SAMPLES = int(_samp["n_samples"])
N_CHAINS = int(_samp["n_chains"])
N_DISCARD = int(_samp.get("n_discard", 1))
THIN_COEFF = int(_samp.get("thin_coeff", 1))

LR = float(_opt["lr"])
DIAG_SHIFT = float(_opt["diag_shift"])
SOLVER = str(_opt.get("solver", "cholesky"))
GRAD_CLIP = float(_opt.get("grad_clip", 100.0))
STD_SWEEPS = int(_opt["std_sweeps"])
ROT_SWEEPS = int(_opt["rot_sweeps"])

INIT_SWEEP = int(_rot.get("init_sweep", STD_SWEEPS))
ROT_SEED = int(_rot.get("seed", SEED))

_cfg_dir = os.path.dirname(CONFIG_PATH)
CKPT_DIR = os.path.abspath(os.path.join(_cfg_dir, str(_io.get("ckpt_dir", "ckpts"))))
OUTPUT_DIR = os.path.abspath(os.path.join(_cfg_dir, str(_io.get("output_dir", "results"))))
ED_DIR = os.path.abspath(os.path.join(_cfg_dir, str(_io.get("ed_dir", "ed_reference"))))
CKPT_EVERY = int(_io.get("ckpt_every", 100))
RUN_TAG = str(_io.get("run_tag", "")).strip()

GEOM_TAG = f"{LX}x{LY}_{'pbc' if PBC else 'obc'}"
_TAG = f"heis_{GEOM_TAG}_j1_{J1}_j2_{J2}"
HISTORY_TAG = _TAG if not RUN_TAG else f"{_TAG}_{RUN_TAG}"

ED_GS_PATH = os.path.join(ED_DIR, f"ed_gs_j2_{J2}.npz")
STD_CKPT_FMT = os.path.join(CKPT_DIR, f"std_ckpt_{HISTORY_TAG}_{{:06d}}.pkl")
ROT_CKPT_FMT = os.path.join(CKPT_DIR, f"rot_ckpt_{HISTORY_TAG}_{{:06d}}.pkl")
STD_HISTORY_PATH = os.path.join(OUTPUT_DIR, f"history_std_{HISTORY_TAG}.npz")
ROT_HISTORY_PATH = os.path.join(OUTPUT_DIR, f"history_rot_{HISTORY_TAG}.npz")
