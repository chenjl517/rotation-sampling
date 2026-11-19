"""Parameter loader for the 4x4 PBC J1-J2 XXZ local-quench demos."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def _cli_paths() -> tuple[str | None, str | None]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--model-config", default=None)
    args, _ = pre.parse_known_args(sys.argv[1:])
    return args.model_config, args.config


def _resolve_paths() -> tuple[str, str]:
    cli_model, cli_algo = _cli_paths()
    model = cli_model or os.environ.get("LOCAL_ROT_MODEL_CONFIG") or os.path.join(SCRIPT_DIR, "config_model.toml")
    algo = cli_algo or os.environ.get("LOCAL_ROT_CONFIG")
    if algo is None:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else ""
        candidate = os.path.join(script_dir, "config.toml") if script_dir else ""
        algo = candidate if candidate and os.path.exists(candidate) else os.path.join(SCRIPT_DIR, "tvmc_quench_4x4", "config.toml")
    return (os.path.abspath(os.path.expanduser(model)), os.path.abspath(os.path.expanduser(algo)))


def _load(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


MODEL_CONFIG_PATH, CONFIG_PATH = _resolve_paths()
_model = _load(MODEL_CONFIG_PATH)
_algo = _load(CONFIG_PATH)

_lat, _ham, _quench, _ansatz = _model["lattice"], _model["hamiltonian"], _model["quench"], _model["ansatz"]
_mio = _model.get("io", {})

LX, LY = int(_lat["lx"]), int(_lat["ly"])
N = LX * LY
PBC = bool(_lat["pbc"])

J1 = float(_ham["j1"])
J2 = float(_ham["j2"])
DELTA = float(_ham["delta"])

EXCI_POS = int(_quench["exci_pos"])
TOTAL_SZ_POST = float(_quench["total_sz_post"])

RBM_ALPHA = int(_ansatz["rbm_alpha"])
RBM_BIAS = bool(_ansatz.get("rbm_bias", True))

_model_dir = os.path.dirname(MODEL_CONFIG_PATH)
CKPT_DIR = os.path.abspath(os.path.join(_model_dir, str(_mio.get("ckpt_dir", "ckpts"))))
ED_DIR = os.path.abspath(os.path.join(_model_dir, str(_mio.get("ed_dir", "ed_reference"))))

GEOM_TAG = f"{LX}x{LY}_{'pbc' if PBC else 'obc'}"
_TAG = f"{GEOM_TAG}_j2_{J2}_delta_{DELTA}"
ED_GS_PATH = os.path.join(ED_DIR, f"ed_gs_{_TAG}.npz")
ED_DYN_PATH = os.path.join(ED_DIR, f"ed_dynamics_{_TAG}.npz")
RBM_GS_TAG = f"j2_{J2}_delta_{DELTA}_alpha{RBM_ALPHA}"
RBM_GS_PARAMS_PATH = os.path.join(CKPT_DIR, f"rbm_gs_params_{RBM_GS_TAG}.pkl")

_samp, _tvmc = _algo["sampling"], _algo["tvmc"]
_helper, _ptvmc, _io = _algo.get("helper", {}), _algo.get("ptvmc", {}), _algo.get("io", {})

SEED = int(_samp["seed"])
EVOL_SAMPLES = int(_samp["n_samples"])
EVOL_CHAINS = int(_samp["n_chains"])
MACHINE_POW = int(_samp.get("machine_pow", 2))
P_EXCHANGE = float(_samp.get("p_exchange", 1.0))

DT = float(_tvmc["dt"])
STD_SWEEPS = int(_tvmc["std_sweeps"])
ROT_SWEEPS = int(_tvmc.get("rot_sweeps", 0))
EVOL_DIAG_SHIFT = float(_tvmc.get("diag_shift", 1e-8))
KERNEL = str(_tvmc.get("kernel", "qgt"))

HELPER_SZ = float(_helper["sz"]) if "sz" in _helper else None
HELPER_COEFF = float(_helper.get("coeff", 0.0))

PTVMC_SOLVER = str(_ptvmc.get("solver", "LPE3")).upper()
PTVMC_N_ITER = int(_ptvmc.get("n_iter", 50))
PTVMC_LR = float(_ptvmc.get("lr", 0.05))
PTVMC_DIAG_SHIFT = float(_ptvmc.get("diag_shift", 1e-8))
PTVMC_AUTOTUNE = bool(_ptvmc.get("autotune", True))
PTVMC_AUTOTUNE_TARGET = float(_ptvmc.get("autotune_target", 0.9))
PTVMC_DIAG_SHIFT_MIN = float(_ptvmc.get("diag_shift_min", 1e-11))
PTVMC_DIAG_SHIFT_MAX = float(_ptvmc.get("diag_shift_max", 0.1))
PTVMC_SAMPLE_UPHI = bool(_ptvmc.get("sample_uphi", False))

RUN_TAG = str(_io.get("run_tag", "")).strip()
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(CONFIG_PATH), str(_io.get("output_dir", "results"))))
RUN_CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(CONFIG_PATH), str(_io.get("ckpt_dir", "ckpts"))))
HISTORY_TAG = _TAG if not RUN_TAG else f"{_TAG}_{RUN_TAG}"

ROT_CKPT_FMT = os.path.join(RUN_CKPT_DIR, f"rot_ckpt_{HISTORY_TAG}_{{:06d}}.pkl")
STD_CKPT_FMT = os.path.join(RUN_CKPT_DIR, f"std_ckpt_{HISTORY_TAG}_{{:06d}}.pkl")
ROT_HISTORY_PATH = os.path.join(OUTPUT_DIR, f"history_rotsr_{HISTORY_TAG}.npz")
STD_HISTORY_PATH = os.path.join(OUTPUT_DIR, f"history_std_{HISTORY_TAG}.npz")
ROT_INFIDELITY_PATH = os.path.join(OUTPUT_DIR, f"infidelity_rot_{HISTORY_TAG}.npz")
STD_INFIDELITY_PATH = os.path.join(OUTPUT_DIR, f"infidelity_std_{HISTORY_TAG}.npz")
