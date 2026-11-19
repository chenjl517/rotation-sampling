"""Checkpoint save/load utilities for VMC optimization."""

import pickle
from pathlib import Path


def save_checkpoint(checkpoint_path: str, vstate_base, history: list, config: dict):
    """Save optimization checkpoint."""
    checkpoint = {
        'parameters': vstate_base.parameters,
        'sampler_state': getattr(vstate_base, 'sampler_state', None),
        'history': history,
        'config': config,
    }
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load optimization checkpoint."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint
