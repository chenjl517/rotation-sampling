import jax
import jax.numpy as jnp
import netket as nk


def _count_valid_single_flip_sites(σ, sz0: float):
    sz = jnp.sum(σ) / 2.0
    proposed_sz = sz - σ.astype(jnp.float64)
    valid_mask = jnp.abs(proposed_sz - sz0) <= 1.0 + 1e-6
    n_valid = jnp.sum(valid_mask)
    return valid_mask, n_valid


def _sample_masked_index(key, valid_mask):
    found = jnp.any(valid_mask)
    logits = jnp.where(valid_mask, 0.0, -1.0e30)
    idx = jax.random.categorical(key, logits)
    return found, idx.astype(jnp.int32)


class SingleSpinFlipWindowRule(nk.sampler.rules.MetropolisRule):
    """Flip exactly 1 spin while staying inside the Sz window {sz0-1, sz0, sz0+1}."""

    sz0: float
    """Central Sz sector of the physical state."""

    def __init__(self, sz0: float):
        self.sz0 = sz0

    def transition(rule, sampler, machine, parameters, state, key, σ):
        n_chains = σ.shape[0]
        keys = jax.random.split(key, n_chains)
        keys = jnp.asarray(keys)

        @jax.vmap
        def _propose(key, σ):
            valid_mask, n_valid = _count_valid_single_flip_sites(σ, rule.sz0)
            found, idx = _sample_masked_index(key, valid_mask)

            σp = σ.at[idx].set(-σ[idx])
            σp = jnp.where(found, σp, σ)

            _, n_valid_p = _count_valid_single_flip_sites(σp, rule.sz0)
            safe_n_valid = jnp.maximum(n_valid.astype(jnp.float64), 1.0)
            safe_n_valid_p = jnp.maximum(n_valid_p.astype(jnp.float64), 1.0)
            log_corr = jnp.log(safe_n_valid) - jnp.log(safe_n_valid_p)
            log_corr = jnp.where(found, log_corr, -jnp.inf)

            return σp, log_corr

        return _propose(keys, σ)

    def __repr__(self):
        return f"SingleSpinFlipWindowRule(sz0={self.sz0})"


class DoubleSpinFlipRule(nk.sampler.rules.MetropolisRule):
    """Flip two distinct random spins (symmetric proposal, keeps the Z parity)."""

    def transition(rule, sampler, machine, parameters, state, key, σ):
        n_chains, n_sites = σ.shape
        k1, k2 = jax.random.split(key)
        i = jax.random.randint(k1, (n_chains,), 0, n_sites)
        j = (i + jax.random.randint(k2, (n_chains,), 1, n_sites)) % n_sites
        rows = jnp.arange(n_chains)
        σp = σ.at[rows, i].multiply(-1)
        σp = σp.at[rows, j].multiply(-1)
        return σp, None

    def __repr__(self):
        return "DoubleSpinFlipRule()"
