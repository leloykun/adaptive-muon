from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import utils
from optax._src import transform
from optax._src.transform import ScaleByAdamState


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    adaptive_scale_min = -1.0,
    adaptive_scale_max = 1.0,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False,
    adaptive: bool = True,
) -> base.GradientTransformation:
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    nu = otu.tree_zeros_like(params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: b1 * m + (1 - b1) * g,
          otu.tree_bias_correction(mu, b1, numerics.safe_increment(count_inc)),
          otu.tree_bias_correction(updates, b1, count_inc),
      )
    else:
      mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
    # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
    # unclear why. Other Nadam implementations also omit the extra b2 factor.
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    updates = jax.tree.map(
        lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
        mu_hat,
        nu_hat,
        is_leaf=lambda x: x is None,
    )
    if adaptive:
      updates = jax.tree.map(
        lambda x, y: jnp.einsum('ij,ij->', x, y).clip(min=adaptive_scale_min, max=adaptive_scale_max) * y,
        mu_hat,
        updates,
      )
    mu = otu.tree_cast(mu, mu_dtype)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def adam(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    adaptive_scale_min: float = -1.0,
    adaptive_scale_max: float = 1.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
    adaptive: bool = True,
) -> base.GradientTransformation:
  return combine.chain(
      scale_by_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          adaptive_scale_min=adaptive_scale_min,
          adaptive_scale_max=adaptive_scale_max,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
          adaptive=adaptive,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )
