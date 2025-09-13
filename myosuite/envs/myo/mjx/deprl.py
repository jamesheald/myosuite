import jax
import jax.numpy as jnp
from flax import struct
from typing import Any, Dict, Tuple
from myosuite.envs.myo.mjx import make
import time
from functools import partial

@struct.dataclass
class DEPParams:
  kappa: float = 100.0
  tau: int = 20
  bias_rate: float = 0.00002
  time_dist: int = 5
  s4avg: int = 2
  buffer_size: int = 200
  sensor_delay: int = 1
  regularization: int = 32
  with_learning: bool = True
  force_scale: float = 0.0003 #  0.04


@struct.dataclass
class DEPState:
  # static sizes
  num_sensors: int
  num_motors: int
  n_env: int

  # controller matrices
  M: jnp.ndarray           # shape (n_env, num_motors, num_sensors)
  C: jnp.ndarray           # shape (n_env, num_motors, num_sensors)
  C_norm: jnp.ndarray      # same shape
  Cb: jnp.ndarray          # shape (n_env, num_motors)

  # smoothed observation
  obs_smoothed: jnp.ndarray  # (n_env, num_sensors)

  # circular buffer: separate obs & actions for clarity
  buffer_obs: jnp.ndarray   # (buffer_size, n_env, num_sensors)
  buffer_act: jnp.ndarray   # (buffer_size, n_env, num_motors)

  pointer: jnp.ndarray      # scalar int, next write index
  size: jnp.ndarray         # scalar int, how many valid entries
  t: jnp.ndarray            # time step counter

  params: DEPParams
  act_scale: jnp.ndarray    # shape (num_motors,) or scalar
  min_muscle: jnp.ndarray
  max_muscle: jnp.ndarray
  min_force: jnp.ndarray
  max_force: jnp.ndarray


def dep_init(mjx_model: Any, n_env: int, params: DEPParams = DEPParams()) -> DEPState:
  """Initialize state from mujoco model metadata (or sizes)."""
  num_sensors = int(mjx_model.nu)   # adjust if sensors differ
  num_motors = int(mjx_model.nu)

  M = jnp.broadcast_to(-jnp.eye(num_motors, num_sensors), (n_env, num_motors, num_sensors))
  C = jnp.zeros((n_env, num_motors, num_sensors))
  C_norm = jnp.zeros_like(C)
  Cb = jnp.zeros((n_env, num_motors))
  obs_smoothed = jnp.zeros((n_env, num_sensors))
  buffer_obs = jnp.zeros((params.buffer_size, n_env, num_sensors))
  buffer_act = jnp.zeros((params.buffer_size, n_env, num_motors))
  pointer = jnp.array(0, dtype=jnp.int32)
  size = jnp.array(0, dtype=jnp.int32)
  t = jnp.array(0, dtype=jnp.int32)
  act_scale = jnp.ones((num_motors,))  # or scalar as needed
  min_muscle = jnp.ones((n_env, num_motors)) * 100.0
  max_muscle = jnp.zeros((n_env, num_motors))
  min_force = jnp.ones((n_env, num_motors))  * 100.0
  max_force = -jnp.ones((n_env, num_motors)) * 100.0

  return DEPState(
      num_sensors=num_sensors,
      num_motors=num_motors,
      n_env=n_env,
      M=M,
      C=C,
      C_norm=C_norm,
      Cb=Cb,
      obs_smoothed=obs_smoothed,
      buffer_obs=buffer_obs,
      buffer_act=buffer_act,
      pointer=pointer,
      size=size,
      t=t,
      params=params,
      act_scale=act_scale,
      min_muscle=min_muscle,
      max_muscle=max_muscle,
      min_force=min_force,
      max_force=max_force,
  )


# ---- small utilities ----
def _add_to_buffer(buffer: jnp.ndarray, pointer: jnp.ndarray, value: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Write `value` at `pointer` row of buffer, return new buffer and (pointer+1) mod N."""
  N = buffer.shape[0]
  buffer = buffer.at[pointer].set(value)
  pointer = (pointer + 1) % N
  return buffer, pointer


def _get_recent_index(pointer: jnp.ndarray, s: int, N: int) -> jnp.ndarray:
  """Return index of the s-th most recent (1 => latest)."""
  return (pointer - s) % N


# ---- main computations ----
def compute_obs_from_envstate(env_state: Any, dep: DEPState, eps=0.1) -> jnp.ndarray:
  """
  env_state is assumed to have fields:
    env_state.data._impl.actuator_length  -> shape (n_env, num_sensors)
    env_state.data.actuator_force         -> shape (n_env, num_sensors)  (or adapt accordingly)
  Returns normalized obs of shape (n_env, num_sensors)
  """
  # read arrays (adapt field names to your mjx Data layout)
  lce = env_state.data._impl.actuator_length   # (n_env, num_sensors)
  f = env_state.data.actuator_force            # (n_env, num_sensors) - adapt if name differs

  # update running max/min safely
  max_muscle = jnp.maximum(dep.max_muscle, lce)
  min_muscle = jnp.minimum(dep.min_muscle, lce)
  max_force = jnp.maximum(dep.max_force, f)
  min_force = jnp.minimum(dep.min_force, f)

  # return updated dep
  dep = dep.replace(
    max_muscle=max_muscle,
    min_muscle=min_muscle,
    max_force=max_force,
    min_force=min_force,
  )

  p = dep.params
  # safe normalization
  norm_len = (lce - dep.min_muscle) / (dep.max_muscle - dep.min_muscle + eps)
  norm_force = (f - dep.min_force) / (dep.max_force - dep.min_force + eps)

  obs = ( (norm_len - 0.5) * 2.0 ) + p.force_scale * norm_force
  return obs, dep


def _q_norm(q: jnp.ndarray, reg_power: int) -> jnp.ndarray:
  reg = 10.0 ** (-reg_power)
  # denom = jnp.linalg.norm(q, axis=-1, keepdims=True) + reg
  # return 1.0 / denom.squeeze(-1)
  denom = jnp.linalg.norm(q, axis=-1) + reg
  return 1.0 / denom


def _compute_action_from_C(dep: DEPState) -> jnp.ndarray:
  # q = C_norm @ obs_smoothed  => shapes: (n_env, num_motors, num_sensors) x (n_env, num_sensors) -> (n_env, num_motors)
  q = jnp.einsum("ijk, ik->ij", dep.C_norm, dep.obs_smoothed)
  qnorm = _q_norm(q, dep.params.regularization)
  q = q * qnorm[:, None]   # broadcast per env
  y = jnp.tanh(q * dep.params.kappa + dep.Cb)
  # clip then scale
  y = jnp.clip(y, -1.0, 1.0)
  y = y * dep.act_scale  # scale actions
  return y


def learn_controller(dep: DEPState, tau: int) -> DEPState:
  """Update C, C_norm, and Cb bias using buffer state. Returns updated DEPState."""
  C_new = compute_C_from_buffer(dep, tau)
  R = jnp.einsum("ijk, imk->ijm", C_new, dep.M)  # shape (n_env, n_motors, n_motors)
  reg = 10.0 ** (-dep.params.regularization)
  factor = dep.params.kappa / (jnp.linalg.norm(R, axis=-1) + reg)  # (n_env, n_motors)
  C_norm = jnp.einsum("ijk,ik->ijk", C_new, factor)

  # update biases with last action in buffer
  N = dep.params.buffer_size
  last_idx = _get_recent_index(dep.pointer, 2, N)
  yy = dep.buffer_act[last_idx]  # shape (n_env, num_motors)
  Cb_new = dep.Cb - (jnp.clip(yy * dep.params.bias_rate, -0.05, 0.05) + dep.Cb * 0.001)

  return dep.replace(C=C_new, C_norm=C_norm, Cb=Cb_new)

def compute_C_from_buffer(dep: DEPState, tau: int) -> jnp.ndarray:
  N = dep.params.buffer_size
  pointer = dep.pointer
  t = dep.t
  time_dist = dep.params.time_dist
  # tau = dep.params.tau
  max_s = jnp.maximum(0, jnp.minimum(t - time_dist, tau) - 2)
  # s_vals = jnp.arange(20)  # fixed length
  s_vals = jnp.arange(tau)  # fixed length

  def compute_one(s):
      s_loop    = s + 2
      x_idx     = _get_recent_index(pointer, s_loop, N)
      xx_idx    = _get_recent_index(pointer, s_loop + 1, N)
      xx_t_idx  = _get_recent_index(pointer, s_loop + time_dist, N)
      xxx_t_idx = _get_recent_index(pointer, s_loop + 1 + time_dist, N)

      x     = dep.buffer_obs[x_idx]
      xx    = dep.buffer_obs[xx_idx]
      xx_t  = dep.buffer_obs[xx_t_idx]
      xxx_t = dep.buffer_obs[xxx_t_idx]

      chi = x - xx
      v   = xx_t - xxx_t
      mu  = jnp.einsum("ijk,ik->ij", dep.M, chi)

      return jnp.einsum("ij, ik->ijk", mu, v)

  contribs = jax.vmap(compute_one)(s_vals)  # (tau, n_env, n_motors, n_sensors)

  # Mask out contributions where s >= max_s
  mask = (s_vals < max_s).astype(dep.C.dtype)[:, None, None, None]
  contribs = contribs * mask

  return contribs.sum(axis=0)  # (n_env, n_motors, n_sensors)

# ---- top-level step: pure function ----
def dep_step(dep: DEPState, env_state: Any, tau: int) -> Tuple[DEPState, jnp.ndarray]:
  """
  Given current DEPState and env_state (mujoco/jax state providing actuator_length and actuator_force),
  compute observation, optionally learn, compute action, and return updated DEPState and action.
  """

  obs, dep = compute_obs_from_envstate(env_state, dep)

  s4 = dep.params.s4avg
  obs_smoothed = jnp.where((s4 > 1) & (dep.t > 0), dep.obs_smoothed + (obs - dep.obs_smoothed) / s4, obs)

  buffer_obs_new, pointer_new = _add_to_buffer(dep.buffer_obs, dep.pointer, obs_smoothed)
  
  dep = dep.replace(obs_smoothed=obs_smoothed,
                    buffer_obs=buffer_obs_new,
                    pointer=pointer_new,
                    size=jnp.minimum(dep.size + 1, dep.params.buffer_size))

  cond = jnp.logical_and((dep.params.with_learning), (dep.size > (2 + dep.params.time_dist)))
  dep = jax.lax.cond(cond,
                       lambda d: learn_controller(d, tau),
                       lambda d: d,
                       dep)

  y = _compute_action_from_C(dep) 

  buffer_act_new, _ = _add_to_buffer(dep.buffer_act, dep.pointer - 1, y)

  dep = dep.replace(buffer_act=buffer_act_new,
                    t=dep.t + 1)

  return dep, y

def main(env_name):
  """Run training and evaluation for the specified environment."""

  env = make(env_name)

  # Initialize batched environments
  mjx_model = env.mjx_model
  n_env = 1_000
  env_states = jax.vmap(lambda key: env.reset(key))(jax.random.split(jax.random.PRNGKey(0), n_env))
  dep_states = dep_init(mjx_model, n_env)

  tau_static = int(dep_states.params.tau)
  dep_step_jit = jax.jit(dep_step, static_argnums=2)

  def rollout(dep_state, env_state, num_steps: int):
    
    def step_env(carry, _):
        dep_s, env_s = carry
        dep_s, action = dep_step_jit(dep_s, env_s, tau_static)       # dep_s, action: both (n_env, ...)
        env_s = jax.vmap(env.step)(env_s, action)    # vectorized env step over n_env
        return (dep_s, env_s), None

    (dep_state, env_state), _ = jax.lax.scan(
        step_env,
        (dep_state, env_state),
        xs=None,
        length=num_steps,
    )
    return dep_state, env_state

  rollout_jit = jax.jit(rollout, static_argnums=2)

  # Run rollout
  t0 = time.time()
  dep_states, env_states = rollout_jit(dep_states, env_states, 100)
  t1 = time.time()

  print(f"Scan rollout (10_000 steps) took {t1 - t0:.4f} seconds")

  breakpoint()

if __name__ == "__main__":
  main("MjxHandReachFixed-v0")
#   main("MjxElbowPoseRandom-v0")