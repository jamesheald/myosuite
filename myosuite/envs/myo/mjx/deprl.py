import jax
import jax.numpy as jnp
from flax import struct
from typing import Any, Tuple, List
from myosuite.envs.myo.mjx import make
import time
from functools import partial

from flax.training.train_state import TrainState
import optax
import flax.linen as nn
from jax import random

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

  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)

  env = make(env_name)

  # Initialize batched environments
  mjx_model = env.mjx_model
  n_env = 1_024
  env_states = jax.vmap(lambda key: env.reset(key))(jax.random.split(subkey, n_env))
  dep_states = dep_init(mjx_model, n_env)

  tau_static = int(dep_states.params.tau)
  dep_step_jit = jax.jit(dep_step, static_argnums=2)

  from typing import NamedTuple
  from brax.training.acme.types import NestedArray
  class Transition(NamedTuple):
    """Container for a transition."""
    observation: NestedArray
    action: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray
  from brax.training import replay_buffers
  
  dummy_obs = jnp.zeros((env_states.obs['state'].shape[-1],))
  dummy_action = jnp.zeros((mjx_model.nu,))
  dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=dummy_obs,
      action=dummy_action,
      next_observation=dummy_obs,
      extras={'state_extras': {'truncation': 0.0}, 'policy_extras': {}},
  )

  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=n_env * 100,
      dummy_data_sample=dummy_transition,
      sample_batch_size=n_env,
  )
  key, subkey = jax.random.split(key)
  buffer_state = replay_buffer.init(subkey)

  def rollout(dep_state, env_state, buffer_state, num_steps):
    
    def step_env(carry, _):
        dep_s, env_s, buffer_state = carry
        obs = env_s.obs.copy()
        dep_s, action = dep_step_jit(dep_s, env_s, tau_static)       # dep_s, action: both (n_env, ...)
        env_s = jax.vmap(env.step)(env_s, action)    # vectorized env step over n_env
        transitions = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=obs,
            action=action,
            next_observation=env_s.obs,
            extras={},
        )
        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return (dep_s, env_s, buffer_state), None

    (dep_state, env_state, buffer_state), _ = jax.lax.scan(
        step_env,
        (dep_state, env_state, buffer_state),
        xs=None,
        length=num_steps,
    )
    return dep_state, env_state, buffer_state

  rollout_jit = jax.jit(rollout, static_argnums=3)

  # Run rollout
  t0 = time.time()
  dep_states, env_states, buffer_state = rollout_jit(dep_states, env_states, buffer_state, 100)
  t1 = time.time()

  print(f"Scan rollout (10_000 steps) took {t1 - t0:.4f} seconds")

  # reset dep and env with noise!

  class DynamicsNet(nn.Module):
    h_dims_dynamics: List
    num_control_variables: int
    drop_out_rates: List

    def setup(self):

        self.dynamics = [nn.Sequential([nn.Dense(features=h_dim), nn.LayerNorm(), nn.relu]) for h_dim in self.h_dims_dynamics]
        self.dynamics_out = nn.Dense(features=self.num_control_variables*2)

        self.dropout = [nn.Dropout(rate=drop_out_rate) for drop_out_rate in self.drop_out_rates]

    def __call__(self, obs, action, key, deterministic=False):

      def get_log_var(x):
        """
        sigma = log(1 + exp(x))
        """
        sigma = nn.softplus(x) + 1e-6
        log_var = 2 * jnp.log(sigma)
        return log_var

      x = jnp.concatenate((obs, action), axis=-1)       
      for fn, dropout in zip(self.dynamics, self.dropout):
          x = fn(x)
          key, subkey = jax.random.split(key)
          x = dropout(x, deterministic, subkey)

      x = self.dynamics_out(x)

      y_prime_mean, y_prime_scale = jnp.split(x, 2, axis=-1)
      y_prime_log_var = get_log_var(y_prime_scale)

      # fixed_log_sigma = jnp.log(1.)  # Fixed log_sigma, independent of a
      # return mu, fixed_log_sigma * jnp.ones_like(mu)  # Broadcast to [batch]

      return y_prime_mean, y_prime_log_var


  dynamics = DynamicsNet(h_dims_dynamics=[256,256],
                        num_control_variables=mjx_model.nv,
                        drop_out_rates=[0., 0.])
  
  key, subkey = jax.random.split(key)

  max_grad_norm = 0.5
  dynamics_learning_rate = 1e-4

  dynamics_state = TrainState.create(
      apply_fn=dynamics.apply,
      params=dynamics.init(subkey, dummy_obs, dummy_action, subkey, False),
      tx=
      optax.chain(
          optax.clip_by_global_norm(max_grad_norm),
          optax.adam(learning_rate=dynamics_learning_rate),
      ),
  )

  def dynamics_update(dynamics_state, buffer_state, key):

    def dynamics_loss_fn(params, key, observations, actions, next_observations):
      def log_likelihood_diagonal_Gaussian(x, mu, log_var):
          """
          Calculate the log likelihood of x under a diagonal Gaussian distribution
          var_min is added to the variances for numerical stability
          """
          log_likelihood = -0.5 * (log_var + jnp.log(2 * jnp.pi) + (x - mu) ** 2 / jnp.exp(log_var))
          return log_likelihood
          # return -(x - mu) ** 2
      y_prime_mean, y_prime_log_var = dynamics_state.apply_fn(params, observations, actions, key, deterministic=False)
      delta_obs = next_observations-observations
      dynamics_loss = -log_likelihood_diagonal_Gaussian(delta_obs[:, :mjx_model.nv], y_prime_mean, y_prime_log_var)
      return dynamics_loss.mean()

    key, subkey = jax.random.split(key)

    buffer_state, transitions = replay_buffer.sample(buffer_state)
    observations = transitions.observation
    actions = transitions.action
    next_observations = transitions.next_observation

    dynamics_loss, grads = jax.value_and_grad(dynamics_loss_fn, has_aux=False)(dynamics_state.params, subkey, observations, actions, next_observations)
    dynamics_state = dynamics_state.apply_gradients(grads=grads)

    return dynamics_state, dynamics_loss
  
  jit_dynamics_update = jax.jit(dynamics_update)

  # Training loop
  for step in range(100_000):

      key, subkey = jax.random.split(key)
      dynamics_state, dynamics_loss = jit_dynamics_update(dynamics_state, buffer_state, subkey)
      
      if step % 200 == 0:
          print(f"Step {step}, Loss: {dynamics_loss:.4f}")

  print("Training complete. Encoder params optimized.")

  breakpoint()

  # import imageio
  # import mujoco
  # renderer = mujoco.Renderer(env._mj_model, height=240, width=320)
  # video_writer = imageio.get_writer('/nfs/nhome/live/jheald/myosuite/myosuite/envs/myo/mjx/deprl_policy.mp4', fps=60)
  # jit_env_step = jax.jit(env.step)
  # env_state = env.reset(jax.random.PRNGKey(0))
  # for i in range(100):
  #   print(i)
  #   dep_states, action = dep_step_jit(dep_states, env_state, tau_static)       # dep_s, action: both (n_env, ...)
  #   env_state = jit_env_step(env_state, action[0,:])    # vectorized env step over n_env
  #   mj_data = mujoco.mjx.get_data(env._mj_model, env_state.data)
  #   renderer.update_scene(mj_data, camera='hand_side_inter')
  #   frame = renderer.render()
  #   video_writer.append_data(frame)

  # video_writer.close()

  breakpoint()

  # import jax
  # from jax import numpy as jp
  # from flax import struct
  # from mujoco_playground._src.mjx_env import State

  # import io
  # import imageio

  # import wandb

  # from mujoco_playground import wrapper

  # def make_minimal_state(full_state):
  #   """Create a minimal State suitable for rendering only."""

  #   @struct.dataclass
  #   class MinimalData:
  #       qpos: jp.ndarray
  #       qvel: jp.ndarray
  #       mocap_pos: jp.ndarray
  #       mocap_quat: jp.ndarray
  #       xfrc_applied: jp.ndarray

  #   minimal_data = MinimalData(
  #       qpos=full_state.data.qpos,
  #       qvel=full_state.data.qvel,
  #       mocap_pos=full_state.data.mocap_pos,
  #       mocap_quat=full_state.data.mocap_quat,
  #       xfrc_applied=full_state.data.xfrc_applied
  #   )

  #   return State(
  #       data=minimal_data,      # only the arrays used for rendering
  #       obs={},                 # empty dummy
  #       reward=jp.array(0.0),   # dummy
  #       done=jp.array(False),   # dummy
  #       metrics={},             # empty
  #       info={}                 # empty
  #   )

  # eval_env = wrapper.wrap_for_brax_training(env, episode_length=env._max_steps, action_repeat=1)
  # jit_env_reset = jax.jit(eval_env.reset)
  # jit_env_step = jax.jit(eval_env.step)
  # def make_jit_policy(make_policy, params, deterministic):
  #   return jax.jit(make_policy(params, deterministic))

  # def policy_params_fn(num_steps, make_policy, params):

  #   policy = make_jit_policy(make_policy, params, deterministic=True)

  #   # hand
  #   # cams = ['side_view',
  #   # 'front_view',
  #   # 'hand_top',
  #   # 'hand_bottom',
  #   # 'hand_side_inter',
  #   # 'hand_side_exter',
  #   # 'plam_lookat']

  #   cam_name = 'hand_side_inter'

  #   key = jax.random.PRNGKey(seed=num_steps)
  #   key, subkey = jax.random.split(key)
  #   state = jit_env_reset(rng=subkey[None,:])
  #   eval_env._mj_model.site_pos[eval_env._target_sids] = state.info['targets']

  #   rollout = [make_minimal_state(state)]
  #   for i in range(eval_env._max_steps):
  #       key, subkey = jax.random.split(key)
  #       action, _ = policy(state.obs, subkey[None, :])
  #       state = jit_env_step(state, action)
  #       rollout.append(make_minimal_state(state))

  #   frames = eval_env.render(rollout, height=240, width=320, camera=cam_name)

  #   video_bytes = io.BytesIO()
  #   imageio.mimwrite(video_bytes, frames, format='mp4')
  #   video_bytes.seek(0)

  #   wandb.log({"rollout_video": wandb.Video(video_bytes, format="mp4")}, step=num_steps)

  # return policy_params_fn





  # import imageio
  # import mujoco

  # def render_offscreen_and_save_video(dep_s, env, num_steps, video_path='output_video.mp4'):
  #     """
  #     Simulates the model and saves an offscreen rendering as a video.

  #     Args:
  #         mj_model (mujoco.MjModel): The standard MuJoCo model.
  #         mj_data (mujoco.MjData): The standard MuJoCo data.
  #         mjx_model (mjx.Model): The MJX model object.
  #         mjx_data (mjx.Data): The MJX data object.
  #         num_steps (int): The number of simulation steps to record.
  #         video_path (str): The path to save the video file.
  #     """

  #     print(f"\nRendering offscreen and saving video to '{video_path}' for {num_steps} steps...")

  #     # Create an offscreen renderer with a specific window size
  #     renderer = mujoco.Renderer(env._mj_model, height=240, width=320)
  #     video_writer = imageio.get_writer(video_path, fps=60)

  #     jit_reset = jax.jit(env.reset)
  #     jit_step = jax.jit(env.step)

  #     # Reset the environment to get the initial state
  #     rng = jax.random.PRNGKey(seed=1)
  #     state = jit_reset(rng=rng)

  #     # We use a standard loop because we need to update the scene at each step,
  #     # which is a side-effect that is incompatible with JAX JIT compilation.
  #     for i in range(num_steps):

  #         # act_rng, rng = jax.random.split(rng)
  #         dep_s, action = dep_step_jit(dep_s, state, tau_static)

  #         # Step the environment with the action
  #         state = jit_step(state, action[0,:])

  #         # Get the updated data from the GPU to the CPU for rendering.
  #         mj_data = mujoco.mjx.get_data(env._mj_model, state.data)

  #         # Update the renderer's scene and capture the image
  #         renderer.update_scene(mj_data, camera='hand_side_inter')
  #         frame = renderer.render()
  #         video_writer.append_data(frame)

  #     video_writer.close()
  #     print(f"Video saved successfully to '{video_path}'.")

  # render_offscreen_and_save_video(dep_states,
  #                                 env,
  #                                 1_000,
  #                                 video_path='/nfs/nhome/live/jheald/myosuite/myosuite/envs/myo/mjx/deprl_policy.mp4')

if __name__ == "__main__":
  main("MjxHandReachFixed-v0")
#   main("MjxElbowPoseRandom-v0")

# PYTHONPATH=/nfs/nhome/live/jheald/myosuite python deprl.py 

# class FlowNet(nn.Module):
#   h_dims_conditioner: int
#   num_bijector_params: int
#   num_coupling_layers: int
#   a_dim: int

#   def setup(self):

#     # final linear layer of each conditioner initialised to zero so that the flow is initialised to the identity function
#     self.conditioners = [nn.Sequential([nn.Dense(features=self.h_dims_conditioner), nn.relu,\
#                                         nn.Dense(features=self.h_dims_conditioner), nn.relu,\
#                                         nn.Dense(features=self.num_bijector_params*self.a_dim, bias_init=constant(jnp.log(jnp.exp(1.)-1.)), kernel_init=zeros_init())])
#                           for layer_i in range(self.num_coupling_layers)]
      
#   def __call__(self):

#     def make_flow():
    
#       mask = jnp.arange(self.a_dim) % 2 # every second element is masked
#       mask = mask.astype(bool)

#       def bijector_fn(params):
#           shift, arg_soft_plus = jnp.split(params, 2, axis=-1)
#           return distrax.ScalarAffine(shift=shift-jnp.log(jnp.exp(1.)-1.), scale=jax.nn.softplus(arg_soft_plus)+1e-3)
  
#       layers = []
#       for layer_i in range(self.num_coupling_layers):
#           layer = distrax.MaskedCoupling(mask=mask, bijector=bijector_fn, conditioner=self.conditioners[layer_i])
#           layers.append(layer)
#           mask = jnp.logical_not(mask) # flip mask after each layer
      
#       return distrax.Chain(layers)

#     return make_flow()
  
# class PrecoderNet(nn.Module):
#   h_dims_conditioner: int
#   num_bijector_params: int
#   num_coupling_layers: int
#   a_dim: int
#   def setup(self):
      
#     self.flow = FlowNet(h_dims_conditioner=self.h_dims_conditioner,
#                             num_bijector_params=self.num_bijector_params,
#                             num_coupling_layers=self.num_coupling_layers,
#                             a_dim=self.a_dim)
      
#   def __call__(self, z):
#       z_with_zeros = jnp.concatenate((z, jnp.zeros(z.shape)), axis=-1)
#       bijector = self.flow()
#       a = nn.tanh(bijector.forward(z_with_zeros))
#       return a