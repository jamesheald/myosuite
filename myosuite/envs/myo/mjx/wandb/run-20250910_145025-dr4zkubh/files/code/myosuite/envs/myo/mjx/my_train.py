"""Train a PPO agent using brax."""

import os
os.environ["MUJOCO_GL"] = "egl"

import functools
import time
import pickle
import jax
print(f"Current backend: {jax.default_backend()}")
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from myosuite.envs.myo.mjx import ppo_config

from myosuite.envs.myo.mjx import make, get_default_config
from mujoco_playground import wrapper

import wandb

def main(env_name):
  """Run training and evaluation for the specified environment."""

  env, ppo_params, network_factory = load_env_and_network_factory(env_name)

  wandb_run = wandb.init(
                        project=env_name,
                        # config=dict(config),
                        config=dict(ppo_params),
                        # group=wandb_cfg.group,
                        # sync_tensorboard=True,
                        # monitor_gym=True,
                        save_code=True,
                        # name=name,
                        # notes=notes,
                        # id=None,
                        # resume=run_id is not None
                    )
  import numpy as np

  from mujoco_playground import wrapper

  eval_env = wrapper.wrap_for_brax_training(env, episode_length=150, action_repeat=1)
  # The function that generates the video, now outside of the Brax loop
  def policy_params_fn(num_steps, make_policy, params, eval_it=0, eval_env=None):

      if eval_it % 1 == 0:
          
          jit_env_reset = jax.jit(eval_env.reset)
          jit_env_step = jax.jit(eval_env.step)
          def make_jit_policy(make_policy, params, deterministic):
              return jax.jit(make_policy(params, deterministic))

          import io
          import imageio

          
          policy = make_jit_policy(make_policy, params, deterministic=True)

          key = jax.random.PRNGKey(seed=1)
          key, subkey = jax.random.split(key)
          state = jit_env_reset(rng=subkey[None,:])

          frames = []
          frame = eval_env.render(state, height=240, width=320)#, camera='skew_side')
          frames.append(frame)
          for _ in range(150):
              key, subkey = jax.random.split(key)
              action, _ = policy(state.obs, subkey[None, :])
              state = jit_env_step(state, action)   # JIT-safe step
              frame = eval_env.render(state, height=240, width=320)#, camera='skew_side')
              frames.append(frame)

          video_bytes = io.BytesIO()
          imageio.mimwrite(video_bytes, np.stack(frames, axis=0), format='mp4') #, codec='libx264')
          video_bytes.seek(0)

          wandb.log({"rollout_video": wandb.Video(video_bytes, format="mp4")}, step=num_steps)

  policy_params_fn = functools.partial(policy_params_fn, eval_env=eval_env)

  ppo_params['policy_params_fn'] = policy_params_fn

  # Train the model
  make_inference_fn, params, _ = ppo.train(
      environment=env,
      progress_fn=progress,
      network_factory=network_factory,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=ppo_params.pop("num_eval_envs"),
      **ppo_params,
  )

  print(f"Time to JIT compile: {times[1] - times[0]}")
  print(f"Time to train: {times[-1] - times[1]}")

  with open('playground_params.pickle', 'wb') as handle:
      pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_env_and_network_factory(env_name):
  env = make(env_name)
  ppo_params = dict(ppo_config)

  print(f"Training on environment:\n{env_name}")
  print(f"Environment Config:\n{get_default_config(env_name)}")
  print(f"PPO Training Parameters:\n{ppo_config}")

  if "network_factory" in ppo_params:
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.pop("network_factory")
    )
  else:
    network_factory = ppo_networks.make_ppo_networks

  return env, ppo_params, network_factory


times = [time.monotonic()]
# Progress function for logging
def progress(num_steps, metrics):
  times.append(time.monotonic())
  print(f"Step {num_steps} at {times[-1]}: reward={metrics['eval/episode_reward']:.3f}")
  wandb.log(metrics, step=num_steps)


if __name__ == "__main__":
  main("MjxHandReachFixed-v0")
