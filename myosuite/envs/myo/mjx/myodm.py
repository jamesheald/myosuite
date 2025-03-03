import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datetime import datetime
from etils import epath
import functools
from typing import Any, Dict, Sequence, Tuple, Union
import time

import jax
from jax import numpy as jp
import numpy as np
from jax import nn

import mujoco
from mujoco import mjx

from myosuite.logger.reference_motion import ReferenceMotion

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import train as sac
from brax.io import html, mjcf, model

from matplotlib import pyplot as plt
import mediapy as media

# from jax import config
# config.update('jax_disable_jit', True)
# config.update("jax_debug_nans", True)

# changes:
# get rid of margins, certain collisions, ellipsoid skin objects, sidesites

class TrackEnv(PipelineEnv):

  # https://github.com/MyoHub/myosuite/blob/main/myosuite/envs/myo/assets/hand/myohand_object.xml

  def __init__(
      self,
      init_qpos,
      object_name='stanfordbunny',
      # object_name='cylinderlarge',
      model_path='/../assets/hand/myohand_object.xml',
      episode_length=75,
      normalize_act=True,
      frame_skip=10,
      **kwargs,
  ):
    
    cwd = os.path.dirname(os.path.abspath(__file__))
    time_stamp = str(time.time())

    # process model_path to import the right object
    with open(cwd + model_path, "r") as file:
        processed_xml = file.read()
        processed_xml = processed_xml.replace("OBJECT_NAME", object_name)
    processed_model_path = (
        cwd + model_path[:-4] + time_stamp + "_processed.xml"
    )

    with open(processed_model_path, "w") as file:
        file.write(processed_xml)
  
    mj_model = mujoco.MjModel.from_xml_path(processed_model_path)

    os.remove(processed_model_path)

    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    mj_model.opt.timestep = 0.002 # dt = mj_model.opt.timestep * frame_skip

    sys = mjcf.load_model(mj_model)

    del mj_model

    kwargs['n_frames'] = kwargs.get('n_frames', frame_skip)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    # def _setup(self, rng: jp.ndarray) -> State:
        
    # self._motion_start_time = motion_start_time
    # self._target_sid = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'target')
    # self._object_bid = mujoco.mj_name2id(env.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'elbow')

    self.obj_err_scale = 50
    
    
    self.lift_bonus_thresh = 0.02
    # self._lift_z = self.sim.data.xipos[self._object_bid][2] + self.lift_bonus_thresh

      # return 

    # self.obs_keys = ['qpos', 'qvel', 'act', 'time'] # need to add reference trajectory
    # self.rwd_keys_wt = {
    #     'object': 1.0,
    #     'lift': 4.0,
    #     'act_reg': 50,
    # }

    self._episode_length = episode_length
    self._normalize_act = normalize_act

    self._init_qpos = init_qpos

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    data = self.pipeline_init(self._init_qpos, self._init_qpos*0.)

    # obs_dict = self._get_obs_dict(data)
    # obs = self._get_obs(obs_dict)
    obs = self._get_obs_vec(data)

    reward, done, zero = jp.zeros(3)

    # metrics = {
    #     'object': zero,
    #     'lift': zero,
    #     'act_reg': zero,
    #     'solved': zero,
    # }

    # return State(data, obs, reward, done, metrics)
    return State(data, obs, reward, done)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    data0 = state.pipeline_state

    # if self._normalize_act:
      # action = nn.sigmoid(5.0 * (action - 0.5))

    # action = nn.sigmoid(5.0 * (action - 0.5))

    data = self.pipeline_step(data0, action)

    # obs_dict = self._get_obs_dict(data)
    # obs = self._get_obs(obs_dict)
    # rwd_dict = self._get_reward(obs_dict)

    obs = self._get_obs_vec(data)
    # rwd_dict = self._get_reward({})

    # state.metrics.update(
    #     object=rwd_dict['object'],
    #     lift=rwd_dict['lift'],
    #     act_reg=-rwd_dict['act_reg'],
    #     solved=rwd_dict['solved'],
    # )

    return state.replace(
        pipeline_state=data, obs=obs, reward=obs.mean(), done=0.
        # pipeline_state=data, obs=obs, reward=rwd_dict['dense'], done=rwd_dict['done']
    )

  # def _get_obs_dict(self, data: mjx.Data) -> jp.ndarray:

  #   obs_dict = {}
  #   obs_dict['time'] = jp.atleast_1d(data.time)
  #   obs_dict['qpos'] = data.qpos.copy()
  #   obs_dict['qvel'] = data.qvel.copy()
  #   obs_dict['act'] = data.act.copy()

  #   return obs_dict

  # def _get_obs(self, obs_dict: Dict[str, jp.ndarray]) -> jp.ndarray:

  #   obs_list = [jp.zeros(0)]
  #   for key in self.obs_keys:
  #       obs_list.append(obs_dict[key].ravel()) # ravel helps with images
  #   obsvec = jp.concatenate(obs_list)

  #   return obsvec

  def _get_obs_vec(self, data: mjx.Data) -> jp.ndarray:

    obsvec = jp.concatenate((jp.atleast_1d(data.time), data.qpos.copy(), data.qvel.copy(), data.act.copy()))

    return obsvec
  
  # def _norm2(self, x: jp.ndarray) -> jp.ndarray:
   
  #   return jp.sum(jp.square(x))

  # def _get_reward(
  #     self, obs_dict: Dict[str, jp.ndarray],
  # ) -> jp.ndarray:

  #   obj_com_err = 1. # jp.sqrt(self.norm2(tgt_obj_com - obj_com))
  #   obj_rot_err = 1. # self.rotation_distance(obj_rot, tgt_obj_rot, False) / jp.pi
  #   obj_reward = jp.exp(-self.obj_err_scale * (obj_com_err + 0.1 * obj_rot_err))

  #   lift_bonus = 1. #(tgt_obj_com[2] >= self._lift_z) and (obj_com[2] >= self._lift_z)

  #   act_mag = 1. # jp.linalg.norm(obs_dict['act'], axis=-1)/self.sys.nu if self.sys.nu !=0 else 0

  #   rwd_dict =  {
  #           # Optional Keys
  #           'object':   1.*obj_reward,
  #           'lift':   1.*lift_bonus,
  #           'act_reg': -1.*act_mag,
  #           # 'penalty': -1.*(reach_dist>far_th),
  #           # Must keys
  #           'sparse':  -1.,
  #           'solved':  0.,
  #           'done':    0.,}
  #           # 'sparse':  -1.*reach_dist,
  #           # 'solved':  1.*(reach_dist<near_th),
  #           # 'done':    1.*(reach_dist>far_th),}

  #   rwd_dict['dense'] = sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])

  #   return rwd_dict

def main():

  cwd = os.path.dirname(os.path.abspath(__file__))
  reference = cwd + '/../myodm/data/' + 'MyoHand_stanfordbunny_inspect1.npz'
  # reference = cwd + '/../myodm/data/' + 'MyoHand_cylinderlarge_inspect1.npz'
  motion_extrapolation = True # hold the last frame if motion is over
  motion_start_time = 0. # useful to skip initial motion

  # prep reference
  ref = ReferenceMotion(
      reference_data=reference,
      motion_extrapolation=motion_extrapolation,
      random_generator=np.random,
  )

  def _quat2mat(quat):
    """ Convert Quaternion to Euler Angles """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    _FLOAT_EPS = np.finfo(np.float64).eps

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))

  def _mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    _FLOAT_EPS = np.finfo(np.float64).eps
    _EPS4 = _FLOAT_EPS * 4.0

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                            -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                            -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                            -np.arctan2(-mat[..., 0, 2], cy),
                            -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                            -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                            0.0)
    return euler

  def _quat2euler(quat):
    """ Convert Quaternion to Euler Angles """
    return _mat2euler(_quat2mat(quat))
  
  robot_init, object_init = ref.get_init()
  init_qpos = np.empty(35) # np.empty(self.sys.nq)
  init_qpos[: ref.robot_dim] = robot_init
  init_qpos[ref.robot_dim : ref.robot_dim + 3] = object_init[:3]
  init_qpos[-3:] = _quat2euler(object_init[3:])

  del ref, robot_init, object_init

  print(jax.devices())
  from jax import config
  config.update('jax_disable_jit', False)
  config.update('jax_debug_nans', True)
  # config.update('jax_enable_x64', False)

  env_name = 'myoFingerReachFixed-MJX-v0'
  envs.register_environment(env_name, TrackEnv)
  env = envs.get_environment(env_name, init_qpos=jp.array(init_qpos))

  def _render(rollouts, video_type='single'):

    videos = []
    for rollout in rollouts:
      
      # change the target position of the environment for rendering
      env.sys.mj_model.site_pos[env._target_sids] = rollout['IFtip_target']

      if video_type == 'single':
        videos += env.render(rollout['states'])
      elif video_type == 'multiple':
        videos.append(env.render(rollout['states']))

    if video_type == 'single':
      media.write_video(cwd + '/video.mp4', videos, fps=1.0 / env.dt, qp=18) 
    elif video_type == 'multiple':
      for i, video in enumerate(videos):
        media.write_video(cwd + '/video' + str(i) + '.mp4', video, fps=1.0 / env.dt, qp=18) 

    return None

  # train_fn = functools.partial(
  #     ppo.train, num_timesteps=1_000, num_evals=1, reward_scaling=0.1,
  #     episode_length=env._episode_length, normalize_observations=True, action_repeat=1,
  #     unroll_length=1, num_minibatches=1, num_updates_per_batch=1,
  #     discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=1, # num_envs=3072,
  #     batch_size=1, seed=0)

  # train_fn = functools.partial(
  #   sac.train, num_timesteps=1_000, num_evals=0, reward_scaling=1., episode_length=env._episode_length,
  #   normalize_observations=True, action_repeat=1, discounting=0.997, learning_rate=6e-4,
  #   num_envs=1, batch_size=1, grad_updates_per_step=1, max_devices_per_host=1,
  #   max_replay_size=10_000, min_replay_size=10, seed=1)

  train_fn = functools.partial(sac.train, num_timesteps=10_485, num_evals=20, reward_scaling=5, #  num_timesteps=7_864_320
                               episode_length=env._episode_length, normalize_observations=True, action_repeat=1,
                               discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=128,
                               grad_updates_per_step=32, max_devices_per_host=1, max_replay_size=10_485,
                               min_replay_size=10_000, seed=1)
  

  x_data = []
  y_data = []
  ydataerr = []
  times = [datetime.now()]

  max_y, min_y = 13000, 0
  def progress(num_steps, metrics):

    print(f"num steps: {num_steps}, eval/episode_reward: {metrics['eval/episode_reward']}")

    # times.append(datetime.now())
    # x_data.append(num_steps)
    # y_data.append(metrics['eval/episode_reward'])
    # ydataerr.append(metrics['eval/episode_reward_std'])

    # plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
    # plt.ylim([min_y, max_y])

    # plt.xlabel('# environment steps')
    # plt.ylabel('reward per episode')
    # plt.title(f'y={y_data[-1]:.3f}')

    # plt.errorbar(
    #     x_data, y_data, yerr=ydataerr)
    # plt.show()

  make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

  print(f'time to jit: {times[1] - times[0]}')
  print(f'time to train: {times[-1] - times[1]}')

  # cwd = os.path.dirname(os.path.abspath(__file__))

  # model.save_params(cwd + '/params', params)
  # params = model.load_params(cwd + '/params')
  # inference_fn = make_inference_fn(params)
  # jit_inference_fn = jax.jit(inference_fn)

  exit()

  backend = 'positional' # @param ['generalized', 'positional', 'spring']
  env = envs.create(env_name=env_name, backend=backend, episode_length=env._episode_length, init_qpos=jp.array(init_qpos))

  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(env.step)

  rollouts = []
  i = 0
  for episode in range(3):
    rng = jax.random.PRNGKey(seed=episode)
    state = jit_env_reset(rng=rng)
    rollout = {}
    states = []
    while not (state.done or state.info['truncation']):
      print(i)
      states.append(state.pipeline_state)
      act_rng, rng = jax.random.split(rng)
      # act, _ = jit_inference_fn(state.obs, act_rng)
      act = jp.zeros(env.action_size)
      state = jit_env_step(state, act)
      i += 1

    rollout['states'] = states
    rollouts.append(rollout)

  # _render(rollouts)

  breakpoint()

if __name__ == '__main__':
  main()