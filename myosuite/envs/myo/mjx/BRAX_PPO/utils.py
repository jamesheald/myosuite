import jax
from jax import numpy as jp
import mujoco

from functools import partial

def get_precoder(dynamics_apply, normalizer_params, params, observation, linearization_point, dropout_mask, subspace_basis_method):

  def modified_gram_schmidt(vectors):
    """
    adapted from https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/math/gram_schmidt.py
    fundamental change: while_loop replaced with scan
    Args:
    vectors: A Tensor of shape `[d, n]` of `d`-dim column vectors to
      orthonormalize.

    Returns:
    A Tensor of shape `[d, n]` corresponding to the orthonormalization.
    """
    def body_fn(vecs, i):
        u = jp.nan_to_num(vecs[:,i]/jp.linalg.norm(vecs[:,i]))
        weights = u@vecs
        masked_weights = jp.where(jp.arange(num_vectors) > i, weights, 0.)
        vecs = vecs - jp.outer(u,masked_weights)
        return vecs, None
    num_vectors = vectors.shape[-1]
    vectors, _ = jax.lax.scan(body_fn, vectors, jp.arange(num_vectors - 1))
    vec_norm = jp.linalg.norm(vectors, axis=0, keepdims=True)
    return jp.nan_to_num(vectors/vec_norm)

  def dynamics_forward_pass(linearization_point, normalizer_params, params, observation, dropout_mask):
    pred_delta_cv = dynamics_apply(normalizer_params, params, observation, linearization_point, dropout_mask)
    return pred_delta_cv

  get_jacobian = jax.jacrev(dynamics_forward_pass, has_aux=False)

  F = get_jacobian(linearization_point, normalizer_params, params, observation, dropout_mask)

  if subspace_basis_method == 'normalize':
    precoder = (F / (jp.linalg.norm(F, axis=-1, keepdims=True) + 1e-8)).T
  if subspace_basis_method == 'GS':
    precoder = modified_gram_schmidt(F.T)
  if subspace_basis_method == 'UVT':
    U, _, Vh = jp.linalg.svd(F, full_matrices=False)
    precoder = Vh.T@U.T
  
  return precoder

def bernoulli_mask(key, shape, keep_prob=0.9):
    """Generates a dropout-style Bernoulli mask scaled by 1/keep_prob."""
    key, subkey = jax.random.split(key)
    mask = (jax.random.uniform(subkey, shape) < keep_prob).astype(jp.float32)
    return mask / keep_prob

def get_controlled_variable_fn(self, controlled_variable):

  if controlled_variable == 'ObjCvel':

    num_controlled_variables = 6

    def get_controlled_variable(self, data):

      return data.cvel[self._object_bid].copy()
    
  if controlled_variable == 'ShoulderDoFObjCvel':

    joint_names = ['MyoArm_v0.01/elv_angle', 'MyoArm_v0.01/shoulder_elv', 'MyoArm_v0.01/shoulder_rot']
    joint_ids = jp.array([mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])
    humerus_bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, 'MyoArm_v0.01/humerus')

    num_controlled_variables =  len(joint_names) + 6

    def get_controlled_variable(self, data):

      obj_vel = data.cvel[self._object_bid].copy()
      v_obj_ang = obj_vel[:3] # angular velocity
      v_obj_lin = obj_vel[3:] # linear velocity
      obj_pos = data.xpos[self._object_bid].copy()

      humerus_vel = data.cvel[humerus_bid] # rot:lin velocity
      w_h = humerus_vel[:3] # humerus angular velocity
      v_h = humerus_vel[3:] # humerus linear velocity
      humerus_pos = data.xpos[humerus_bid]

      r = obj_pos - humerus_pos
      v_due_to_humerus = v_h + jp.cross(w_h, r)
      v_obj_resid = v_obj_lin - v_due_to_humerus
      w_obj_resid = v_obj_ang - w_h
      obj_twist = jp.concatenate((w_obj_resid, v_obj_resid))

      return jp.concatenate((data.qvel[joint_ids],
                              obj_twist),
                              axis=0).astype(jp.float32)
    
  if controlled_variable == 'ShoulderDoFObjCvelHaptic':

    joint_names = ['MyoArm_v0.01/elv_angle', 'MyoArm_v0.01/shoulder_elv', 'MyoArm_v0.01/shoulder_rot']
    joint_ids = jp.array([mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])
    humerus_bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, 'MyoArm_v0.01/humerus')

    num_controlled_variables = len(joint_names) + 6 + 1

    def get_controlled_variable(self, data):

      obj_vel = data.cvel[self._object_bid].copy()
      v_obj_ang = obj_vel[:3] # angular velocity
      v_obj_lin = obj_vel[3:] # linear velocity
      obj_pos = data.xpos[self._object_bid].copy()

      humerus_vel = data.cvel[humerus_bid] # rot:lin velocity
      w_h = humerus_vel[:3] # humerus angular velocity
      v_h = humerus_vel[3:] # humerus linear velocity
      humerus_pos = data.xpos[humerus_bid]

      r = obj_pos - humerus_pos
      v_due_to_humerus = v_h + jp.cross(w_h, r)
      v_obj_resid = v_obj_lin - v_due_to_humerus
      w_obj_resid = v_obj_ang - w_h
      obj_twist = jp.concatenate((w_obj_resid, v_obj_resid))

      return jp.concatenate((data.qvel[joint_ids],
                              (data.sensordata[:19]**2).mean(keepdims=True),
                              obj_twist),
                              axis=0).astype(jp.float32)
    
  if controlled_variable == 'ShoulderDoFObjCvelTips':

    joint_names = ['MyoArm_v0.01/elv_angle', 'MyoArm_v0.01/shoulder_elv', 'MyoArm_v0.01/shoulder_rot']
    joint_ids = jp.array([mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])
    humerus_bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, 'MyoArm_v0.01/humerus')

    body_names = ['MyoArm_v0.01/distal_thumb',
                'MyoArm_v0.01/distph2'
                'MyoArm_v0.01/distph3',
                'MyoArm_v0.01/distph4',
                'MyoArm_v0.01/distph5']
    body_ids = jp.array([mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names])

    for i in range(self._mj_model.nsite):
            name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name == "MyoArm_v0.01/S_grasp":
                palm_sid = i

    num_controlled_variables =  len(joint_names) + 6 + 15

    def get_controlled_variable(self, data):

      obj_vel = data.cvel[self._object_bid].copy()
      v_obj_ang = obj_vel[:3] # angular velocity
      v_obj_lin = obj_vel[3:] # linear velocity
      obj_pos = data.xpos[self._object_bid].copy()

      humerus_vel = data.cvel[humerus_bid] # rot:lin velocity
      w_h = humerus_vel[:3] # humerus angular velocity
      v_h = humerus_vel[3:] # humerus linear velocity
      humerus_pos = data.xpos[humerus_bid]

      r = obj_pos - humerus_pos
      v_due_to_humerus = v_h + jp.cross(w_h, r)
      v_obj_resid = v_obj_lin - v_due_to_humerus
      w_obj_resid = v_obj_ang - w_h
      obj_twist = jp.concatenate((w_obj_resid, v_obj_resid))

      palm_pos = data.site_xpos[palm_sid]
      palm_rot = jp.reshape(data.site_xmat[palm_sid], (3, 3))

      tip0 = palm_rot.T @ (data.cvel[body_ids[0]][3:]-palm_pos)
      tip1 = palm_rot.T @ (data.cvel[body_ids[1]][3:]-palm_pos)
      tip2 = palm_rot.T @ (data.cvel[body_ids[2]][3:]-palm_pos)
      tip3 = palm_rot.T @ (data.cvel[body_ids[3]][3:]-palm_pos)
      tip4 = palm_rot.T @ (data.cvel[body_ids[4]][3:]-palm_pos)

      digit_tips = jp.concatenate((tip0,
                                    tip1,
                                    tip2,
                                    tip3,
                                    tip4), axis=0)

      return jp.concatenate((data.qvel[joint_ids],
                              digit_tips,
                              obj_twist),
                              axis=0).astype(jp.float32)
    
  if controlled_variable == 'DoF7Tips':

    joint_names = ['MyoArm_v0.01/elv_angle',
                   'MyoArm_v0.01/shoulder_elv',
                   'MyoArm_v0.01/shoulder_rot'
                   'MyoArm_v0.01/elbow_flexion'
                   'MyoArm_v0.01/pro_sup'
                   'MyoArm_v0.01/deviation'
                   'MyoArm_v0.01/flexion']
    joint_ids = jp.array([mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])
    humerus_bid = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, 'MyoArm_v0.01/humerus')

    body_names = ['MyoArm_v0.01/distal_thumb',
                'MyoArm_v0.01/distph2'
                'MyoArm_v0.01/distph3',
                'MyoArm_v0.01/distph4',
                'MyoArm_v0.01/distph5']
    body_ids = jp.array([mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names])

    for i in range(self._mj_model.nsite):
            name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name == "MyoArm_v0.01/S_grasp":
                palm_sid = i

    num_controlled_variables =  len(joint_names) + 15

    def get_controlled_variable(self, data):

      obj_vel = data.cvel[self._object_bid].copy()
      v_obj_ang = obj_vel[:3] # angular velocity
      v_obj_lin = obj_vel[3:] # linear velocity
      obj_pos = data.xpos[self._object_bid].copy()

      humerus_vel = data.cvel[humerus_bid] # rot:lin velocity
      w_h = humerus_vel[:3] # humerus angular velocity
      v_h = humerus_vel[3:] # humerus linear velocity
      humerus_pos = data.xpos[humerus_bid]

      r = obj_pos - humerus_pos
      v_due_to_humerus = v_h + jp.cross(w_h, r)
      v_obj_resid = v_obj_lin - v_due_to_humerus
      w_obj_resid = v_obj_ang - w_h
      obj_twist = jp.concatenate((w_obj_resid, v_obj_resid))

      palm_pos = data.site_xpos[palm_sid]
      palm_rot = jp.reshape(data.site_xmat[palm_sid], (3, 3))

      tip0 = palm_rot.T @ (data.cvel[body_ids[0]][3:]-palm_pos)
      tip1 = palm_rot.T @ (data.cvel[body_ids[1]][3:]-palm_pos)
      tip2 = palm_rot.T @ (data.cvel[body_ids[2]][3:]-palm_pos)
      tip3 = palm_rot.T @ (data.cvel[body_ids[3]][3:]-palm_pos)
      tip4 = palm_rot.T @ (data.cvel[body_ids[4]][3:]-palm_pos)

      digit_tips = jp.concatenate((tip0,
                                    tip1,
                                    tip2,
                                    tip3,
                                    tip4), axis=0)

      return jp.concatenate((data.qvel[joint_ids],
                              digit_tips),
                              axis=0).astype(jp.float32)
    
  if controlled_variable == 'ArmQvel':

    joint_names = ['MyoArm_v0.01/elv_angle',
                    'MyoArm_v0.01/shoulder_elv',
                    'MyoArm_v0.01/shoulder_rot',
                    'MyoArm_v0.01/elbow_flexion',
                    'MyoArm_v0.01/pro_sup',
                    'MyoArm_v0.01/deviation',
                    'MyoArm_v0.01/flexion',
                    'MyoArm_v0.01/cmc_abduction',
                    'MyoArm_v0.01/cmc_flexion',
                    'MyoArm_v0.01/mp_flexion',
                    'MyoArm_v0.01/ip_flexion',
                    'MyoArm_v0.01/mcp2_flexion',
                    'MyoArm_v0.01/mcp2_abduction',
                    'MyoArm_v0.01/pm2_flexion',
                    'MyoArm_v0.01/md2_flexion',
                    'MyoArm_v0.01/mcp3_flexion',
                    'MyoArm_v0.01/mcp3_abduction',
                    'MyoArm_v0.01/pm3_flexion',
                    'MyoArm_v0.01/md3_flexion',
                    'MyoArm_v0.01/mcp4_flexion',
                    'MyoArm_v0.01/mcp4_abduction',
                    'MyoArm_v0.01/pm4_flexion',
                    'MyoArm_v0.01/md4_flexion',
                    'MyoArm_v0.01/mcp5_flexion',
                    'MyoArm_v0.01/mcp5_abduction',
                    'MyoArm_v0.01/pm5_flexion',
                    'MyoArm_v0.01/md5_flexion']
    joint_ids = jp.array([mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])

    num_controlled_variables = len(joint_names)

    def get_controlled_variable(self, data):

      return data.qvel[joint_ids]
    
  if controlled_variable == 'WristTips':

    model = 'FSWH/'

    joint_names = [model + 'pro_sup',
                   model + 'deviation',
                   model + 'flexion']
    joint_ids = jp.array([mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])

    body_names = [model + 'distal_thumb',
                  model + 'distph2',
                  model + 'distph3',
                  model + 'distph4',
                  model + 'distph5']
    body_ids = jp.array([mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names])

    for i in range(self._mj_model.nsite):
            name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name == model + "S_grasp":
                palm_sid = i

    num_controlled_variables = len(joint_names) + 15

    def make_controlled_variable_fn(palm_sid, body_ids, joint_ids):
      def controlled_variable(self, data):
          palm_pos = data.site_xpos[palm_sid]
          palm_rot = jp.reshape(data.site_xmat[palm_sid], (3, 3))

          tips = []
          for bid in body_ids:
              tip = palm_rot.T @ (data.xpos[bid] - palm_pos)
              # tip = palm_rot.T @ (data.cvel[bid][3:] - palm_pos)
              tips.append(tip)

          digit_tips = jp.concatenate(tips, axis=0)

          return jp.concatenate(
              (data.qvel[joint_ids], digit_tips), axis=0
          ).astype(jp.float32)

      return controlled_variable
    
    get_controlled_variable = make_controlled_variable_fn(palm_sid, body_ids, joint_ids)
    
  return get_controlled_variable, num_controlled_variables