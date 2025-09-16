# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO networks."""

from typing import Literal, Sequence, Tuple

# from brax.training import distribution
# from brax.training import networks
from myosuite.envs.myo.mjx.BRAX_PPO import distribution
from myosuite.envs.myo.mjx.BRAX_PPO import networks as networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax
from jax import numpy as jnp


from functools import partial

@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  # dynamics_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution
  # keep_prob: float
  # subspace_basis_method: str
  # linearization_point: float


def make_inference_fn(ppo_networks: PPONetworks, precoder_state):
  """Creates params and inference function for the PPO agent."""

  def make_policy(
      params: types.Params, deterministic: bool = False
  ) -> types.Policy:
    policy_network = ppo_networks.policy_network
    # dynamics_network = ppo_networks.dynamics_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    def policy(
        observations: types.Observation, key_sample: PRNGKey,
    ) -> Tuple[types.Action, types.Extra]:
      param_subset = (params[0], params[1])  # normalizer and policy params
      logits = policy_network.apply(*param_subset, observations)

      # composed_bijector = precoder_state.apply_fn(precoder_state.params, observations['precoder'])

      if deterministic:
        return ppo_networks.parametric_action_distribution.mode(logits, precoder_state, observations['precoder']), {}
      # raw_actions = parametric_action_distribution.sample_no_postprocessing(
          # logits, key_sample
      # )
      # log_prob = parametric_action_distribution.log_prob(logits, raw_actions, composed_bijector)

      raw_actions, postprocessed_actions, log_prob = parametric_action_distribution.sample_log_prob(logits, key_sample, precoder_state, observations['precoder'])
      
      # postprocessed_actions = parametric_action_distribution.postprocess(
      #     raw_actions, composed_bijector
      # )

      # postprocessed_actions = precoder_state.apply_fn(precoder_state.params, raw_actions, observations['precoder'])
      
      extras = {'log_prob': log_prob,
                'raw_action': raw_actions,
                }
      return postprocessed_actions, extras

    return policy

  return make_policy

def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    # num_controlled_variables: int = 0,
    # dynamics_hidden_layer_sizes: Sequence[int] = (256,) * 2,
    # dynamics_obs_key: str = 'state',
    # keep_prob: float = 1.0,
    # subspace_basis_method: str = 'normalize', # normalize, GS, UVT
    # linearization_point: float = 0.0,
    # dim_dependent_latent_std: bool = False,
    # postprocessor: str = 'identity', # identity or tanh
    # warm_start_bias: float = 0.0,
    # pi_init: float = 0.5,
) -> PPONetworks:
  """Make PPO networks with preprocessor."""
  parametric_action_distribution: distribution.ParametricDistribution
  if distribution_type == 'normal':
    parametric_action_distribution = distribution.NormalDistribution(
        event_size=action_size
    )
  elif distribution_type == 'tanh_normal':
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
  elif distribution_type == 'sigmoid_normal':
    parametric_action_distribution = distribution.NormalSigmoidDistribution(
        event_size=action_size
    )
  elif distribution_type == 'precoder_normal':
    parametric_action_distribution = distribution.NormalPrecoderSigmoidDistribution(
        event_size=action_size,
        D=39,
    )
  elif distribution_type == 'diag_plus_low_rank':
    parametric_action_distribution = distribution.NormalDiagPlusLowRankDistribution(
        event_size=action_size,
        postprocessor=postprocessor
    )
  elif distribution_type == 'latent_normal':
    parametric_action_distribution = distribution.LatentNormalDistribution(
        event_size=action_size
    )
  elif distribution_type == 'mixture':
    parametric_action_distribution = distribution.MultivariateNormalMixtureFullCovariance(
        event_size=action_size
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
      distribution_type=distribution_type,
      noise_std_type=noise_std_type,
      init_noise_std=init_noise_std,
      state_dependent_std=state_dependent_std,
      # num_controlled_variables=num_controlled_variables,
      # dim_dependent_latent_std=dim_dependent_latent_std,
      # warm_start_bias=warm_start_bias,
      # pi_init=pi_init,
  )
  value_network = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
  )

  # dynamics_network = networks.make_dynamics_network(
  #     num_controlled_variables=num_controlled_variables,
  #     obs_size=observation_size,
  #     action_size=action_size,
  #     preprocess_observations_fn=preprocess_observations_fn,
  #     hidden_layer_sizes=dynamics_hidden_layer_sizes,
  #     obs_key=dynamics_obs_key,
  # )
  
  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
      # dynamics_network=dynamics_network,
      # keep_prob=keep_prob,
      # subspace_basis_method=subspace_basis_method,
      # linearization_point=linearization_point,
  )