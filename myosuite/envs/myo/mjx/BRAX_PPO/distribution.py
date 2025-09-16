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

"""Probability distributions in JAX."""

import abc
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


class ParametricDistribution(abc.ABC):
  """Abstract class for parametric (action) distribution."""

  def __init__(self, param_size, postprocessor, event_ndims, reparametrizable):
    """Abstract class for parametric (action) distribution.

    Specifies how to transform distribution parameters (i.e. actor output)
    into a distribution over actions.

    Args:
      param_size: size of the parameters for the distribution
      postprocessor: bijector which is applied after sampling (in practice, it's
        tanh or identity)
      event_ndims: rank of the distribution sample (i.e. action)
      reparametrizable: is the distribution reparametrizable
    """
    self._param_size = param_size
    self._postprocessor = postprocessor
    self._event_ndims = event_ndims  # rank of events
    self._reparametrizable = reparametrizable
    assert event_ndims in [0, 1]

  @abc.abstractmethod
  def create_dist(self, parameters):
    """Creates distribution from parameters."""
    pass

  @property
  def param_size(self):
    return self._param_size

  @property
  def reparametrizable(self):
    return self._reparametrizable

  def postprocess(self, event, precoder_state, observations):
    return self._postprocessor.forward(event, precoder_state, observations)

  def inverse_postprocess(self, event):
    return self._postprocessor.inverse(event)

  def sample_no_postprocessing(self, parameters, seed):
    return self.create_dist(parameters).sample(seed=seed)

  def sample(self, parameters, seed):
    """Returns a sample from the postprocessed distribution."""
    return self.postprocess(self.sample_no_postprocessing(parameters, seed))
  
  def sample_log_prob(self, parameters, seed, precoder_state, observations):
     dist = self.create_dist(parameters)
     raw_actions = dist.sample(seed=seed)
     log_probs = dist.log_prob(raw_actions).sum(axis=-1)
     postprocessed_actions, fldj = self._postprocessor.forward_log_det_jacobian(raw_actions, precoder_state, observations)
     log_probs -= fldj
     return raw_actions, postprocessed_actions, log_probs
  
  def log_prob_entropy(self, parameters, raw_actions, seed, precoder_state, observations):
     dist = self.create_dist(parameters)
     log_probs = dist.log_prob(raw_actions).sum(axis=-1)
     _, fldj = self._postprocessor.forward_log_det_jacobian2(raw_actions, precoder_state, observations)
     log_probs -= fldj
     entropy = dist.entropy().sum(axis=-1)
     _, fldj = self._postprocessor.forward_log_det_jacobian2(
        dist.sample(seed=seed), precoder_state, observations
        )
     entropy += fldj
     return log_probs, entropy

  def mode(self, parameters, precoder_state, observations):
    """Returns the mode of the postprocessed distribution."""
    return self.postprocess(self.create_dist(parameters).mode(), precoder_state, observations)
  
  def stds(self, parameters):
    return self.get_stds(*parameters)
  
  def pi(self, parameters):
    return self.get_pi(*parameters)

  def log_prob(self, parameters, actions, composed_bijector):
    """Compute the log probability of actions."""
    dist = self.create_dist(parameters)
    log_probs = dist.log_prob(actions)
    log_probs -= jnp.sum(self._postprocessor.forward_log_det_jacobian(actions, composed_bijector), axis=-1)
    # if isinstance(self._postprocessor, TanhBijector):
    #   if isinstance(dist, _NormalDiagPlusLowRankDistribution):
    #     log_probs -= jnp.sum(self._postprocessor.forward_log_det_jacobian(actions), axis=-1)
    #   else:
    #     log_probs -= self._postprocessor.forward_log_det_jacobian(actions)
    # if isinstance(dist, _NormalDistribution) or isinstance(dist, _LatentNormalDistribution):
    #   log_probs = jnp.sum(log_probs, axis=-1)  # sum over action dimension
    return log_probs

  def entropy(self, parameters, seed):
    """Return the entropy of the given distribution."""
    dist = self.create_dist(parameters)
    entropy = dist.entropy()
    if isinstance(self._postprocessor, TanhBijector):
        entropy += self._postprocessor.forward_log_det_jacobian(
            dist.sample(seed=seed)
        )
    if isinstance(dist, _NormalDistribution) or isinstance(dist, _LatentNormalDistribution):
      entropy = jnp.sum(entropy, axis=-1)
    return entropy


class _NormalDistribution:
  """Normal distribution."""

  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

  def sample(self, seed):
    return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

  def mode(self):
    return self.loc

  def log_prob(self, x):
    log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
    log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
    return log_unnormalized - log_normalization

  def entropy(self):
    log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
    entropy = 0.5 + log_normalization
    return entropy * jnp.ones_like(self.loc)


class TanhBijector:
  """Tanh Bijector."""

  def forward(self, x):
    return jnp.tanh(x)

  def inverse(self, y):
    return jnp.arctanh(y)

  def forward_log_det_jacobian(self, x):
    return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))
  
class SigmoidBijector:
  """Shifted-and-scaled Sigmoid Bijector."""

  def __init__(self, slope=5.0, center=0.5, eps=1e-7):
      self.slope = slope
      self.center = center
      self.eps = eps  # for numerical stability in log terms

  def forward(self, x):
      # σ(slope * (x - center))
      return jax.nn.sigmoid(self.slope * (x - self.center))

  def inverse(self, y):
      # logit(y)/slope + center
      y = jnp.clip(y, self.eps, 1.0 - self.eps)  # avoid log(0)
      return self.center + (1.0 / self.slope) * jnp.log(y / (1.0 - y))

  def forward_log_det_jacobian(self, x):
      # log(slope) + log(sigmoid(s)) + log(1 - sigmoid(s)), with s = slope*(x-center)
      s = self.slope * (x - self.center)
      # use softplus for stability: log(sigmoid(s)) = -softplus(-s), log(1-sigmoid(s)) = -softplus(s)
      return jnp.log(self.slope) - jax.nn.softplus(-s) - jax.nn.softplus(s)

# # ----- Tests -----
# bij = SigmoidBijector()

# # Check inverse(forward(x)) ≈ x
# x = jnp.linspace(-2, 2, 10)
# y = bij.forward(x)
# x_recon = bij.inverse(y)
# print("recon error:", abs(x - x_recon).mean())

# grad_fn = jax.vmap(jax.grad(lambda xi: bij.forward(xi)))
# jacobian = grad_fn(x)  # shape (10,)
# logdet_from_grad = jnp.log(jnp.abs(jacobian))

# manual_logdet = jax.vmap(bij.forward_log_det_jacobian)(x)

# print("grad error:", abs(logdet_from_grad - manual_logdet).mean())

class PrecoderBijector:
    def __init__(self, d, D):
        self.d = d  # Dimension of input z
        self.D = D  # Dimension of output a

    def forward(self, z, precoder_state, observations):
        # composed_bijector = precoder_state.apply_fn(precoder_state.params, observations)
        # # x: z with shape (batch1, batch2, ..., d)
        # # Returns: a = f([x, 0, ..., 0]) with shape (batch1, batch2, ..., D)
        # x_pad = jnp.concatenate((x, jnp.zeros(x.shape[:-1] + (self.D - self.d,))), axis=-1)
        # return composed_bijector.forward(x_pad)
        a = precoder_state.apply_fn(precoder_state.params, z, observations)
        return a

    def inverse(self, y):
        # y: a with shape (batch1, batch2, ..., D)
        # Returns: z with shape (batch1, batch2, ..., d) or NaN for invalid samples
        z_pad = self.bijector.inverse(y)
        # Check if last D-d components are zero for each sample
        # is_valid = jnp.all(jnp.abs(z_pad[..., self.d:]) < 1e-6, axis=-1)  # Shape: (batch1, batch2, ...)
        z = z_pad[..., :self.d]  # Shape: (batch1, batch2, ..., d)
        # Where is_valid is False, set output to NaN
        # return jnp.where(is_valid[..., None], z, jnp.full_like(z, jnp.nan))
        return z
    
    def forward_log_det_jacobian2(self, x, precoder_state, observations):
        # x: z with shape (batch1, batch2, ..., d)
        # Returns: 0.5 * log det(J^T J) for each sample, shape (batch1, batch2, ...)
        def forward_fn(z, observations):
            # composed_bijector = precoder_state.apply_fn(precoder_state.params, observations)
            # z_pad = jnp.concatenate((z, jnp.zeros(z.shape[:-1] + (self.D - self.d,))), axis=-1)
            # a = composed_bijector.forward(z_pad)
            a = precoder_state.apply_fn(precoder_state.params, z, observations)
            return a, a
        
        def single_jvp(z):
            # Compute forward transformation
            # y = forward_fn(z, obs=obs)  # Shape: (batch1, batch2, ..., D)
            # Compute D x d Jacobian (df/dz) using jacfwd
            # y = forward_fn(z)
            Jac, y = jax.vmap(jax.vmap(jax.jacrev(forward_fn, has_aux=True)))(z, observations)  # Shape: (batch1, batch2, ..., D, d)
            gram = jnp.einsum('...jk,...jl->...kl', Jac, Jac)
            # print("y shape", z.shape)
            # print("z shape", z.shape)
            # Compute gram matrix: J^T J
            # gram = jnp.matmul(J.transpose(*range(J.ndim-2), J.ndim-1, J.ndim-2), J)
            # gram: (batch1, batch2, ..., d, d), e.g., (128, 23, 23)
            # Compute logdet
            _, logdet_gram = jax.vmap(jax.vmap(jnp.linalg.slogdet))(gram)
            # logdet_gram = logdet_gram[1]  # Extract logdet
            return y, 0.5 * logdet_gram
        
        y, fldj = single_jvp(x)  # Shape: (batch1, batch2, ...)
        return y, fldj

    def forward_log_det_jacobian(self, x, precoder_state, observations):
        # x: z with shape (batch1, batch2, ..., d)
        # Returns: 0.5 * log det(J^T J) for each sample, shape (batch1, batch2, ...)
        def forward_fn(z, observations):
            # composed_bijector = precoder_state.apply_fn(precoder_state.params, observations)
            # z_pad = jnp.concatenate((z, jnp.zeros(z.shape[:-1] + (self.D - self.d,))), axis=-1)
            # a = composed_bijector.forward(z_pad)
            a = precoder_state.apply_fn(precoder_state.params, z, observations)
            return a, a
        
        def single_jvp(z):
            # Compute forward transformation
            # y = forward_fn(z, obs=obs)  # Shape: (batch1, batch2, ..., D)
            # Compute D x d Jacobian (df/dz) using jacfwd
            # y = forward_fn(z)
            Jac, y = jax.vmap(jax.jacrev(forward_fn, has_aux=True))(z, observations)  # Shape: (batch1, batch2, ..., D, d)
            gram = jnp.einsum('...jk,...jl->...kl', Jac, Jac)
            # print("y shape", z.shape)
            # print("z shape", z.shape)
            # Compute gram matrix: J^T J
            # gram = jnp.matmul(J.transpose(*range(J.ndim-2), J.ndim-1, J.ndim-2), J)
            # gram: (batch1, batch2, ..., d, d), e.g., (128, 23, 23)
            # Compute logdet
            _, logdet_gram = jax.vmap(jnp.linalg.slogdet)(gram)
            # logdet_gram = logdet_gram[1]  # Extract logdet
            return y, 0.5 * logdet_gram

        
        # def single_jvp(z):
        #     # Create basis with batch dimensions
        #     basis = jnp.eye(self.d)[None]  # Shape: (1, d, d)
        #     batch_shape = z.shape[:-1]  # e.g., (128,) or ()
        #     for i, size in enumerate(batch_shape):
        #         basis = basis.repeat(size, axis=0)  # Shape: (batch1, batch2, ..., d, d)

        #     def jvp_per_basis(z, b):
        #         y, jvp_b = jax.jvp(lambda z: forward_fn(z), (z,), (b,))  # jvp_b: (D,)
        #         return y, jvp_b

        #     # Vectorize over basis columns
        #     y, J_cols = jax.vmap(jvp_per_basis, in_axes=(None, -1), out_axes=(None, 0))(z, basis)
        #     # J_cols: (batch1, batch2, ..., D, d), e.g., (128, 39, 23)
        #     # Compute gram matrix: J^T J for each sample
        #     gram = jnp.matmul(J_cols.transpose(*range(J_cols.ndim-2), J_cols.ndim-1, J_cols.ndim-2), J_cols)
        #     # gram: (batch1, batch2, ..., d, d), e.g., (128, 23, 23)
        #     # Compute logdet
        #     if gram.ndim > 2:
        #         sign, logdet_gram = jax.vmap(jnp.linalg.slogdet, in_axes=0)(gram)
        #         logdet_gram = logdet_gram[1]  # Extract logdet
        #     else:
        #         sign, logdet_gram = jnp.linalg.slogdet(gram)
        #     return y, 0.5 * logdet_gram

        # Vectorize over all batch dimensions
        # num_batch_dims = x.ndim - 1  # Number of batch dimensions
        # vmap_fn = single_jvp
        # for _ in range(num_batch_dims):
        #     vmap_fn = jax.vmap(vmap_fn, in_axes=0, out_axes=0)
        # y, fldj = vmap_fn(x)  # Shape: (batch1, batch2, ...)
        y, fldj = single_jvp(x)  # Shape: (batch1, batch2, ...)
        return y, fldj

class NormalPrecoderSigmoidDistribution(ParametricDistribution):
  """Normal distribution followed by tanh."""

  def __init__(self, event_size, D, min_std=0.001, var_scale=1):
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
      min_std: minimum std for the gaussian.
      var_scale: adjust the gaussian's scale parameter.
    """
    # We apply tanh to gaussian actions to bound them.
    # Normally we would use TransformedDistribution to automatically
    # apply tanh to the distribution.
    # We can't do it here because of tanh saturation
    # which would make log_prob computations impossible. Instead, most
    # of the code operate on pre-tanh actions and we take the postprocessor
    # jacobian into account in log_prob computations.
    super().__init__(
        param_size=2 * event_size,
        postprocessor=PrecoderBijector(event_size, D),
        event_ndims=1,
        reparametrizable=True,
    )
    self._min_std = min_std
    self._var_scale = var_scale

  def create_dist(self, parameters):
    loc, scale = jnp.split(parameters, 2, axis=-1)
    scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
    return _NormalDistribution(loc=loc, scale=scale)

class NormalTanhDistribution(ParametricDistribution):
  """Normal distribution followed by tanh."""

  def __init__(self, event_size, min_std=0.001, var_scale=1):
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
      min_std: minimum std for the gaussian.
      var_scale: adjust the gaussian's scale parameter.
    """
    # We apply tanh to gaussian actions to bound them.
    # Normally we would use TransformedDistribution to automatically
    # apply tanh to the distribution.
    # We can't do it here because of tanh saturation
    # which would make log_prob computations impossible. Instead, most
    # of the code operate on pre-tanh actions and we take the postprocessor
    # jacobian into account in log_prob computations.
    super().__init__(
        param_size=2 * event_size,
        postprocessor=TanhBijector(),
        event_ndims=1,
        reparametrizable=True,
    )
    self._min_std = min_std
    self._var_scale = var_scale

  def create_dist(self, parameters):
    loc, scale = jnp.split(parameters, 2, axis=-1)
    scale = (jax.nn.softplus(scale) + self._min_std) * self._var_scale
    return _NormalDistribution(loc=loc, scale=scale)


class IdentityPostprocessor:
  """Identity postprocessor."""

  def forward(self, x):
    return x

  def inverse(self, x):
    return x

  def forward_log_det_jacobian(self, x):
    return jnp.zeros_like(x)


class NormalDistribution(ParametricDistribution):
  """Normal distribution."""

  def __init__(self, event_size: int) -> None:
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
    """
    super().__init__(
        param_size=event_size,
        postprocessor=IdentityPostprocessor(),
        event_ndims=1,
        reparametrizable=True,
    )

  def create_dist(self, parameters):
    return _NormalDistribution(*parameters)
  
class NormalDiagPlusLowRankDistribution(ParametricDistribution):
  """Multivariate normal distribution with diagonal plus low rank covariance."""

  def __init__(self, event_size: int, postprocessor: str) -> None:
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
    """
    postprocessor_map = {
        "identity": IdentityPostprocessor(),
        "sigmoid": SigmoidBijector(),
        "tanh": TanhBijector(),
    }
    super().__init__(
        param_size=event_size,
        postprocessor=postprocessor_map[postprocessor],
        event_ndims=1,
        reparametrizable=True,
    )

  def create_dist(self, parameters):
    return _NormalDiagPlusLowRankDistribution(*parameters)
  
  def get_stds(self, loc, scale_diag_factor, scale_perturb_diag, scale_perturb_factor):
    return scale_diag_factor, scale_perturb_diag
  
class _NormalDiagPlusLowRankDistribution:
    """Diagonal + Low-Rank multivariate normal distribution.

    Covariance = diag(cov_diag_factor) + cov_perturb_factor @ cov_perturb_factor.T
    """

    def __init__(self, loc, scale_diag_factor, scale_perturb_diag, scale_perturb_factor):
        self.loc = loc
        self._dist = tfd.MultivariateNormalDiagPlusLowRankCovariance(
            loc=loc,
            cov_diag_factor=scale_diag_factor**2,
            cov_perturb_factor=jnp.einsum('...ij,...j->...ij', scale_perturb_factor, scale_perturb_diag)
        )

        self.std = scale_diag_factor.mean()
        self.latent_std = scale_perturb_diag.mean()

    def sample(self, seed):
        return self._dist.sample(seed=seed)

    def mode(self):
        # For a Gaussian, the mode is the mean
        return self.loc

    def log_prob(self, x):
        return self._dist.log_prob(x)

    def entropy(self):
        return self._dist.entropy()
    
class LatentNormalDistribution(ParametricDistribution):
  """Multivariate normal distribution with diagonal plus low rank covariance."""

  def __init__(self, event_size: int) -> None:
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
    """
    super().__init__(
        param_size=event_size,
        postprocessor=IdentityPostprocessor(),
        event_ndims=1,
        reparametrizable=True,
    )

  def create_dist(self, parameters):
    return _LatentNormalDistribution(*parameters)
  
class _LatentNormalDistribution:
    """p(a): a = Az + b, where z ~ N(loc, diag(scale))
    """
    def __init__(self, loc, scale, precoder, linearization_point):

      self.loc = loc
      self.scale = scale
      self.precoder = precoder
      self.linearization_point = linearization_point

      self.std = scale.mean()
      self.latent_std = scale.mean()

    def precode(self, z):
      a_no_bias = jnp.einsum('...mn,...n->...m', self.precoder, z)
      a = a_no_bias + jnp.broadcast_to(self.linearization_point, a_no_bias.shape)
      return a

    def sample(self, seed):
      z = jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc
      a = self.precode(z)
      return a, z

    def mode(self):
      z = self.loc
      a = self.precode(z)
      return a

    def log_prob(self, z):
      # def solve_one(P, a):
      #   Q, R = jnp.linalg.qr(P)
      #   z = jnp.linalg.solve(R, Q.T @ a)
      #   return z
      # def one_vmap(precoder, a):
      #     return jax.vmap(solve_one)(precoder, a)
      # def two_vmap(precoder, a):
      #     return jax.vmap(jax.vmap(solve_one))(precoder, a)

      # print(self.precoder.shape, a.shape)
      # breakpoint()
      # z = jax.lax.cond(a.ndim == 2, one_vmap, two_vmap, self.precoder, a)
      log_unnormalized = -0.5 * jnp.square(z / self.scale - self.loc / self.scale)
      log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
      return log_unnormalized - log_normalization

    def entropy(self):
      log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
      entropy = 0.5 + log_normalization
      return entropy * jnp.ones_like(self.loc)
    
class MultivariateNormalMixtureFullCovariance(ParametricDistribution):
  """Multivariate normal mixture distribution with full covariance."""

  def __init__(self, event_size: int) -> None:
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
    """
    super().__init__(
        param_size=event_size,
        postprocessor=IdentityPostprocessor(),
        event_ndims=1,
        reparametrizable=True,
    )

  def create_dist(self, parameters):
    return _MultivariateNormalMixtureFullCovariance(*parameters)
  
  def get_stds(self, loc, latent_loc, scale_diag_factor, latent_scale_diag_factor, pi, precoder, linearization_point):
    return scale_diag_factor, latent_scale_diag_factor
    # return jnp.array([scale_diag_factor.mean()]), jnp.array([latent_scale_diag_factor.mean()])
  
  def get_pi(self, loc, latent_loc, scale_diag_factor, latent_scale_diag_factor, pi, precoder, linearization_point):
    return pi
  
class _MultivariateNormalMixtureFullCovariance:
    """Mixture of Gaussians
    """

    def __init__(self, loc, latent_loc, scale_diag_factor, latent_scale_diag_factor, pi, precoder, linearization_point):
        
        # cov = jnp.diag(scale_diag_factor)
        # batch_size = loc.shape[0]
        # cov = jnp.broadcast_to(cov, (batch_size, cov.shape[0], cov.shape[1]))

        D = scale_diag_factor.shape[-1]
        # Create diagonal (..., D, D)
        diag = jnp.einsum('...d,ij->...ij', scale_diag_factor, jnp.eye(D, dtype=scale_diag_factor.dtype))
        # Broadcast to match loc batch axes (...batch..., D, D)
        target_shape = loc.shape[:-1] + (D, D)
        cov = jnp.broadcast_to(diag, target_shape)
        
        latent_loc_no_bias = jnp.einsum('...mn,...n->...m', precoder, latent_loc)
        self.latent_loc = latent_loc_no_bias + jnp.broadcast_to(linearization_point, latent_loc_no_bias.shape)

        A = jnp.einsum('...ij,...j->...ij', precoder, latent_scale_diag_factor)
        latent_cov = jnp.einsum('...ik,...jk->...ij', A, A)

        self.pi = jnp.squeeze(pi)
        self.mixture_loc = self.pi * loc + (1. - self.pi) * self.latent_loc
        delta1 = loc - self.mixture_loc
        delta2 = self.latent_loc - self.mixture_loc
        self.mixture_cov = (self.pi * cov + (1. - self.pi) * latent_cov
                            + self.pi * jnp.einsum('...i,...j->...ij', delta1, delta1)
                            + (1. - self.pi) * jnp.einsum('...i,...j->...ij', delta2, delta2)
        )

        self._dist = tfd.MultivariateNormalFullCovariance(
            loc=self.mixture_loc,
            covariance_matrix=self.mixture_cov,
        )

    def sample(self, seed):
        return self._dist.sample(seed=seed)

    def mode(self):
        # For a Gaussian, the mode is the mean
        return self.mixture_loc

    def log_prob(self, x):
        return self._dist.log_prob(x)

    def entropy(self):
        return self._dist.entropy()