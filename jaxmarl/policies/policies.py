"""
Agent policy architectures and their dependencies
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )

    
class AgentMLP(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, unused, x):
        obs, dones = x

        embedding = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs))
        embedding = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding))
        q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)

        # hidden, q_vals is the original return for AgentRNN
        # this is just to keep the training loop code consistent
        return unused, q_vals 

class AgentHyperMLP(nn.Module):
    # TODO: if using this for experiments, add the hypernet_kwargs/hyper_forward from AgentHyperRNN
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float
    dim_capabilities: int # per team
    hypernet_kwargs: dict

    @nn.compact
    def __call__(self, unused, x):
        orig_obs, dones = x

        # separate obs into capabilities and observations
        # (env gives obs = orig obs+cap)
        # NOTE: this is hardcoded to match simple_spread's computation
        cap = orig_obs[:, :, -self.dim_capabilities:] # last dim_cap elements in obs are cap
        obs = orig_obs[:, :, :-self.dim_capabilities]

        time_steps, batch_size, obs_dim = obs.shape

        # hypernetwork
        w_1 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.hidden_dim*self.action_dim, init_scale=self.hypernet_init_scale)(orig_obs)
        b_1 = nn.Dense(self.action_dim, kernel_init=orthogonal(self.hypernet_init_scale), bias_init=constant(0.))(orig_obs)
        
        # reshaping
        w_1 = w_1.reshape(time_steps, batch_size, self.hidden_dim, self.action_dim)
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.action_dim)
    
        # two-layer encoder
        embedding = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs))
        embedding = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding))

        # target network after encoder
        q_vals = jnp.matmul(embedding[:, :, None, :], w_1) + b_1

        # hidden, q_vals is the original return for AgentRNN
        # this keeps the train loop consistent
        return unused, q_vals 


class AgentRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        # NOTE: SimpleSpread gives obs as obs+cap (concatenated) and zeroes out
        # the capabilities if capability_aware=False in config. Thus, no change
        # is needed here for capability aware/unaware.
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)

        return hidden, q_vals


class AgentHyperRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float
    dim_capabilities: int # per team
    hypernet_kwargs: dict

    def hyper_forward(self, in_dim, out_dim, target_in, hyper_in, time_steps, batch_size):
        """
        Compute y = xW + b where W/b are created by a hypernetwork.

        in_dim : dimension of target input x
        out_dim : dimension of target output y
        target_in : target input
        hyper_in : input to hypernet

        time_steps/batch_size : parallel dims
        """
        num_weights = (in_dim * out_dim)
        weight_hypernet = HyperNetwork(hidden_dim=self.hypernet_kwargs["HIDDEN_DIM"], output_dim=num_weights, init_scale=self.hypernet_kwargs["INIT_SCALE"], num_layers=self.hypernet_kwargs["NUM_LAYERS"], use_layer_norm=self.hypernet_kwargs["USE_LAYER_NORM"])
        weights = weight_hypernet(hyper_in).reshape(time_steps, batch_size, in_dim, out_dim)

        num_biases = out_dim
        bias_hypernet = HyperNetwork(hidden_dim=self.hypernet_kwargs["HIDDEN_DIM"], output_dim=num_biases, init_scale=0, num_layers=self.hypernet_kwargs["NUM_LAYERS"], use_layer_norm=self.hypernet_kwargs["USE_LAYER_NORM"])
        biases = bias_hypernet(hyper_in).reshape(time_steps, batch_size, 1, out_dim)

        # compute y = xW + b
        # NOTE: slicing here expands embedding to be (1, in_dim) @ (in_dim, out_dim)
        # with leading dims for time_steps, batch_size
        target_out = jnp.matmul(target_in[:, :, None, :], weights) + biases
        target_out = target_out.squeeze(axis=2) # remove extra dim needed for computation
        return target_out

    @nn.compact
    def __call__(self, hidden, x, train=True):
        orig_obs, dones = x

        # separate obs into capabilities and observations
        # (env gives obs = orig obs+cap)
        # NOTE: this is hardcoded to match simple_spread's computation
        cap = orig_obs[:, :, -self.dim_capabilities:]
        obs = orig_obs[:, :, :-self.dim_capabilities]

        time_steps, batch_size, obs_dim = obs.shape

        # encoder MLP
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        # RNN 
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        # TODO: try sending hidden state to hypernet instead of raw obs
        # issue is that ScannedRNN doesn't aggregate the hidden state over
        # timesteps, so it's not possible to concatenate the hidden_state to
        # caps when leading dim TS != 1

        # decoder hypernet
        q_vals = self.hyper_forward(self.hidden_dim, self.action_dim, embedding, orig_obs, time_steps, batch_size)
        return hidden, q_vals

class HyperNetwork(nn.Module):
    """HyperNetwork for generating weights of QMix' mixing network."""
    hidden_dim: int
    output_dim: int
    init_scale: float
    # defaults for QMIX mixer network
    num_layers: int = 2
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers-1):
            x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)

        x = nn.Dense(self.output_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(x)
        return x
