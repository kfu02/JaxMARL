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
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float
    hypernet_hidden_dim: int
    hypernet_init_scale: float
    dim_capabilities: int # per team

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
        # w_2 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.hidden_dim*self.action_dim, init_scale=self.hypernet_init_scale)(cap_repr)
        # b_2 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.action_dim, init_scale=self.hypernet_init_scale)(cap_repr)
        
        # reshaping
        w_1 = w_1.reshape(time_steps, batch_size, self.hidden_dim, self.action_dim)
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.action_dim)
        # w_2 = w_2.reshape(time_steps, batch_size, self.hidden_dim, self.action_dim)
        # b_2 = b_2.reshape(time_steps, batch_size, 1, self.action_dim)
    
        # two-layer encoder
        embedding = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs))
        embedding = nn.relu(nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding))

        # target network after encoder
        q_vals = jnp.matmul(embedding[:, :, None, :], w_1) + b_1
        # embedding = nn.relu(jnp.matmul(obs[:, :, None, :], w_1) + b_1)
        # q_vals = jnp.matmul(embedding, w_2) + b_2
        # q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)

        # num_weights = (dim_capabilities * self.hidden_dim)
        # weight_hypernet = HyperNetwork(hidden_dim=self.hypernet_dim, output_dim=num_weights, init_scale=self.hypernet_init_scale)
        # w_1 = weight_hypernet(cap_repr).reshape(time_steps, batch_size, dim_capabilities, self.hidden_dim)
        #
        # num_biases = self.hidden_dim
        # bias_hypernet = HyperNetwork(hidden_dim=self.hypernet_dim, output_dim=num_biases, init_scale=0)
        # b_1 = bias_hypernet(cap_repr).reshape(time_steps, batch_size, 1, self.hidden_dim)
        #
        # num_weights = (self.hidden_dim * self.hidden_dim)
        # weight_hypernet = HyperNetwork(hidden_dim=self.hypernet_dim, output_dim=num_weights, init_scale=self.hypernet_init_scale)
        # w_2 = weight_hypernet(cap_repr).reshape(time_steps, batch_size, self.hidden_dim, self.hidden_dim)
        #
        # num_biases = self.hidden_dim
        # bias_hypernet = HyperNetwork(hidden_dim=self.hypernet_dim, output_dim=num_biases, init_scale=0)
        # b_2 = bias_hypernet(cap_repr).reshape(time_steps, batch_size, 1, self.hidden_dim)
        #
        # embedding = (cap_repr[:, :, None, :] @ w_1) + b_1
        # embedding = nn.relu(embedding)
        # embedding = (embedding @ w_2) + b_2
        # embedding = nn.relu(embedding)

        # q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)

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
    hypernet_dim: int
    hypernet_init_scale: int
    dim_capabilities: int # per team

    @nn.compact
    def __call__(self, hidden, x):
        orig_obs, dones = x

        # separate obs into capabilities and observations
        # (env gives obs = orig obs+cap)
        # NOTE: this is hardcoded to match simple_spread's computation
        cap = orig_obs[:, :, -self.dim_capabilities:]
        obs = orig_obs[:, :, :-self.dim_capabilities]

        time_steps, batch_size, obs_dim = obs.shape

        # encoder
        # original
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        # hypernet
        # hyper_input = obs_dim
        # hyper_output = self.hidden_dim
        # num_weights = (hyper_input * hyper_output)
        # weight_hypernet = HyperNetwork(hidden_dim=self.hypernet_dim, output_dim=num_weights, init_scale=self.hypernet_init_scale)
        # weights = weight_hypernet(cap).reshape(time_steps, batch_size, hyper_input, hyper_output)
        #
        # bias_hypernet = HyperNetwork(hidden_dim=self.hypernet_dim, output_dim=hyper_output, init_scale=0)
        # biases = bias_hypernet(cap).reshape(time_steps, batch_size, 1, hyper_output)

        # NOTE: slicing here expands embedding to be (1, embed_dim) @ (embed_dim, act_dim)
        # with leading dims for time_steps, batch_size
        # embedding = jnp.matmul(obs[:, :, None, :], weights) + biases
        # embedding = embedding.squeeze(axis=2) # remove extra dim needed for computation
        # embedding = nn.relu(embedding)

        # RNN 
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # decoder
        # original
        # q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)

        # hypernet
        num_weights = (self.hidden_dim * self.action_dim)
        weight_hypernet = HyperNetwork(hidden_dim=self.hypernet_dim, output_dim=num_weights, init_scale=self.hypernet_init_scale)
        weights = weight_hypernet(orig_obs).reshape(time_steps, batch_size, self.hidden_dim, self.action_dim)

        num_biases = self.action_dim
        bias_hypernet = HyperNetwork(hidden_dim=self.hypernet_dim, output_dim=num_biases, init_scale=0)
        biases = bias_hypernet(orig_obs).reshape(time_steps, batch_size, 1, self.action_dim)

        # manually calculate q_vals = (embedding @ weights) + b
        # NOTE: slicing here expands embedding to be (1, embed_dim) @ (embed_dim, act_dim)
        # with leading dims for time_steps, batch_size
        q_vals = jnp.matmul(embedding[:, :, None, :], weights) + biases
        q_vals = q_vals.squeeze(axis=2) # remove extra dim needed for computation

        return hidden, q_vals

class HyperNetwork(nn.Module):
    """HyperNetwork for generating weights of QMix' mixing network."""
    hidden_dim: int
    output_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(x)
        return x

