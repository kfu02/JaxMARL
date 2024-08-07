"""
End-to-End JAX Implementation of QMix.

Notice:
- Agents are controlled by a single RNN architecture.
- You can choose if sharing parameters between agents or not.
- Works also with non-homogenous agents (different obs/action spaces)
- Experience replay is a simple buffer with uniform sampling.
- Uses Double Q-Learning with a target agent network (hard-updated).
- You can use TD Loss (pymarl2) or DDQN loss (pymarl)
- Adam optimizer is used instead of RMSPROP.
- The environment is reset at the end of each episode.
- Trained with a team reward (reward['__all__'])
- At the moment, last_actions are not included in the agents' observations.

The implementation closely follows the original Pymarl: https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
from flax.core import frozen_dict
import wandb
import hydra
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, SMAXLogWrapper, CTRolloutManager
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts

from jaxmarl.environments.mpe import MPEVisualizer
from jaxmarl.environments.mpe.simple import State


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


# transformer
class EncoderBlock(nn.Module):
    input_dim : int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads : int  # NOTE: input must be divisible by num_heads
    dim_feedforward : int
    init_scale: float
    dropout_prob : float = 0.

    def setup(self):
        # Attention layer
        self.self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_prob,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
        )

        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0)),
            nn.Dense(self.input_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

        # learnable input feature scaling
        # self.input_feature_scaling = self.param('input_feature_scaling', nn.initializers.ones, (self.input_dim,))

    def __call__(self, x, mask=None, deterministic=False):
        # masking
        if mask is not None:
            mask = jnp.repeat(nn.make_attention_mask(mask, mask), self.num_heads, axis=-3)

        # input normalization (by z-score, only within team -- makes no assumptions about capability distribution but loses absolute info)
        x = (x - x.mean(axis=-2, keepdims=True)) / (x.std(axis=-2, keepdims=True) + 1e-9)

        # learnable input feature scaling (apply abs as scaling should not change sign)
        # x = x * jnp.abs(self.input_feature_scaling)

        # self attention
        attended = self.self_attn(inputs_q=x, inputs_kv=x, mask=mask, deterministic=deterministic)

        # see the effect of attention
        # if deterministic:
        #     jax.debug.print("x {} -> attn {}", x, attended)

        x = self.norm1(attended + x)
        x = x + self.dropout(x, deterministic=deterministic)

        # MLP part
        feedforward = self.linear[0](x)
        feedforward = nn.relu(feedforward)
        feedforward = self.linear[1](feedforward)

        x = self.norm2(feedforward+x)
        x = x + self.dropout(x, deterministic=deterministic)

        return x


class TransformerEncoder(nn.Module):
    num_layers : int
    input_dim : int # of a token, in our case dim_cap
    num_heads : int
    dim_feedforward : int
    dropout_prob : float
    init_scale : int

    def setup(self):
        self.layers = [EncoderBlock(input_dim=self.input_dim, 
                                    num_heads=self.num_heads,
                                    dim_feedforward=self.dim_feedforward,
                                    init_scale=self.init_scale,
                                    dropout_prob=self.dropout_prob)
                       for _ in range(self.num_layers)]

    def __call__(self, x, mask=None, train=True):
        for l in self.layers:
            x = l(x, mask=mask, deterministic=not train)
        return x


class AgentHyperRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float
    hypernet_dim: int
    hypernet_init_scale: int
    num_capabilities: int # per agent
    num_agents: int
    use_capability_transformer: bool
    transformer_kwargs: dict

    @nn.compact
    def __call__(self, hidden, x, train=True):
        obs, dones = x

        # separate obs into capabilities and observations
        # (env gives obs = orig obs+cap)
        # NOTE: this is hardcoded to match simple_spread's computation
        dim_capabilities = self.num_agents * self.num_capabilities
        cap = obs[:, :, -dim_capabilities:]
        obs = obs[:, :, :-dim_capabilities]

        # only feed obs through RNN encoder
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # transform capabilities via transformer before passing to cap_hypernet (if flag given)
        cap_repr = cap
        if self.use_capability_transformer:
            # break apart the capabilities by agent, so that attention is applied across agents
            time_steps, batch_size, _ = cap.shape
            reshaped_cap = jnp.reshape(cap, (time_steps * batch_size, self.num_agents, self.num_capabilities)) # ["batch_size", seq_len, tkn_len]

            # apply a "positional embedding", but instead of the advanced cos/sin embedding, just add a 0/1 IS_SELF flag a la transfqmix (and IS_SELF should always be applied to the first agent)
            is_self_flag = jnp.zeros((time_steps*batch_size, self.num_agents, 1)).at[:, 0, 0].set(1)
            reshaped_cap = jnp.concatenate([reshaped_cap, is_self_flag], axis=-1)

            # NOTE: rejected using an embedding to preprocess the capabilities as they are already semantically meaningful (and experimental evidence showed poorer performance)
            transformer_encoder = TransformerEncoder(
                input_dim=self.num_capabilities+1,
                num_layers=self.transformer_kwargs["NUM_LAYERS"],
                num_heads=self.transformer_kwargs["NUM_HEADS"],
                dim_feedforward=self.transformer_kwargs["DIM_FF"],
                init_scale=self.transformer_kwargs["INIT_SCALE"],
                dropout_prob=self.transformer_kwargs["DROPOUT_PROB"],
            )
            transformed_cap = transformer_encoder(reshaped_cap, train=train)
            # if not train:
            #     jax.debug.print("orig cap {} -> {}", reshaped_cap[0, ...], transformed_cap[0, ...])

            # only take the transformed version of the ego-agent's capabilities
            cap_repr = transformed_cap[:, 0, :].reshape((time_steps, batch_size, -1))

        # then use capability hypernet for last layer
        num_weights = (self.action_dim * self.hidden_dim)
        num_biases = self.action_dim
        cap_hypernet = HyperNetwork(hidden_dim=self.hypernet_dim, output_dim=num_weights+num_biases, init_scale=self.hypernet_init_scale)(cap_repr)

        # extract weights + biases from hypernet output
        time_steps, batch_size, obs_dim = obs.shape
        weights = cap_hypernet[:, :, :num_weights].reshape(time_steps, batch_size, self.hidden_dim, self.action_dim)
        biases = cap_hypernet[:, :, -num_biases:].reshape(time_steps, batch_size, 1, self.action_dim)

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

class MixingNetwork(nn.Module):
    """
    Mixing network for projecting individual agent Q-values into Q_tot. Follows the original QMix implementation.
    """
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, q_vals, states):
        
        n_agents, time_steps, batch_size = q_vals.shape
        q_vals = jnp.transpose(q_vals, (1, 2, 0)) # (time_steps, batch_size, n_agents)
        
        # hypernetwork
        w_1 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.embedding_dim*n_agents, init_scale=self.init_scale)(states)
        b_1 = nn.Dense(self.embedding_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(states)
        w_2 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.embedding_dim, init_scale=self.init_scale)(states)
        b_2 = HyperNetwork(hidden_dim=self.embedding_dim, output_dim=1, init_scale=self.init_scale)(states)
        
        # monotonicity and reshaping
        w_1 = jnp.abs(w_1.reshape(time_steps, batch_size, n_agents, self.embedding_dim))
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(time_steps, batch_size, self.embedding_dim, 1))
        b_2 = b_2.reshape(time_steps, batch_size, 1, 1)
    
        # mix
        hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
        q_tot  = jnp.matmul(hidden, w_2) + b_2
        
        return q_tot.squeeze() # (time_steps, batch_size)


class EpsilonGreedy:
    """Epsilon Greedy action selection"""

    def __init__(self, start_e: float, end_e: float, duration: int):
        self.start_e  = start_e
        self.end_e    = end_e
        self.duration = duration
        self.slope    = (end_e - start_e) / duration
        
    @partial(jax.jit, static_argnums=0)
    def get_epsilon(self, t: int):
        e = self.slope*t + self.start_e
        return jnp.clip(e, self.end_e)
    
    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: dict, t: int, rng: chex.PRNGKey):
        
        def explore(q, eps, key):
            key_a, key_e   = jax.random.split(key, 2) # a key for sampling random actions and one for picking
            greedy_actions = jnp.argmax(q, axis=-1) # get the greedy actions 
            random_actions = jax.random.randint(key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1]) # sample random actions
            pick_random    = jax.random.uniform(key_e, greedy_actions.shape)<eps # pick which actions should be random
            chosed_actions = jnp.where(pick_random, random_actions, greedy_actions)
            return chosed_actions
        
        eps = self.get_epsilon(t)
        keys = dict(zip(q_vals.keys(), jax.random.split(rng, len(q_vals)))) # get a key for each agent
        chosen_actions = jax.tree.map(lambda q, k: explore(q, eps, k), q_vals, keys)
        return chosen_actions

class Transition(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    infos: dict


def make_train(config, log_train_env, log_test_env, viz_test_env):
    """
    NOTE: log_train_env and log_test_env should be identical, other than a single
    test_env_flag, which causes log_test_env to only sample from test_capabilities.
    """

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    
    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        # TODO: add preprocess_obs flag to config, maybe rename to "agent_ID"
        # NOTE: added preprocess_obs=False to avoid adding agent_ids to each obs
        # NOTE: this has the side effect of also removing any zero-padding to
        # standardize obs dimensions across agents, may be issue later
        wrapped_env = CTRolloutManager(log_train_env, batch_size=config["NUM_ENVS"], preprocess_obs=False)
        test_env = CTRolloutManager(log_test_env, batch_size=config["NUM_TEST_EPISODES"], preprocess_obs=False) # batched env for testing (has different batch size), chooses fixed team compositions on reset (may be different from train set)
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in log_train_env.agents+['__all__']}

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3) # use a dummy rng here
            key_a = jax.random.split(key_a, log_train_env.num_agents)
            actions = {agent: wrapped_env.batch_sample(key_a[i], agent) for i, agent in enumerate(log_train_env.agents)}
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
            transition = Transition(obs, actions, rewards, dones, infos)
            return env_state, transition
        _, sample_traj = jax.lax.scan(
            _env_sample_step, env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(lambda x: x[:, 0], sample_traj) # remove the NUM_ENV dim
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'],
            min_length_time_axis=config['BUFFER_BATCH_SIZE'],
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
            add_batch_size=config['NUM_ENVS'],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched) 

        # INIT NETWORK
        # init agent
        if not config["AGENT_HYPERAWARE"]:
            agent = AgentRNN(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])
        else:
            agent = AgentHyperRNN(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'], hypernet_dim=config["AGENT_HYPERNET_DIM"], hypernet_init_scale=config["AGENT_HYPERNET_INIT_SCALE"], num_capabilities=log_train_env.num_capabilities, num_agents=log_train_env.num_agents, use_capability_transformer=config["AGENT_USE_CAPABILITY_TRANSFORMER"], transformer_kwargs=config["AGENT_TRANSFORMER_KWARGS"])
        rng, _rng = jax.random.split(rng)

        if config["PARAMETERS_SHARING"]:
            init_x = (
                jnp.zeros((1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)) # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], 1) # (batch_size, hidden_dim)
            agent_params = agent.init(_rng, init_hs, init_x)
        else:
            init_x = (
                jnp.zeros((len(log_train_env.agents), 1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
                jnp.zeros((len(log_train_env.agents), 1, 1)) # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(log_train_env.agents),  1) # (n_agents, batch_size, hidden_dim)
            rngs = jax.random.split(_rng, len(log_train_env.agents)) # a random init for each agent
            agent_params = jax.vmap(agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)

        # log agent param count
        agent_param_count = sum(x.size for x in jax.tree_util.tree_leaves(agent_params))
        wandb.log({"agent_param_count": agent_param_count})
        print("-" * 10)
        print("DETAILED AGENT PARAM COUNT:")
        for name, param in jax.tree_util.tree_flatten_with_path(agent_params)[0]:
            print(f"{name}: {param.shape}")
        print("-" * 10)

        # init mixer
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((len(log_train_env.agents), 1, 1))
        state_size = sample_traj.obs['__all__'].shape[-1]  # get the state shape from the buffer
        init_state = jnp.zeros((1, 1, state_size))
        mixer = MixingNetwork(config['MIXER_EMBEDDING_DIM'], config["MIXER_HYPERNET_HIDDEN_DIM"], config['MIXER_INIT_SCALE'])
        mixer_params = mixer.init(_rng, init_x, init_state)

        # init optimizer
        network_params = frozen_dict.freeze({'agent':agent_params, 'mixer':mixer_params})
        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac
        lr = linear_schedule if config.get('LR_LINEAR_DECAY', False) else config['LR']
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(learning_rate=lr, eps=config['EPS_ADAM'], weight_decay=config['WEIGHT_DECAY_ADAM']),
        )
        train_state = TrainState.create(
            apply_fn=None,
            params=network_params,
            tx=tx,
        )
        # target network params
        target_network_params = jax.tree.map(lambda x: jnp.copy(x), train_state.params)

        # INIT EXPLORATION STRATEGY
        explorer = EpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )

        # depending if using parameters sharing or not, q-values are computed using one or multiple parameters
        if config["PARAMETERS_SHARING"]:
            def homogeneous_pass(params, hidden_state, obs, dones, train=True):
                # concatenate agents and parallel envs to process them in one batch
                agents, flatten_agents_obs = zip(*obs.items())
                original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
                batched_input = (
                    jnp.concatenate(flatten_agents_obs, axis=1), # (time_step, n_agents*n_envs, obs_size)
                    jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
                )
                hidden_state, q_vals = agent.apply(params, hidden_state, batched_input, train=train)
                q_vals = q_vals.reshape(original_shape[0], len(agents), *original_shape[1:-1], -1) # (time_steps, n_agents, n_envs, action_dim)
                q_vals = {a:q_vals[:,i] for i,a in enumerate(agents)}
                return hidden_state, q_vals
        else:
            def homogeneous_pass(params, hidden_state, obs, dones):
                # homogeneous pass vmapped in respect to the agents parameters (i.e., no parameter sharing)
                agents, flatten_agents_obs = zip(*obs.items())
                batched_input = (
                    jnp.stack(flatten_agents_obs, axis=0), # (n_agents, time_step, n_envs, obs_size)
                    jnp.stack([dones[agent] for agent in agents], axis=0), # ensure to not pass other keys (like __all__)
                )
                # computes the q_vals with the params of each agent separately by vmapping
                hidden_state, q_vals = jax.vmap(agent.apply, in_axes=0)(params, hidden_state, batched_input)
                q_vals = {a:q_vals[i] for i,a in enumerate(agents)}
                return hidden_state, q_vals


        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, target_network_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, viz_env_states, rng = runner_state

            # EPISODE STEP
            def _env_step(step_state, unused):

                params, env_state, last_obs, last_dones, hstate, rng, t = step_state

                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                # SELECT ACTION
                # add a dummy time_step dimension to the agent input
                obs_   = {a:last_obs[a] for a in log_train_env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                obs_   = jax.tree.map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree.map(lambda x: x[np.newaxis, :], last_dones)
                # get the q_values from the agent netwoek
                hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
                # remove the dummy time_step dimension and index qs by the valid actions of each agent 
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, wrapped_env.valid_actions)
                # explore with epsilon greedy_exploration
                actions = explorer.choose_actions(valid_q_vals, t, key_a)

                # STEP ENV
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, actions, rewards, dones, infos)

                step_state = (params, env_state, obs, dones, hstate, rng, t+1)
                return step_state, transition


            # prepare the step state and collect the episode trajectory
            rng, _rng = jax.random.split(rng)
            if config["PARAMETERS_SHARING"]:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(log_train_env.agents)*config["NUM_ENVS"]) # (n_agents*n_envs, hs_size)
            else:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(log_train_env.agents), config["NUM_ENVS"]) # (n_agents, n_envs, hs_size)

            step_state = (
                train_state.params['agent'],
                env_state,
                init_obs,
                init_dones,
                hstate, 
                _rng,
                time_state['timesteps'] # t is needed to compute epsilon
            )

            step_state, traj_batch = jax.lax.scan(
                _env_step, step_state, None, config["NUM_STEPS"]
            )

            # BUFFER UPDATE: save the collected trajectory in the buffer
            buffer_traj_batch = jax.tree_util.tree_map(
                lambda x:jnp.swapaxes(x, 0, 1)[:, np.newaxis], # put the batch dim first and add a dummy sequence dim
                traj_batch
            ) # (num_envs, 1, time_steps, ...)
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # LEARN PHASE
            def q_of_action(q, u):
                """index the q_values with action indices"""
                q_u = jnp.take_along_axis(q, jnp.expand_dims(u, axis=-1), axis=-1)
                return jnp.squeeze(q_u, axis=-1)

            def _loss_fn(params, target_network_params, init_hstate, learn_traj):

                obs_ = {a:learn_traj.obs[a] for a in log_train_env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                _, q_vals = homogeneous_pass(params['agent'], init_hstate, obs_, learn_traj.dones)
                _, target_q_vals = homogeneous_pass(target_network_params['agent'], init_hstate, obs_, learn_traj.dones)

                # get the q_vals of the taken actions (with exploration) for each agent
                chosen_action_qvals = jax.tree.map(
                    lambda q, u: q_of_action(q, u)[:-1], # avoid last timestep
                    q_vals,
                    learn_traj.actions
                )

                # get the target q value of the greedy actions for each agent
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q[..., valid_idx], q_vals, wrapped_env.valid_actions)
                target_max_qvals = jax.tree.map(
                    lambda t_q, q: q_of_action(t_q, jnp.argmax(q, axis=-1))[1:], # avoid first timestep
                    target_q_vals,
                    jax.lax.stop_gradient(valid_q_vals)
                )

                # compute q_tot with the mixer network
                chosen_action_qvals_mix = mixer.apply(
                    params['mixer'], 
                    jnp.stack(list(chosen_action_qvals.values())),
                    learn_traj.obs['__all__'][:-1] # avoid last timestep
                )
                target_max_qvals_mix = mixer.apply(
                    target_network_params['mixer'], 
                    jnp.stack(list(target_max_qvals.values())),
                    learn_traj.obs['__all__'][1:] # avoid first timestep
                )

                # compute target
                if config.get('TD_LAMBDA_LOSS', True):
                    # time difference loss
                    def _td_lambda_target(ret, values):
                        reward, done, target_qs = values
                        ret = jnp.where(
                            done,
                            target_qs,
                            ret*config['TD_LAMBDA']*config['GAMMA']
                            + reward
                            + (1-config['TD_LAMBDA'])*config['GAMMA']*(1-done)*target_qs
                        )
                        return ret, ret

                    ret = target_max_qvals_mix[-1] * (1-learn_traj.dones['__all__'][-1])
                    ret, td_targets = jax.lax.scan(
                        _td_lambda_target,
                        ret,
                        (learn_traj.rewards['__all__'][-2::-1], learn_traj.dones['__all__'][-2::-1], target_max_qvals_mix[-1::-1])
                    )
                    targets = td_targets[::-1]
                    loss = jnp.mean(0.5*((chosen_action_qvals_mix - jax.lax.stop_gradient(targets))**2))
                else:
                    # standard DQN loss
                    targets = (
                        learn_traj.rewards['__all__'][:-1]
                        + config['GAMMA']*(1-learn_traj.dones['__all__'][:-1])*target_max_qvals_mix
                    )
                    loss = jnp.mean((chosen_action_qvals_mix - jax.lax.stop_gradient(targets))**2)
                
                return loss


            # sample a batched trajectory from the buffer and set the time step dim in first axis
            rng, _rng = jax.random.split(rng)
            learn_traj = buffer.sample(buffer_state, _rng).experience # (batch_size, 1, max_time_steps, ...)
            learn_traj = jax.tree.map(
                lambda x: jnp.swapaxes(x[:, 0], 0, 1), # remove the dummy sequence dim (1) and swap batch and temporal dims
                learn_traj
            ) # (max_time_steps, batch_size, ...)
            if config["PARAMETERS_SHARING"]:
                init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(log_train_env.agents)*config["BUFFER_BATCH_SIZE"]) # (n_agents*batch_size, hs_size)
            else:
                init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(log_train_env.agents), config["BUFFER_BATCH_SIZE"]) # (n_agents, batch_size, hs_size)

            # compute loss and optimize grad
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            loss, grads = grad_fn(train_state.params, target_network_params, init_hs, learn_traj)
            train_state = train_state.apply_gradients(grads=grads)


            # UPDATE THE VARIABLES AND RETURN
            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in log_train_env.agents+['__all__']}

            # update the states
            time_state['timesteps'] = step_state[-1]
            time_state['updates']   = time_state['updates'] + 1

            # update the target network if necessary
            target_network_params = jax.lax.cond(
                time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                lambda _: jax.tree.map(lambda x: jnp.copy(x), train_state.params),
                lambda _: target_network_params,
                operand=None
            )

            # update the greedy rewards
            rng, _rng = jax.random.split(rng)
            test_results = jax.lax.cond(
                time_state['updates'] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                lambda _: get_greedy_metrics(_rng, train_state.params['agent'], time_state),
                lambda _: {"metrics": test_metrics, "viz_env_states": viz_env_states},
                operand=None
            )
            test_metrics, viz_env_states = test_results["metrics"], test_results["viz_env_states"]

            # update the returning metrics
            metrics = {
                'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                'updates' : time_state['updates'],
                'loss': loss,
                'rewards': jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0).mean(), traj_batch.rewards),
                'eps': explorer.get_epsilon(time_state['timesteps'])
            }
            metrics['test_metrics'] = test_metrics # add the test metrics dictionary

            if config.get('WANDB_ONLINE_REPORT', False):
                def callback(metrics, infos):
                    info_metrics = {
                        k:v[...,0][infos["returned_episode"][..., 0]].mean()
                        for k,v in infos.items() if k!="returned_episode"
                    }
                    wandb.log(
                        {
                            "returns": metrics['rewards']['__all__'].mean(),
                            "timestep": metrics['timesteps'],
                            "updates": metrics['updates'],
                            "loss": metrics['loss'],
                            'epsilon': metrics['eps'],
                            **info_metrics,
                            **{k:v.mean() for k, v in metrics['test_metrics'].items()}
                        }
                    )
                jax.debug.callback(callback, metrics, traj_batch.infos)

            runner_state = (
                train_state,
                target_network_params,
                env_state,
                buffer_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                viz_env_states,
                rng,
            )

            return runner_state, metrics

        def get_greedy_metrics(rng, params, time_state):
            """Help function to test greedy policy during training"""
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                obs_   = {a:last_obs[a] for a in log_test_env.agents}
                obs_   = jax.tree.map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree.map(lambda x: x[np.newaxis, :], last_dones)
                hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_, train=False)
                actions = jax.tree_util.tree_map(lambda q, valid_idx: jnp.argmax(q.squeeze(0)[..., valid_idx], axis=-1), q_vals, test_env.valid_actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos, env_state.env_state) # save all EnvState (not LogEnvState) to visualize
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in log_test_env.agents+['__all__']}
            rng, _rng = jax.random.split(rng)
            if config["PARAMETERS_SHARING"]:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(log_test_env.agents)*config["NUM_TEST_EPISODES"]) # (n_agents*n_envs, hs_size)
            else:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(log_test_env.agents), config["NUM_TEST_EPISODES"]) # (n_agents, n_envs, hs_size)
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate, 
                _rng,
            )
            step_state, (rewards, dones, infos, viz_env_states) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_STEPS"]
            )

            # compute the pct of landmarks covered by an agent at the final timestep (across all envs)
            def pct_landmarks_covered(final_step_state):
                final_env_state = final_step_state[1].env_state
                p_pos = final_env_state.p_pos
                rad = final_env_state.rad
                n_agents = p_pos.shape[-2] // 2
                n_envs = config["NUM_TEST_EPISODES"]

                # TODO: if bored, try to vectorize over n_envs too
                covered_landmarks = 0
                for env in range(n_envs):
                    for landmark in range(n_agents):
                        # get dist of this landmark to all agents
                        landmark_pos = p_pos[env, n_agents+landmark, :]
                        agent_pos = p_pos[env, :n_agents, :]
                        delta_pos = agent_pos - landmark_pos
                        dist_to_agents = jnp.sqrt(jnp.sum(jnp.square(delta_pos), axis=1))

                        # then find the closest agent based on this array 
                        # if that agent's radius > its dist to landmark, it covers it, add to tally
                        closest_agent = jnp.argmin(dist_to_agents)
                        closest_agent_dist = dist_to_agents[closest_agent]
                        closest_agent_rad = rad[env, closest_agent]
                        covered_landmarks = jax.lax.select(closest_agent_rad > closest_agent_dist,
                                                           covered_landmarks+1,
                                                           covered_landmarks)

                return covered_landmarks / (n_agents * n_envs) # divide by n_envs because we are tallying over all n_envs

            # compute the metrics of the first episode that is done for each parallel env
            def first_episode_returns(rewards, dones):
                first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
                first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                return jnp.where(first_episode_mask, rewards, 0.).sum()
            all_dones = dones['__all__']
            first_returns = jax.tree.map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), rewards)
            first_infos   = jax.tree.map(lambda i: jax.vmap(first_episode_returns, in_axes=1)(i[..., 0], all_dones), infos)
            metrics = {
                'test_returns': first_returns['__all__'],# episode returns
                'test_pct_landmarks_covered': pct_landmarks_covered(step_state),
                **{'test_'+k:v for k,v in first_infos.items()},
            }
            if config.get('VERBOSE', False):
                def callback(timestep, val):
                    print(f"Timestep: {timestep}, return: {val}")
                jax.debug.callback(callback, time_state['timesteps']*config['NUM_ENVS'], first_returns['__all__'].mean())
            return {"metrics": metrics, "viz_env_states": viz_env_states}
        
        time_state = {
            'timesteps':jnp.array(0),
            'updates':  jnp.array(0)
        }
        rng, _rng = jax.random.split(rng)
        greedy_ret = get_greedy_metrics(_rng, train_state.params['agent'],time_state) # initial greedy metrics
        test_metrics, viz_env_states = greedy_ret["metrics"], greedy_ret["viz_env_states"]

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            target_network_params,
            env_state,
            buffer_state,
            time_state,
            init_obs,
            init_dones,
            test_metrics,
            viz_env_states,
            _rng
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {'runner_state':runner_state, 'metrics':metrics}
    
    return train

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)

    print('Config:\n', OmegaConf.to_yaml(config))

    env_name = config["env"]["ENV_NAME"]
    alg_name = f'qmix_{"ps" if config["alg"].get("PARAMETERS_SHARING", True) else "ns"}'
    
    # smac init neeeds a scenario
    if 'smax' in env_name.lower():
        config['env']['ENV_KWARGS']['scenario'] = map_name_to_scenario(config['env']['MAP_NAME'])
        env_name = f"{config['env']['ENV_NAME']}_{config['env']['MAP_NAME']}"
        orig_env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        env = SMAXLogWrapper(orig_env)
        # TODO: add test_env to smax
   # overcooked needs a layout 
    elif 'overcooked' in env_name.lower():
        # NOTE: overcooked will not work with current pipeline
        config['env']["ENV_KWARGS"]["layout"] = overcooked_layouts[config['env']["ENV_KWARGS"]["layout"]]
        env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        env = LogWrapper(env)
    else:
        train_env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        log_train_env = LogWrapper(train_env)
        viz_test_env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'], test_env_flag=True)
        log_test_env = LogWrapper(viz_test_env)

    #config["alg"]["NUM_STEPS"] = config["alg"].get("NUM_STEPS", env.max_steps) # default steps defined by the env
    
    hyper_tag = "HYPER" if config["alg"]["AGENT_HYPERAWARE"] else "RNN"
    aware_tag = "aware" if config["env"]["ENV_KWARGS"]["capability_aware"] else "unaware"
    cap_transf_tag = "transformed-cap" if config["alg"]["AGENT_USE_CAPABILITY_TRANSFORMER"] else ""

    wandb_tags = [
        alg_name.upper(),
        env_name.upper(),
        hyper_tag,
        aware_tag,
        "TD_LOSS" if config["alg"].get("TD_LAMBDA_LOSS", True) else "DQN_LOSS",
        f"jax_{jax.__version__}",
    ]
    if config["alg"]["AGENT_USE_CAPABILITY_TRANSFORMER"]:
        wandb_tags.append("cap-transformer")

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=wandb_tags,
        name=f'{hyper_tag} {aware_tag} {cap_transf_tag} / {env_name.upper()}',
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config["alg"], log_train_env, log_test_env, viz_test_env)))
    outs = jax.block_until_ready(train_vjit(rngs))
    
    # save params
    if config['SAVE_PATH'] is not None:

        def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
            flattened_dict = flatten_dict(params, sep=',')
            save_file(flattened_dict, filename)

        model_state = outs['runner_state'][0]
        params = jax.tree.map(lambda x: x[0], model_state.params) # save only params of the firt run
        save_dir = os.path.join(config['SAVE_PATH'], env_name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/{alg_name}.safetensors')
        print(f'Parameters of first batch saved in {save_dir}/{alg_name}.safetensors')

        if config["VISUALIZE_FINAL_POLICY"]:
            viz_env_states = outs['runner_state'][-2]

            # build a list of states manually from vectorized seq returned by
            # make_train() for desired seeds/envs
            for seed in range(config["VIZ_NUM_SEEDS"]):
                for env in range(config["VIZ_NUM_ENVS"]):
                    state_seq = []
                    for i in range(config["alg"]["NUM_STEPS"]):
                        this_step_state = State(
                            p_pos=viz_env_states.p_pos[seed, i, env, ...],
                            p_vel=viz_env_states.p_vel[seed, i, env, ...],
                            c=viz_env_states.c[seed, i, env, ...],
                            accel=viz_env_states.accel[seed, i, env, ...],
                            rad=viz_env_states.rad[seed, i, env, ...],
                            done=viz_env_states.done[seed, i, env, ...],
                            step=i,
                        )
                        state_seq.append(this_step_state)

                    # save visualization to GIF for wandb display
                    visualizer = MPEVisualizer(viz_test_env, state_seq)
                    video_fpath = f'{save_dir}/{alg_name}-seed-{seed}-rollout.gif'
                    visualizer.animate(video_fpath)
                    wandb.log({f"env-{env}-seed-{seed}-rollout": wandb.Video(video_fpath)})

if __name__ == "__main__":
    main()
    
