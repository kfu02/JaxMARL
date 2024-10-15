"""
TODO: rename this file & config files to DAgger!

Basic behavioral cloning implementation.

NOTE: only tested on MPESimpleFire.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union, Callable

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

from jaxmarl.environments.mpe import MPEVisualizer
from jaxmarl.environments.mpe.simple import State

from jaxmarl.policies import ScannedRNN, AgentMLP, AgentHyperMLP, AgentRNN, AgentHyperRNN, HyperNetwork
from jaxmarl.utils import snd

import more_itertools as mit

class Transition(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    infos: dict

class TransitionHstate(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    infos: dict
    hstate: dict

def expert_heuristic_material_transport(obs_dict, cached_values):
    """
    Expert policy to gather samples from for HMT env.
    pi(obs) -> action for each agent/obs

    Input: obs_dict = {agent_i : [timesteps, num_envs, obs_dim]}
    Output: actions = {agent_i : [timesteps, num_envs]} (rather than outputting act_dim, directly output index of best action)

    cached_values are unused.
    """
    # Stack observations to be per-environment
    all_obs = jnp.stack(list(obs_dict.values()))  # [n_agents, ?, n_envs, obs_dim]
    all_obs = all_obs.squeeze(1)  # [n_agents, n_envs, obs_dim]
    
    n_agents, n_envs, obs_dim = all_obs.shape
    
    # action space
    unit_vectors = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])  # [9, 2]

    # split up observation based on below observation space construction 
    # obs = jnp.concatenate([
    #     ego_pos.flatten(),  # 2
    #     rel_other_pos.flatten(),  # N-1, 2
    #     ego_vel.flatten(),  # 2
    #     rel_landmark_p_pos.flatten(), # 3, 2
    #     state.site_quota.flatten() # 1
    #     payload.flatten(), # 1
    #     # NOTE: caps must go last for hypernet logic
    #     ego_cap.flatten(),  # n_cap
    #     other_cap.flatten(),  # N-1, n_cap
    # ])

    # landmark relative locations ordered [concrete depot, lumber depot, construction site]
    landmark_start = (4 + (n_agents-1)*2)
    landmark_end = landmark_start+(3*2)
    rel_landmark_p_pos = all_obs[..., landmark_start:landmark_end].reshape(n_agents, n_envs, 3, 2)  # [n_agents, n_envs, 3, 2]

    # site quota
    quota_start = landmark_end
    quota_end = landmark_end + 2
    site_quota = all_obs[..., quota_start:quota_end]

    # payload
    payload_start = 2+(n_agents-1)*2+2+(3*2)+2
    payload_end = payload_start + 2
    payload = all_obs[..., payload_start:payload_end]  # [n_agents, n_envs]

    # ego capability
    ego_cap_start = payload_end
    ego_cap = all_obs[..., ego_cap_start:ego_cap_start + 2]  # [n_agents, n_envs, 2]

    # If payload == 0: move towards landmark idx = argmax(ego_cap), else move towards construction site
    payload_mask = jnp.bitwise_and(payload[..., 0] == 0, payload[..., 1] == 0)
    # site_quota_mask = jnp.bitwise_or(site_quota[..., 0] < 0, site_quota[..., 1] < 0)
    # target_landmark_idx = jnp.where(jnp.bitwise_and(payload_mask, site_quota_mask), jnp.argmax(ego_cap, axis=-1), 2)  # [n_agents, n_envs]

    target_landmark_idx = jnp.where(payload_mask, jnp.argmax(ego_cap, axis=-1), 2)  # [n_agents, n_envs]

    # target_landmark_idx = jnp.ones((n_agents, n_envs)).astype(int)
    
    # get vectors to landmarks
    target_landmark_rel_pos = jnp.take_along_axis(rel_landmark_p_pos, target_landmark_idx[..., None, None], axis=-2).squeeze(-2)  # [n_agents, n_envs, 2]
    norm_direction = target_landmark_rel_pos / (jnp.linalg.norm(target_landmark_rel_pos, axis=-1, keepdims=True))

    # get optimal direction
    action_alignment = jnp.einsum('aij,bj->aib', norm_direction, unit_vectors)
    optimal_actions = jnp.argmax(action_alignment, axis=-1)
    # optimal_actions = jnp.where(near_target.squeeze(), jnp.zeros_like(optimal_actions), optimal_actions)
    optimal_actions = optimal_actions.T
    actions = {}
    for i, agent_name in enumerate(obs_dict.keys()):
        actions[agent_name] = optimal_actions[:, i].reshape(1,-1)   # make compatible with squeeze in training loop
    
    return actions

def expert_heuristic_simple_fire(env_state, agent_list, cached_values):
    """
    Expert policy to gather samples from for firefighting env.

    Input: env_state = full knowledge of state at time t
    Output: actions = {agent_i : [timesteps, num_envs]} (rather than outputting act_dim, directly output index of best action)

    cached_values are there to speed up computation.
    """
    valid_set_partitions, num_landmarks = cached_values["valid_set_partitions"], cached_values["num_landmarks"]
    num_agents = len(agent_list)

    def solve_one_env(p_pos, rad):
        """
        p_pos : pos of agents & fires at time t
        rad : rad of agents & fires at time t

        return the best action for all agents
        """
        # NOTE: this is hardcoded to match get_obs() in simple_fire.py
        # first n_agents elements of p_pos are agents, rest are landmarks
        agent_pos = p_pos[:num_agents]
        fire_pos = p_pos[num_agents:]
        agent_rad = rad[:num_agents]
        fire_rad = rad[num_agents:]

        # concat a single 0 to the end of agent_rad
        # this is necessary to handle the padded -1s in valid_set_partitions, and should not mess up the true non-padded indices
        # (e.g. agent_rad = [1, 2, 3, 0], allocation [0,1], [2,-1] -> [1+2, 3+0], allocation [1,-1], [0,2] -> [2+0, 1+3]
        agent_rad = jnp.concatenate([agent_rad, jnp.zeros(shape=(1,))])

        # for each possible allocation of agents to fires
        # (set partitions is a list of lists, e.g. [[0,1],[2]], which we can interpret as 0,1 => fire 0, 2 => fire 1)
        all_rew = jnp.zeros(shape=(len(valid_set_partitions),))-10
        for a_i, allocation in enumerate(valid_set_partitions):
            # compute the reward if we allocated according to this split
            reward = 0
            for i in range(num_landmarks):
                fire_ff = fire_rad[i]
                team_ff = jnp.sum(agent_rad[allocation[i]]) 
                # cap positive reward per fire at 0
                reward += jnp.where(team_ff >= fire_ff, 0, team_ff - fire_ff)

            # update all_rew
            all_rew = all_rew.at[a_i].set(reward)

        # based on rew, find a best allocation (can be ties thanks to cap)
        best_index = jnp.argmax(all_rew).astype(int)
        best_allocation = valid_set_partitions[best_index]

        # then compute best actions for each agent based on best allocation
        def best_action_for_agent(agent_i):
            my_fire = jnp.argwhere(best_allocation == agent_i, size=1)[0][0]
            
            unit_vectors = jnp.array([[0,0], [-1,0], [+1,0], [0,-1], [0,+1]])
            dir_to_fire_pos = fire_pos[my_fire] - agent_pos[agent_i]
            dir_to_fire_pos = dir_to_fire_pos / jnp.linalg.norm(dir_to_fire_pos)
            dot = jnp.dot(unit_vectors, dir_to_fire_pos)

            # always pick the discrete action which maximizes progress towards fire
            # (as measured by similarity of discrete action with desired heading vector)
            best_action = jnp.argmax(dot)
            return best_action 

        all_best_actions = jax.vmap(best_action_for_agent)(jnp.arange(num_agents))
        return all_best_actions

    # find best actions across all envs, then separate into best for each agent/env
    all_acts = jax.vmap(solve_one_env, in_axes=[0, 0])(env_state.p_pos, env_state.rad)
    # all_acts: [n_envs, n_agents]
    actions = {}
    for i, agent_name in enumerate(agent_list):
        actions[agent_name] = all_acts[..., i]

    return actions

def make_train(config, log_train_env, log_test_env, expert_heuristic: Callable, expert_cached_values: dict, env_name="MPE_simple_fire"):
    """
    (1) collect 1 seed's worth of traj data with the expert heuristic and logs expert metrics
    (2) train an imitation learning policy to match the collected data (cross-entropy loss)
    (3) evaluate IL policy
    """
    # NOTE: this is copy-pasted from qmix.py and modified -- bad practice
    def fire_env_metrics(final_env_state):
        """
        Return success rate (pct of envs where both fires are put out)
        and percent of fires which are put out, out of all fires.
        """
        p_pos = final_env_state.p_pos
        rads = final_env_state.rad

        num_agents = log_train_env.num_agents
        num_landmarks = rads.shape[-1] - num_agents
        num_envs = config["NUM_ENVS"]

        def _agent_in_range(agent_i: int, agent_p_pos, landmark_pos, landmark_rad):
            """
            Finds all agents in range of a single landmark.
            """
            delta_pos = agent_p_pos[agent_i] - landmark_pos
            dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
            return (dist < landmark_rad)

        def _fire_put_out(landmark_i: int, agent_p_pos, agent_rads, landmark_p_pos, landmark_rads):
            """
            Determines if a single landmark is covered by enough ff power.
            """
            landmark_pos = landmark_p_pos[landmark_i, :]
            landmark_rad = landmark_rads[landmark_i]

            agents_on_landmark = jax.vmap(_agent_in_range, in_axes=[0, None, None, None])(jnp.arange(num_agents), agent_p_pos, landmark_pos, landmark_rad)
            firefighting_level = jnp.sum(jnp.where(agents_on_landmark, agent_rads, 0))
            return firefighting_level > landmark_rad

        def _fires_put_out_per_env(env_i, p_pos, rads):
            """
            Determines how many fires are covered in a single parallel env.
            """
            agent_p_pos = p_pos[env_i, :num_agents, :]
            landmark_p_pos = p_pos[env_i, num_agents:, :]

            agent_rads = rads[env_i, :num_agents]
            landmark_rads = rads[env_i, num_agents:]

            landmarks_covered = jax.vmap(_fire_put_out, in_axes=[0, None, None, None, None])(jnp.arange(num_landmarks), agent_p_pos, agent_rads, landmark_p_pos, landmark_rads)

            return landmarks_covered

        fires_put_out = jax.vmap(_fires_put_out_per_env, in_axes=[0, None, None])(jnp.arange(num_envs), p_pos, rads)
        # envs where num_landmarks fires are put out / total
        success_rate = jnp.count_nonzero(jnp.sum(fires_put_out, axis=1) == num_landmarks) / num_envs
        # sum of all fires put out / total num fires
        pct_fires_put_out = jnp.sum(fires_put_out) / (num_envs * num_landmarks)
        return success_rate, pct_fires_put_out

    def train(rng):
        # -------------------- --------------------
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        # NOTE: added preprocess_obs=False to avoid adding agent_ids to each obs
        # NOTE: this has the side effect of also removing any zero-padding to
        # standardize obs dimensions across agents, may be issue later
        wrapped_env = CTRolloutManager(log_train_env, batch_size=config["NUM_ENVS"], preprocess_obs=False)
        test_env = CTRolloutManager(log_test_env, batch_size=config["NUM_TEST_EPISODES"], preprocess_obs=False) # batched env for testing (has different batch size), chooses fixed team compositions on reset (may be different from train set)
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in log_train_env.agents+['__all__']}
        # end INIT ENV
        # -------------------- --------------------
        # INIT EXPERT_BUFFER 

        # randomly sample 1 trajectory to learn the structure, needed to init flashbax buffers
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3) # use a dummy rng here
            key_a = jax.random.split(key_a, log_train_env.num_agents)
            actions = {agent: wrapped_env.batch_sample(key_a[i], agent) for i, agent in enumerate(log_train_env.agents)}
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
            transition = Transition(obs, actions, rewards, dones, infos)
            viz_env_state = env_state.env_state
            return env_state, (transition, viz_env_state)

        _, (sample_traj, sample_viz_env_states) = jax.lax.scan(
            _env_sample_step, env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(lambda x: x[:, 0], sample_traj) # remove the NUM_ENV dim
        expert_buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=int(config['EXPERT_BUFFER_SIZE']//config['NUM_ENVS']),
            min_length_time_axis=config['UPDATE_BATCH_SIZE'],
            sample_batch_size=config['UPDATE_BATCH_SIZE'],
            add_batch_size=config['NUM_ENVS'],
            sample_sequence_length=1,
            period=1,
        )
        expert_buffer_state = expert_buffer.init(sample_traj_unbatched) 

        def collect_trajectory(runner_state, unused):
            """
            Add 1 trajectory to the expert_buffer, and update the runner_state from the previous step accordingly. Compatible with jax.lax.scan.
            """
            traj_count, expert_buffer_state, env_state, init_obs, init_dones, _, _, rng = runner_state

            def _expert_env_step(step_state, unused):
                """
                Handles a single step in the env, acting according to the expert heuristic. Compatible with jax.lax.scan.

                Save Transitions to be added to replay buffer.
                """

                env_state, last_obs, last_dones, rng = step_state

                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                # select action from expert
                # add a dummy time_step dimension to the agent input
                obs_   = {a:last_obs[a] for a in log_train_env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                obs_   = jax.tree.map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree.map(lambda x: x[np.newaxis, :], last_dones)

                actions = expert_heuristic(env_state.env_state, log_train_env.agents, expert_cached_values)

                # step in env with action
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, actions, rewards, dones, infos) # last_obs is right (at last_obs, take this action)

                # update step_state for next step, return collected transition/viz_env_state to be aggregated
                step_state = (env_state, obs, dones, rng)
                viz_env_state = env_state.env_state
                return step_state, (transition, viz_env_state)

            # prepare the step state and collect the episode trajectory
            rng, _rng = jax.random.split(rng)
            step_state = (
                env_state,
                init_obs,
                init_dones,
                _rng,
            )

            # step_state is the final step_state after NUM_STEPS
            # traj_batch/viz_env_states are sequences len NUM_STEPS (from a single rollout)
            # NOTE: viz_env_states overwritten each time, this is okay for visualization purposes
            step_state, (traj_batch, viz_env_states) = jax.lax.scan(
                _expert_env_step, step_state, None, config["NUM_STEPS"]
            )

            # compute metrics for this trajectory
            final_env_state = step_state[0].env_state
            rewards = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0).mean(), traj_batch.rewards)
            
            if env_name == "MPE_simple_fire":
                fire_metrics = fire_env_metrics(final_env_state)
                metrics = {
                    "returns": rewards['__all__'].mean(),
                    "fire_success_rate": fire_metrics[0],
                    "pct_fires_put_out": fire_metrics[1],
                }
            if env_name == "MPE_simple_transport":
                info_metrics = {
                    'quota_met': jnp.max(traj_batch.infos['quota_met'], axis=0).mean(),
                    'makespan': jnp.min(traj_batch.infos['makespan'], axis=0).mean()
                }
                metrics = {
                    "returns": rewards['__all__'].mean(),
                    **info_metrics,
                }

            # update the expert_buffer state to include this traj
            expert_buffer_traj_batch = jax.tree_util.tree_map(
                lambda x:jnp.swapaxes(x, 0, 1)[:, np.newaxis], # put the batch dim first and add a dummy sequence dim
                traj_batch
            ) # (num_envs, 1, time_steps, ...)
            expert_buffer_state = expert_buffer.add(expert_buffer_state, expert_buffer_traj_batch)

            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in log_train_env.agents+['__all__']}

            # return updated runner_state for next iteration
            runner_state = (
                # updated for next iter
                traj_count+1,
                expert_buffer_state,
                # from reset env
                env_state,
                init_obs,
                init_dones,
                # overwritten each time
                metrics,
                viz_env_states,
                # to branch off of later 
                rng,
            )

            return runner_state, None

        # put INIT_EXPERT_TRAJ trajectories into buffer
        rng, _rng = jax.random.split(rng)
        traj_count = 0
        if env_name == "MPE_simple_fire":
            sample_metrics = {
                "returns": 0,
                "fire_success_rate": 0,
                "pct_fires_put_out": 0,
            }
        if env_name == "MPE_simple_transport":
            sample_metrics = {
                "returns": 0,
                "quota_met": 0,
                "makespan": 0
            }
        expert_runner_state = (
            # init state before lax.scan
            traj_count,
            expert_buffer_state,
            # from reset env
            env_state,
            init_obs,
            init_dones,
            # overwritten each time
            sample_metrics,
            sample_viz_env_states,
            # to branch off of later
            _rng,
        )
        expert_runner_state, _ = jax.lax.scan(
            collect_trajectory, expert_runner_state, None, config["INIT_EXPERT_TRAJ"]
        )
        # update expert buffer state to final returned buffer from collected trajectories
        expert_buffer_state = expert_runner_state[1]
        # also update traj_count
        traj_count += expert_runner_state[0]

        # end INIT EXPERT_BUFFER 
        # -------------------- --------------------
        # LOG EXPERT STATS

        # need to wrap in callback for wandb.log to work w/ jit'd func
        def io_callback(metrics):
            wandb.log({k:v.mean() for k, v in metrics.items()})
        expert_metrics = expert_runner_state[-3]
        expert_metrics = {f"expert/{k}": v for k, v in expert_metrics.items()} # add prefix
        jax.debug.callback(io_callback, expert_metrics)
        # LOG EXPERT STATS
        # -------------------- --------------------
        # TRAIN POLICY

        # init policy
        if not config["AGENT_RECURRENT"]:
            if not config["AGENT_HYPERAWARE"]:
                agent = AgentMLP(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])
            else:
                agent = AgentHyperMLP(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'], hypernet_kwargs=config["AGENT_HYPERNET_KWARGS"], dim_capabilities=log_train_env.dim_capabilities)
        else: 
            if not config["AGENT_HYPERAWARE"]:
                agent = AgentRNN(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])
            else:
                agent = AgentHyperRNN(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'], hypernet_kwargs=config["AGENT_HYPERNET_KWARGS"], dim_capabilities=log_train_env.dim_capabilities)

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
                jnp.zeros((len(wrapped_env.training_agents), 1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
                jnp.zeros((len(wrapped_env.training_agents), 1, 1)) # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(wrapped_env.training_agents),  1) # (n_agents, batch_size, hidden_dim)
            rngs = jax.random.split(_rng, len(wrapped_env.training_agents)) # a random init for each agent
            agent_params = jax.vmap(agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)

        # log agent param count
        agent_param_count = sum(x.size for x in jax.tree_util.tree_leaves(agent_params))
        wandb.log({"policy/agent_param_count": agent_param_count})
        print("-" * 10)
        print("DETAILED AGENT PARAM COUNT:")
        for name, param in jax.tree_util.tree_flatten_with_path(agent_params)[0]:
            print(f"{name}: {param.shape}")
        print("-" * 10)

        # define batched computation of policy
        if config["PARAMETERS_SHARING"]:
            def homogeneous_pass(params, hidden_state, obs, dones):
                # concatenate agents and parallel envs to process them in one batch
                agents, flatten_agents_obs = zip(*obs.items())
                original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
                batched_input = (
                    jnp.concatenate(flatten_agents_obs, axis=1), # (time_step, n_agents*n_envs, obs_size)
                    jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
                )
                hidden_state, q_vals = agent.apply(params, hidden_state, batched_input)
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

        # init training params
        def linear_schedule(count):
            frac = 1.0 - (count / config["DAGGER_ITERATIONS"])
            return config["LR"] * frac
        lr = linear_schedule if config.get('LR_LINEAR_DECAY', False) else config['LR']
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(learning_rate=lr, eps=config['EPS_ADAM'], weight_decay=config['WEIGHT_DECAY_ADAM']),
        )
        network_params = frozen_dict.freeze({'agent':agent_params})
        train_state = TrainState.create(
            apply_fn=None,
            params=network_params,
            tx=tx,
        )

        # define DAGGER dataset expansion
        def collect_dagger_trajectory(runner_state, unused):
            """
            Collect a trajectory with DAgger's mixture of expert/policy, but save the expert's actions at each state. (_dagger_env_step)

            Then add new samples to buffer.
            """
            traj_count, expert_buffer_state, env_state, init_obs, init_dones, rng = runner_state

            def _dagger_env_step(step_state, unused):
                """
                Define a policy which samples from expert with prob BETA, else from current learned policy. Sample traj from env with this policy.

                Then, re-query expert on each state of traj to get new data for expert_buffer.
                """
                env_state, last_obs, last_dones, last_h, rng = step_state
                rng, key_a, key_s, key_dagger = jax.random.split(rng, 4)

                # Get actions from both learned policy and expert
                obs_ = {a: last_obs[a] for a in log_train_env.agents}
                obs_ = jax.tree.map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree.map(lambda x: x[np.newaxis, :], last_dones)
                h_state, policy_action_logits = homogeneous_pass(train_state.params['agent'], last_h, obs_, dones_)
                policy_actions = jax.tree_util.tree_map(lambda q, valid_idx: jnp.argmax(q.squeeze(0)[..., valid_idx], axis=-1), policy_action_logits, wrapped_env.valid_actions)

                expert_actions = expert_heuristic(env_state.env_state, log_train_env.agents, expert_cached_values)

                # pick expert actions with probability beta, else choose learned policy
                beta = config['DAGGER_BETA']  # Probability of using expert action
                use_expert = jax.random.bernoulli(key_dagger, p=beta, shape=(config["NUM_ENVS"],))
                actions = jax.tree_util.tree_map(
                    lambda exp_a, pol_a: jnp.where(use_expert[:, None], exp_a, pol_a),
                    expert_actions, policy_actions
                )

                # Step environment with chosen action
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                
                # Store expert action for learning
                transition = Transition(last_obs, expert_actions, rewards, dones, infos)

                step_state = (env_state, obs, dones, h_state, rng)
                viz_env_state = env_state.env_state
                return step_state, (transition, viz_env_state)

            # Run DAgger collection
            rng, _rng = jax.random.split(rng)
            step_state = (env_state, init_obs, init_dones, ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(wrapped_env.training_agents)*config["NUM_ENVS"]), _rng)
            step_state, (traj_batch, viz_env_states) = jax.lax.scan(
                _dagger_env_step, step_state, None, config["NUM_STEPS"]
            )

            # update the expert_buffer state to include this traj
            expert_buffer_traj_batch = jax.tree_util.tree_map(
                lambda x:jnp.swapaxes(x, 0, 1)[:, np.newaxis], # put the batch dim first and add a dummy sequence dim
                traj_batch
            ) # (num_envs, 1, time_steps, ...)
            expert_buffer_state = expert_buffer.add(expert_buffer_state, expert_buffer_traj_batch)

            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in log_train_env.agents+['__all__']}

            # return updated runner_state for next iteration
            runner_state = (
                # updated for next iter
                traj_count+1,
                expert_buffer_state,
                # from reset env
                env_state,
                init_obs,
                init_dones,
                # to branch off of later 
                rng,
            )

            return runner_state, None

        # define single policy update step
        def _update_step(policy_runner_state, unused):
            """
            Perform one batched update to policy params. Collect & log metrics after every update.

            NOTE: batched sample is over num_traj, as the RNN needs to sample
            successive states to update the hidden_state correctly
            """
            # unpack runner state
            total_updates, train_state, expert_buffer_state, policy_metrics, policy_viz_env_states, rng = policy_runner_state

            # define loss
            def _loss_fn(params, init_hstate, learn_traj):
                """
                Compute cross-entropy loss between learn_traj and policy output (defined by params).

                NOTE: the policy architectures return "q_vals" for consistency
                with QMIX training. However, here I abuse "q_vals" to actually
                be action logits. In the discrete case there is no difference.
                """
                obs = {a:learn_traj.obs[a] for a in wrapped_env.training_agents} # ensure to not pass the global state (obs["__all__"]) to the network
                _, policy_action_logits = homogeneous_pass(params['agent'], init_hstate, obs, learn_traj.dones) # [TS, n_envs, act_dim]
                # NOTE: I don't think the hstate is being passed fwd in time correctly here, but this is what QMIX has...?

                # then get expert actions at t
                # [TS, n_envs] (act dim squeezed out as these are indices)
                expert_actions = {a:learn_traj.actions[a] for a in wrapped_env.training_agents} # ensure to not pass the global state (obs["__all__"]) to the network

                # compute cross-entropy loss over policy actions vs expert action labels 
                # preserving TS/batch dims is fine!
                if config["PARAMETERS_SHARING"]:
                    # if shared-param, merge all agents
                    all_policy_logits = jnp.stack(list(policy_action_logits.values())) # [n_agents, TS, n_envs, act_dim]
                    all_expert_actions = jnp.stack(list(expert_actions.values())) # [n_agents, TS, n_envs]
                    # convert expert actions to one-hot
                    all_expert_logits = jax.nn.one_hot(all_expert_actions, all_policy_logits.shape[-1])

                    # then compute MSE loss between policy and expert logits
                    # NOTE: I thought this should be softmax_cross_entropy()? but it seems like robomimic BC baseline uses MSE
                    loss = jnp.mean((all_policy_logits - jax.lax.stop_gradient(all_expert_logits))**2)
                    return loss
                else:
                    exit("NON-HOMOGENEOUS POLICY LOSS NOT IMPLEMENTED")

            # sample a batch of trajectory from the expert_buffer to learn from
            rng, _rng = jax.random.split(rng)
            learn_traj = expert_buffer.sample(expert_buffer_state, _rng).experience # (batch_size, 1, max_time_steps, ...)
            learn_traj = jax.tree.map(
                lambda x: jnp.swapaxes(x[:, 0], 0, 1), # remove the dummy sequence dim (1) and swap batch and temporal dims
                learn_traj
            ) # (max_time_steps, batch_size, ...)
            if config["PARAMETERS_SHARING"]:
                init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(wrapped_env.training_agents)*config["UPDATE_BATCH_SIZE"]) # (n_agents*batch_size, hs_size)
            else:
                init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(wrapped_env.training_agents), config["UPDATE_BATCH_SIZE"]) # (n_agents, batch_size, hs_size)

            # compute loss and optimize grad (over full traj)
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            loss, grads = grad_fn(train_state.params, init_hs, learn_traj)
            train_state = train_state.apply_gradients(grads=grads)

            # get updated metrics
            # NOTE: could also only update metrics on an interval, see config["TEST_INTERVAL"] in QMIX
            rng, _rng = jax.random.split(rng)
            policy_metrics, policy_viz_env_states = test_policy(_rng, train_state.params["agent"])

            # add prefix to policy_metrics, log metrics + other info
            logging_metrics = {f"policy/{k}": v for k, v in policy_metrics.items()}
            logging_metrics["policy/loss"] = loss
            logging_metrics["policy/updates"] = total_updates
            jax.debug.callback(io_callback, logging_metrics)

            # update runner state for next update
            policy_runner_state = (
                # updated for next iter
                total_updates + 1,
                train_state,
                # collected from expert
                expert_buffer_state,
                # will be discarded, but needs to be in runner state so only latest saved
                policy_metrics,
                policy_viz_env_states,
                # to branch from later
                rng,
            )
            return policy_runner_state, None

        # create test routine for learned policy
        # (this is called within update_step, but must be defined here for init)
        def test_policy(rng, policy_params):
            """
            Get policy metrics/viz by deploying policy in the env.

            (See get_greedy_metrics() in QMIX)
            """
            def _policy_env_step(step_state, unused):
                """
                Handles a single step in the env, acting according to the policy. Compatible with jax.lax.scan.
                """
                policy_params, env_state, last_obs, last_dones, last_h, rng = step_state

                # prepare rngs for actions and step
                rng, key_s = jax.random.split(rng)

                # select action from policy
                obs_   = {a:last_obs[a] for a in test_env.training_agents} # ensure to not pass the global state (obs["__all__"]) to the network
                obs_   = jax.tree.map(lambda x: x[np.newaxis, :], obs_) # add a dummy time_step dimension to the agent input
                dones_ = jax.tree.map(lambda x: x[np.newaxis, :], last_dones) # add a dummy time_step dimension to the agent input
                h_state, policy_action_logits = homogeneous_pass(policy_params, last_h, obs_, dones_)
                actions = jax.tree_util.tree_map(lambda q, valid_idx: jnp.argmax(q.squeeze(0)[..., valid_idx], axis=-1), policy_action_logits, test_env.valid_actions) # greedy-select most likely action

                # step in env with action
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                transition = TransitionHstate(last_obs, actions, rewards, dones, infos, h_state) # last_obs is right (at last_obs, take this action)

                # update step_state for next step, return collected transition/viz_env_state to be aggregated
                step_state = (policy_params, env_state, obs, dones, h_state, rng)
                viz_env_state = env_state.env_state
                return step_state, (transition, viz_env_state)

            # run policy in env
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in test_env.training_agents+['__all__']}
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(test_env.training_agents)*config["NUM_TEST_EPISODES"]) # (n_agents*NUM_TEST_EPISODES, hs_size)
            policy_step_state = (policy_params, env_state, init_obs, init_dones, init_hs, rng)
            policy_step_state, (policy_traj_batch, policy_viz_env_states) = jax.lax.scan(
                _policy_env_step, policy_step_state, None, config["NUM_STEPS"]
            )

            # get snd, NOTE: dim_c multiplier is currently hardcoded since it works for both fire and transport 
            snd_value = snd(
                rollouts=policy_traj_batch.obs,
                hiddens=policy_traj_batch.hstate,
                dim_c=len(test_env.training_agents)*2,
                params=policy_step_state[0],
                alg='qmix',
                agent=agent
            )
            
            # compute metrics for this trajectory
            final_env_state = policy_step_state[1].env_state
            rewards = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0).mean(), policy_traj_batch.rewards)
            if env_name == "MPE_simple_fire":
                fire_metrics = fire_env_metrics(final_env_state)
                policy_metrics = {
                    "returns": rewards['__all__'].mean(),
                    "snd": snd_value,
                    "fire_success_rate": fire_metrics[0],
                    "pct_fires_put_out": fire_metrics[1],
                }
            if env_name == "MPE_simple_transport":
                info_metrics = {
                    'quota_met': jnp.max(policy_traj_batch.infos['quota_met'], axis=0),
                    'makespan': jnp.min(policy_traj_batch.infos['makespan'], axis=0)
                }
                policy_metrics = {
                    "returns": rewards['__all__'].mean(),
                    "snd": snd_value,
                    **info_metrics,
                }

            return policy_metrics, policy_viz_env_states

        jax.debug.print("Starting main loop of DAgger...")
        # DAgger main loop
        total_updates = jnp.array(0)
        for dagger_iter in range(config['DAGGER_ITERATIONS']):
            # Collect new data, add it to buffer
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in log_train_env.agents+['__all__']}
            dagger_runner_state = (
                # init runner state from previous dagger_iter
                traj_count,
                expert_buffer_state,
                # from reset env
                env_state,
                init_obs,
                init_dones,
                # to branch off of later
                _rng,
            )
            dagger_runner_state, _ = jax.lax.scan(
                collect_dagger_trajectory, dagger_runner_state, None, config["DAGGER_TRAJECTORIES_PER_ITER"]
            )
            # update for next outer loop
            traj_count = expert_runner_state[0]
            expert_buffer_state = dagger_runner_state[1]

            # Update policy from buffer & test
            rng, _rng = jax.random.split(rng)
            policy_metrics, policy_viz_env_states = test_policy(_rng, train_state.params["agent"]) # init metrics/viz_env_states for lax.scan
            rng, _rng = jax.random.split(rng)
            policy_runner_state = (
                # init
                total_updates,
                train_state,
                # fixed
                expert_buffer_state,
                # overwritten each time
                policy_metrics,
                policy_viz_env_states,
                # to branch from
                _rng,
            )
            policy_runner_state, _ = jax.lax.scan(
                _update_step, policy_runner_state, None, config["DAGGER_UPDATES_PER_ITER"]
            )
            # update for next outer loop
            total_updates = policy_runner_state[0]
            train_state = policy_runner_state[1]

            # Update beta
            if config["DAGGER_BETA_LINEAR_DECAY"]:
                frac = 1.0 - (dagger_iter / config["DAGGER_ITERATIONS"])
                config['DAGGER_BETA'] = config['DAGGER_BETA'] * frac

        # end TRAIN POLICY
        # -------------------- --------------------

        # finally, return info aggregated across all trajectories
        return {'expert_runner_state': expert_runner_state, 'expert_metrics': expert_metrics, 'expert_viz_env_states': expert_runner_state[-2], 'policy_runner_state': policy_runner_state, 'policy_metrics': policy_metrics, "policy_viz_env_states": policy_runner_state[-2],}

    return train

def visualize_states(save_dir, alg_name, viz_test_env, config, viz_env_states, prefix=""):
    """
    Build a list of states manually from vectorized seq returned by make_train() for desired seeds/envs.

    Then build MPEVisualizer and log to Wandb.
    """
    env_name = config["env"]["ENV_NAME"]
    for seed in range(config["NUM_SEEDS"]):
        for env in range(config["VIZ_NUM_ENVS"]):
            state_seq = []
            for i in range(config["alg"]["NUM_STEPS"]):
                if env_name == "MPE_simple_fire":
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
                if env_name == "MPE_simple_transport":
                    this_step_state = State(
                        p_pos=viz_env_states.p_pos[seed, i, env, ...],
                        p_vel=viz_env_states.p_vel[seed, i, env, ...],
                        c=viz_env_states.c[seed, i, env, ...],
                        accel=viz_env_states.accel[seed, i, env, ...],
                        rad=viz_env_states.rad[seed, i, env, ...],
                        done=viz_env_states.done[seed, i, env, ...],
                        capacity=viz_env_states.capacity[seed, i, env, ...],
                        site_quota=viz_env_states.site_quota[seed, i, env, ...],
                        step=i,
                    )
                state_seq.append(this_step_state)

            # save visualization to GIF for wandb display
            visualizer = MPEVisualizer(viz_test_env, state_seq, env_name=config["env"]["ENV_NAME"])
            video_fpath = f'{save_dir}/{alg_name}-seed-{seed}-rollout.gif'
            visualizer.animate(video_fpath)
            wandb.log({f"{prefix}/env-{env}-seed-{seed}-rollout": wandb.Video(video_fpath)})


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)

    print('Config:\n', OmegaConf.to_yaml(config))

    env_name = config["env"]["ENV_NAME"]
    alg_name = "DAgger"
    
    train_env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
    log_train_env = LogWrapper(train_env)
    viz_test_env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'], test_env_flag=True)
    log_test_env = LogWrapper(viz_test_env)

    config["alg"]["NUM_STEPS"] = config["alg"].get("NUM_STEPS", train_env.max_steps) # default steps defined by the env
    
    hyper_tag = "hyper" if config["alg"]["AGENT_HYPERAWARE"] else "normal"
    recurrent_tag = "RNN" if config["alg"]["AGENT_RECURRENT"] else "MLP"
    aware_tag = "aware" if config["env"]["ENV_KWARGS"]["capability_aware"] else "unaware"

    wandb_tags = [
        alg_name,
        "imitation learning",
        env_name,
        hyper_tag,
        recurrent_tag,
        aware_tag,
        f"jax_{jax.__version__}",
    ]

    # add tags added via CLI
    # e.g. ++tag=special-extra-tag
    if 'tag' in config:
        wandb_tags.append(config['tag'])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=wandb_tags,
        name=f'{alg_name} / {hyper_tag} {recurrent_tag} {aware_tag} / {env_name}',
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    # pick the correct expert heuristic, based on env
    expert_heuristic = None
    expert_cached_values = None
    env_name = config['env']['ENV_NAME']
    if env_name == "MPE_simple_fire":
        expert_heuristic = expert_heuristic_simple_fire

        # find all partitions of a set of N agents into k fires, for use in heuristic
        # pad with -1 until each partition is length N, in order to use JIT-compiled func
        # (later when computing rew, we'll ensure -1 -> 0 reward)
        N = log_train_env.num_agents
        k = log_train_env.num_landmarks

        lst = list(range(N))
        all_set_partitions = [part for k in range(1, len(lst) + 1) for part in mit.set_partitions(lst, k)]
        valid_set_partitions = []
        for part in all_set_partitions:
            if len(part) == k:
                fixed_len_part = [jnp.pad(jnp.array(p), (0, N-len(p)), constant_values=(-1,)) for p in part]
                valid_set_partitions.append(fixed_len_part)
        valid_set_partitions = jnp.array(valid_set_partitions)
        expert_cached_values = {"valid_set_partitions": valid_set_partitions, "num_landmarks": k}

    elif env_name == "MPE_simple_transport":
        expert_heuristic = expert_heuristic_material_transport
        expert_cached_values = {}

    # collect one full expert_buffer, then train a policy on that expert_buffer (for each seed)
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config["alg"], log_train_env, log_test_env, expert_heuristic, expert_cached_values, env_name)))
    outs = jax.block_until_ready(train_vjit(rngs))

    expert_traj_count = outs["expert_runner_state"][0]
    wandb.log({"expert/traj_count": jnp.sum(expert_traj_count)})

    if config["VISUALIZE_FINAL_POLICY"]:
        save_dir = os.path.join(config['SAVE_PATH'], env_name)
        os.makedirs(save_dir, exist_ok=True)

        # visualize both policy and expert
        policy_viz_env_states = outs["policy_viz_env_states"]
        visualize_states(save_dir, "LEARNED_POLICY", viz_test_env, config, policy_viz_env_states, prefix="policy")

        expert_viz_env_states = outs["expert_viz_env_states"]
        visualize_states(save_dir, "EXPERT_HEURISTIC", viz_test_env, config, expert_viz_env_states, prefix="expert")

    # force multiruns to finish correctly
    wandb.finish()

if __name__ == "__main__":
    main()
