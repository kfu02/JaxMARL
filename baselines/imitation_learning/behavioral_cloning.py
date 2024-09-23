"""
Basic behavioral cloning implementation.

NOTE: only tested on MPESimpleFire.
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

from jaxmarl.environments.mpe import MPEVisualizer
from jaxmarl.environments.mpe.simple import State

from jaxmarl.policies import ScannedRNN, AgentMLP, AgentHyperMLP, AgentRNN, AgentHyperRNN, HyperNetwork

import more_itertools as mit

class Transition(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    infos: dict

def expert_heuristic_simple_fire(valid_set_partitions, num_landmarks, obs_dict):
    """
    Expert policy to gather samples from.
    pi(obs) -> action for each agent/obs
    """
    num_agents = len(obs_dict)

    def solve_one_env(obs):
        """
        For all agents in one env, obs will take the form [n_agents, obs_dim]. Solve the env for those agents.

        Note task allocation only takes info from the first agent, as everything is fully observable from that POV.
        """

        # NOTE: this is derived from return type of get_obs() in simple_fire.py (vel is missing as it is irrelevant here)
        n_cap = 2
        # unflatten to be indexable by agent
        agent_pos = jnp.reshape(obs[0, :(num_agents*2)], (-1, 2))
        ego_pos = jnp.reshape(agent_pos[0], (1,2))
        rel_other_pos = agent_pos[1:]

        # unflatten to be indexable by fire
        rel_landmark_pos = jnp.reshape(obs[0, -(num_landmarks*2+num_landmarks+num_agents*n_cap):-(num_landmarks+num_agents*n_cap)], (-1, 2))
        landmark_rad = obs[0, -(num_landmarks+num_agents*n_cap): -num_agents*n_cap]
        cap = obs[0, -num_agents*n_cap:]

        # concat a 0 to the end of the agent_rad
        # this is necessary to handle the padded -1s in valid_set_partitions, and should not mess up the true non-padded indices
        # (e.g. agent_rad = [1, 2, 3, 0], allocation [0,1], [2,-1] -> [1+2, 3+0], allocation [1,-1], [0,2] -> [2+0, 1+3]
        agent_rad = jnp.concatenate([cap[1::2], jnp.zeros(shape=(1,))])

        # for each possible allocation of agents to fires
        # (set partitions is a list of lists, e.g. [[0,1],[2]], which we can interpret as 0,1 => fire 0, 2 => fire 1)
        all_rew = jnp.zeros(shape=(len(valid_set_partitions),))-10
        for a_i, allocation in enumerate(valid_set_partitions):
            # compute the reward if we allocated according to this split
            reward = 0
            for i in range(num_landmarks):
                fire_ff = landmark_rad[i]
                team_ff = jnp.sum(agent_rad[allocation[i]]) 
                # cap positive reward per fire at 0
                reward += jnp.where(team_ff >= fire_ff, 0, team_ff - fire_ff)

            # update all_rew
            all_rew = all_rew.at[a_i].set(reward)

        # based on rew, find a best allocation (can be ties thanks to cap)
        best_index = jnp.argmax(all_rew).astype(int)
        best_allocation = valid_set_partitions[best_index]

        # convert agent/fire pos to global coords
        global_agent_pos = jnp.concatenate([ego_pos, ego_pos+rel_other_pos]) #[num_agents, 2]
        global_fire_pos = ego_pos+rel_landmark_pos

        # jax.debug.print("obs {} global_fire_pos {}", obs, global_fire_pos)

        def best_action_for_agent(agent_i):
            my_fire = jnp.argwhere(best_allocation == agent_i, size=1)[0][0]
            
            unit_vectors = jnp.array([[0,0], [-1,0], [+1,0], [0,-1], [0,+1]])
            dir_to_fire_pos = global_fire_pos[my_fire] - global_agent_pos[agent_i]
            dir_to_fire_pos = dir_to_fire_pos / jnp.linalg.norm(dir_to_fire_pos)
            dot = jnp.dot(unit_vectors, dir_to_fire_pos)

            # always pick the discrete action which maximizes progress towards fire
            # (as measured by similarity of discrete action with desired heading vector)
            best_action = jnp.argmax(dot)
            return best_action 

        # then compute best actions for each agent based on assignment
        all_best_actions = jax.vmap(best_action_for_agent)(jnp.arange(num_agents))
        return all_best_actions

    # stack obs to be per-env
    all_obs = jnp.stack(list(obs_dict.values())) # [n_agents, ?, n_envs, obs_dim]
    all_obs = all_obs.squeeze(1) # remove extra unknown dim

    # iterate over n_envs
    all_acts = jax.vmap(solve_one_env, in_axes=[1])(all_obs) # [n_envs, n_agents] (act_dim=1, squeezed out)

    # then separate back into per-agent actions
    actions = {}
    for i, agent_name in enumerate(obs_dict.keys()):
        actions[agent_name] = all_acts[:, i]

    return actions

def make_expert_buffer(config, log_train_env):
    """
    return function collect, which collects 1 seed's worth of traj data with the expert heuristic
    and computes metrics over all data
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

    def collect(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)

        # NOTE: added preprocess_obs=False to avoid adding agent_ids to each obs
        # NOTE: this has the side effect of also removing any zero-padding to
        # standardize obs dimensions across agents, may be issue later
        wrapped_env = CTRolloutManager(log_train_env, batch_size=config["NUM_ENVS"], preprocess_obs=False)
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in log_train_env.agents+['__all__']}

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

        # INIT BUFFER (randomly sample a trajectory to learn the structure)
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
        buffer_size = config['NUM_ENVS'] * config["TRAJECTORIES_PER_ENV"]
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=buffer_size//config['NUM_ENVS'],
            min_length_time_axis=config['BUFFER_BATCH_SIZE'],
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
            add_batch_size=config['NUM_ENVS'],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched) 

        def collect_trajectory(runner_state, unused):
            """
            Add 1 trajectory to the buffer, and update the runner_state from the previous step accordingly. Compatible with jax.lax.scan.
            """
            # break runner_state tuple up
            traj_count, buffer_state, env_state, init_obs, init_dones, rng = runner_state

            def _env_step(step_state, unused):
                """
                Handles a single step in the env, acting according to the expert heuristic. Compatible with jax.lax.scan.
                """

                env_state, last_obs, last_dones, rng = step_state

                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                # select action from expert
                # add a dummy time_step dimension to the agent input
                obs_   = {a:last_obs[a] for a in log_train_env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                obs_   = jax.tree.map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree.map(lambda x: x[np.newaxis, :], last_dones)

                actions = expert_heuristic_simple_fire(valid_set_partitions, log_train_env.num_landmarks, obs_)

                # step in env with action
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, actions, rewards, dones, infos)

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
                _env_step, step_state, None, config["NUM_STEPS"]
            )

            # compute metrics for this trajectory
            final_env_state = step_state[0].env_state
            metrics = fire_env_metrics(final_env_state)

            # update the buffer state to include this traj
            buffer_traj_batch = jax.tree_util.tree_map(
                lambda x:jnp.swapaxes(x, 0, 1)[:, np.newaxis], # put the batch dim first and add a dummy sequence dim
                traj_batch
            ) # (num_envs, 1, time_steps, ...)
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in log_train_env.agents+['__all__']}

            # return updated runner_state for next iteration
            runner_state = (
                # updated for next iter
                traj_count+1,
                buffer_state,
                # from reset env
                env_state,
                init_obs,
                init_dones,
                # to branch off of later 
                rng,
            )

            return runner_state, (metrics, viz_env_states)

        # collect the number of trajectories we want from each env
        rng, _rng = jax.random.split(rng)
        traj_count = 0
        runner_state = (
            # init state before lax.scan
            traj_count,
            buffer_state,
            # from reset env
            env_state,
            init_obs,
            init_dones,
            # to branch off of later
            _rng,
        )
        runner_state, (metrics, viz_env_states) = jax.lax.scan(
            collect_trajectory, runner_state, None, config["TRAJECTORIES_PER_ENV"]
        )

        # find the mean of all metrics for this seed
        metrics = jax.tree_util.tree_map(
            lambda x: x.mean(),
            metrics
        )

        # then return info aggregated across all trajectories
        return {'runner_state': runner_state, 'metrics': metrics, "viz_env_states": viz_env_states}

    return collect

def visualize_states(save_dir, alg_name, viz_test_env, config, viz_env_states):
    """
    Build a list of states manually from vectorized seq returned by make_train() for desired seeds/envs.

    Then build MPEVisualizer and log to Wandb.
    """
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


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)

    print('Config:\n', OmegaConf.to_yaml(config))

    env_name = config["env"]["ENV_NAME"]
    alg_name = "Behavioral Cloning"
    
    train_env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
    log_train_env = LogWrapper(train_env)
    viz_test_env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'], test_env_flag=True)
    log_test_env = LogWrapper(viz_test_env)

    config["alg"]["NUM_STEPS"] = config["alg"].get("NUM_STEPS", train_env.max_steps) # default steps defined by the env
    
    hyper_tag = "hyper" if config["alg"]["AGENT_HYPERAWARE"] else "normal"
    recurrent_tag = "RNN" if config["alg"]["AGENT_RECURRENT"] else "MLP"
    aware_tag = "aware" if config["env"]["ENV_KWARGS"]["capability_aware"] else "unaware"

    wandb_tags = [
        alg_name.upper(),
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
        name=f'BC / {hyper_tag} {recurrent_tag} {aware_tag} / {env_name}',
        name=f'{alg_name} / {env_name}',
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    # collect one full buffer for each seed
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    collect_vjit = jax.jit(jax.vmap(make_expert_buffer(config["alg"], log_train_env)))
    expert_collect_output = jax.block_until_ready(collect_vjit(rngs))

    expert_runner_state = expert_collect_output["runner_state"]
    expert_traj_count, expert_buffers, _, _, _, _ = expert_runner_state

    # TODO: iterate over expert buffers (with lax.scan?) and train policy to imitate demonstrations
    # see line 337 of qmix.py

    jax.debug.print("expert traj count {}", expert_traj_count)
    expert_metrics = expert_collect_output["metrics"]

    # shape for each element = (# seeds, # traj, # steps/env, # envs, # entities, DIM)
    # thus simply take the first traj for visualization purposes
    expert_viz_env_states = expert_collect_output["viz_env_states"]
    expert_viz_env_states = jax.tree_util.tree_map(
        lambda x: x[:, 0, ...],
        expert_viz_env_states
    )

    # need to wrap in callback for wandb.log to work w/ JAX
    def io_callback(metrics, traj_count):
        print("-"* 10, metrics)
        # TODO: cleanup by making metrics a dict of str->value, then generalizing this log
        wandb.log({"expert/success_rate": wandb.Histogram(metrics[0]),
                   "expert/pct_fires_put_out": wandb.Histogram(metrics[1]),
                   "expert/traj_count": traj_count[0]})
    jax.debug.callback(io_callback, expert_metrics, expert_traj_count)
    
    if config["VISUALIZE_FINAL_POLICY"]:
        save_dir = os.path.join(config['SAVE_PATH'], env_name)
        os.makedirs(save_dir, exist_ok=True)

        visualize_states(save_dir, "EXPERT_HEURISTIC", viz_test_env, config, expert_viz_env_states)

    # force multiruns to finish correctly
    wandb.finish()

if __name__ == "__main__":
    main()
