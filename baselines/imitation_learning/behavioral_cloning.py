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

# TODO: make a new shared file to import identical policy architectures from the QMIX.py file

import more_itertools as mit

class Transition(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    infos: dict

@partial(jax.jit, static_argnums=[0, 1])
def expert_heuristic_simple_fire(valid_set_partitions, num_landmarks, obs_dict, valid_actions):
    """
    Expert policy to gather samples from.
    pi(obs) -> action for each agent/obs
    """
    num_agents = len(obs_dict)

    # TODO: need to move the task allocation step here
    # then pass into "one_agent_one_env" along with the right agent indices
    # in order for agents to agree on strategy (otherwise tied strategies render this meaningless)

    def one_agent_one_env(obs):
        # For one agent/env, the obs will always take the form
        #   ego_pos, other_pos, ego_vel, other_vel, landmark_pos, landmark_rad, ego_cap, other_cap

        n_cap = 2
        cap = obs[0, -num_agents*n_cap:]
        # concat a 0 to the end of the agent_rad
        # this is necessary to handle the padded -1s in valid_set_partitions, and should not mess up the true non-padded indices
        # (e.g. agent_rad = [1, 2, 3, 0], allocation [0,1], [2,-1] -> [1+2, 3+0], allocation [1,-1], [0,2] -> [2+0, 1+3]
        agent_rad = jnp.concatenate([cap[1::2], jnp.zeros(shape=(1,))])
        landmark_rad = obs[0, -(num_landmarks+num_agents*n_cap): -num_agents*n_cap]
        landmark_pos = jnp.reshape(obs[0, -(num_landmarks*2+num_landmarks+num_agents*n_cap):-(num_landmarks+num_agents*n_cap)], (-1, 2))
        ego_pos = obs[0, :2]

        all_rew = jnp.zeros(shape=(len(valid_set_partitions),))-10
        # for each possible allocation of agents to fires
        # (set partitions is a list of lists, e.g. [[0,1],[2]], which we can interpret as 0,1 => fire 0, 2 => fire 1)
        for a_i, allocation in enumerate(valid_set_partitions):
            # compute the reward if we allocated according to this split
            reward = 0
            for i in range(num_landmarks):
                fire_ff = landmark_rad[i]
                team_ff = jnp.sum(agent_rad[allocation[i]]) 
                reward = team_ff - fire_ff

            # update all_rew
            all_rew = all_rew.at[a_i].set(reward)

        # cap reward at 0 if team_ff > fire_ff
        # all_rew = jnp.where(all_rew > 0, 0, all_rew)

        # based on rew, find the best allocation
        best_index = jnp.argmax(all_rew).astype(int)
        best_allocation = valid_set_partitions[best_index]

        # then find the index of the fire that robot 0 (ego-robot) is supposed to go to
        # (size=1 bc can only be one place, based on valid_set_partitions)
        # (first_index of that to get the row = fire_index)
        my_fire_index = jnp.argwhere(best_allocation == 0, size=1)[0][0]
        my_fire_pos = landmark_pos[my_fire_index]

        unit_vectors = jnp.array([[0,0], [-1,0], [+1,0], [0,-1], [0,+1]])
        dir_to_fire_pos = my_fire_pos - ego_pos
        dir_to_fire_pos = dir_to_fire_pos / jnp.linalg.norm(dir_to_fire_pos)
        dot = jnp.dot(unit_vectors, dir_to_fire_pos)

        # always pick the discrete action which maximizes progress towards fire
        # (as measured by dot prod similarity of discrete action with desired heading vector)
        # return 1 action per agent per env
        best_action = jnp.argmax(dot)
        # jax.debug.print("unit_vectors {} dir_to_fire_pos {} dot {} best {}", unit_vectors, dir_to_fire_pos, dot, best_action)
        return best_action 

    actions = {}
    for agent, obs in obs_dict.items():
        # obs : 1,8,24 = TS?, N_envs, obs_dim
        # actions[agent] = jnp.zeros(shape=(obs.shape[1],), dtype=jnp.int32)
        actions[agent] = jax.vmap(one_agent_one_env, in_axes=[1])(obs)

    return actions

def make_expert_buffers(config, log_train_env):
    """
    return function collect, which collects 1 seed's worth of traj data with the expert heuristic
    """

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
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'],
            min_length_time_axis=config['BUFFER_BATCH_SIZE'],
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
            add_batch_size=config['NUM_ENVS'],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched) 

        # EPISODE STEP
        def _env_step(step_state, unused):
            """
            Handles a single step in the env, acting according to the expert heuristic. Compatible with jax.lax.scan.
            """

            env_state, last_obs, last_dones, rng = step_state

            # prepare rngs for actions and step
            rng, key_a, key_s = jax.random.split(rng, 3)

            # SELECT ACTION
            # add a dummy time_step dimension to the agent input
            obs_   = {a:last_obs[a] for a in log_train_env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
            obs_   = jax.tree.map(lambda x: x[np.newaxis, :], obs_)
            dones_ = jax.tree.map(lambda x: x[np.newaxis, :], last_dones)

            valid_actions = wrapped_env.valid_actions
            actions = expert_heuristic_simple_fire(valid_set_partitions, log_train_env.num_landmarks, obs_, valid_actions)

            # STEP ENV
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
            transition = Transition(last_obs, actions, rewards, dones, infos)

            step_state = (env_state, obs, dones, rng)
            return step_state, (transition, env_state.env_state)

        # prepare the step state and collect the episode trajectory
        rng, _rng = jax.random.split(rng)

        step_state = (
            env_state,
            init_obs,
            init_dones,
            _rng,
        )

        # x, y = jax.lax.scan(...)
        # in general iters over x, and combines the result of f(x) at each step in y
        # thus traj_batch is a batch of Transitions
        step_state, (traj_batch, viz_env_states) = jax.lax.scan(
            _env_step, step_state, None, config["NUM_STEPS"]
        )

        # BUFFER UPDATE: save the collected trajectory in the buffer
        buffer_traj_batch = jax.tree_util.tree_map(
            lambda x:jnp.swapaxes(x, 0, 1)[:, np.newaxis], # put the batch dim first and add a dummy sequence dim
            traj_batch
        ) # (num_envs, 1, time_steps, ...)
        buffer_state = buffer.add(buffer_state, buffer_traj_batch)

        # UPDATE THE VARIABLES AND RETURN
        # reset the environment
        # TODO: I don't understand the reason they reset the env / runner_state here in the original _update_step
        # rng, _rng = jax.random.split(rng)
        # init_obs, env_state = wrapped_env.batch_reset(_rng)
        # init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in log_train_env.agents+['__all__']}

        # NOTE: I think in the original JaxMARL they pass a "time_state" object
        # which allows updates to be pooled from parallel threads and update
        # here, but here I've destroyed time_state

        # TODO: how to pool buffer states?? or should I pass in one buffer?
        return buffer_state, viz_env_states

    return collect


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
    
    # hyper_tag = "hyper" if config["alg"]["AGENT_HYPERAWARE"] else "normal"
    # recurrent_tag = "RNN" if config["alg"]["AGENT_RECURRENT"] else "MLP"
    # aware_tag = "aware" if config["env"]["ENV_KWARGS"]["capability_aware"] else "unaware"

    wandb_tags = [
        alg_name.upper(),
        env_name,
        # hyper_tag,
        # recurrent_tag,
        # aware_tag,
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
        # name=f'{hyper_tag} {recurrent_tag} {aware_tag} {cap_transf_tag} / {env_name}',
        name=f'{alg_name} / {env_name}',
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_expert_buffers(config["alg"], log_train_env)))
    buffer_states, viz_env_states = jax.block_until_ready(train_vjit(rngs))
    
    if config["VISUALIZE_FINAL_POLICY"]:
        save_dir = os.path.join(config['SAVE_PATH'], env_name)
        os.makedirs(save_dir, exist_ok=True)

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

    # force multiruns to finish correctly
    wandb.finish()

if __name__ == "__main__":
    main()
