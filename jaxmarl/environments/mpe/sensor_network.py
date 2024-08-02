import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, List
from functools import partial
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from gymnax.environments.spaces import Box

def is_connected(adjacency_list):
    """
    Test if adj_list represents a fully connected graph, without BFS/DFS:
    https://math.stackexchange.com/questions/551889/checking-if-a-graph-is-fully-connected
    """
    num_agents = adjacency_list.shape[0]
    S = adjacency_list
    for n in range(num_agents):
        S = S + jnp.linalg.matrix_power(adjacency_list, n)
    return jnp.all(S != 0)


def estimate_area(centers, rads, key):
    # world limits are defined as (2,2) in the visualizer so that's what I'll go with here
    # technically MPE is unbounded
    world_limits = jnp.array([[2, 2]])

    # monte carlo method
    def sample_pt(subkey):
        # sample random point from within rect world limits
        pt = jax.random.normal(subkey, shape=(1,2)) * world_limits

        # compute dist of random point to each circle center
        delta_pos = centers - pt
        dists = jnp.sqrt(jnp.sum(jnp.square(delta_pos), axis=1))

        # subtract rads from dists
        dists_to_ctr = dists - rads

        # if any are < 0, point is within range of a circle, count as good
        # jax.debug.print("pt {} c {} r {} sampled {}", pt, centers, rads, jnp.any(dists_to_ctr < 0))
        return jnp.any(dists_to_ctr < 0)

    # sample num_iters points
    num_iters = 1000
    subkeys = jax.random.split(key, num_iters)
    within_pts = jnp.sum(jax.vmap(sample_pt)(subkeys))

    # area ~= fraction * area of bounding rectangle
    return (within_pts / num_iters) * (world_limits[0][0] * world_limits[0][1])

class SensorNetworkMPE(SimpleMPE):
    def __init__(
        self,
        num_agents=3,
        num_landmarks=0,
        action_type=DISCRETE_ACT,
        capability_aware=True,
        num_capabilities=1,
        **kwargs,
    ):
        agents = ["agent_{}".format(i) for i in range(num_agents)]

        self.capability_aware = capability_aware
        self.num_capabilities = num_capabilities

        observation_spaces = {
            i:Box(-jnp.inf, jnp.inf, (num_agents*(2+2+num_capabilities),)) 
            for i in agents
        }

        colour = [AGENT_COLOUR] * num_agents

        # Env specific parameters
        assert ("sensing_rads" in kwargs), "sensing_rads must be passed to SensorNetworkMPE!!"
        self.sensing_rads = jnp.array(kwargs["sensing_rads"])

        self.test_team = kwargs["test_team"] if "test_team" in kwargs else None

        # Parameters
        # NOTE: rad now passed in, necessity for SimpleSpread modifications
        collide = jnp.concatenate(
            [jnp.full((num_agents), True)]
        )

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=0,
            landmarks=None,
            action_type=action_type,
            observation_spaces=observation_spaces,
            dim_c=0, # no comm
            colour=colour,
            # NOTE: rad now passed in, necessity for SimpleSpread modifications
            # rad=rad,
            collide=collide,
            **kwargs,
        )

        assert self.sensing_rads.shape[0] == self.agent_accels.shape[0], "sensing_rads must have same number of agents as agent_accels!"

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        def _obs(aidx: int):
            ego_pos = state.p_pos[aidx, :]
            # use jnp.roll to remove ego agent from other_pos and other_vel arrays
            other_pos = jnp.roll(state.p_pos, shift=self.num_agents - aidx - 1, axis=0)[
                : self.num_agents - 1
            ]

            ego_vel = state.p_vel[aidx, :]
            # use jnp.roll to remove ego agent from other_vel and other_vel arrays
            other_vel = jnp.roll(state.p_vel, shift=self.num_agents - aidx - 1, axis=0)[
                : self.num_agents - 1
            ]

            other_cap = state.sensing_rads.reshape(self.num_agents, self.num_capabilities)
            ego_cap = other_cap[aidx, :]
            # roll to remove ego agent
            other_cap = jnp.roll(other_cap, shift=self.num_agents - aidx - 1, axis=0)[
                : self.num_agents - 1
            ]

            # zero-out capabilities for non-capability-aware baselines
            if not self.capability_aware:
                other_cap = jnp.zeros(other_cap.shape)
                ego_cap = jnp.zeros(ego_cap.shape)

            obs = jnp.concatenate([
                # ego agent attributes, then, teammate attr
                # for each of pos/vel/cap, in same order, in matching order
                ego_pos.flatten(),  # 2
                other_pos.flatten(),  # N-1, 2
                ego_vel.flatten(),  # 2
                other_vel.flatten(),  # N-1, 2
                ego_cap.flatten(),  # n_cap
                other_cap.flatten(),  # N-1, n_cap
            ])
            # jax.debug.print("pos {} , vel {}, cap {}, obs {}", state.p_pos.flatten(), state.p_vel.flatten(), state.sensing_rads.flatten(), obs)

            return obs

        obs = {a: _obs(i) for i, a in enumerate(self.agents)}
        return obs

    def rewards(self, state: State, key) -> Dict[str, float]:
        """
        Goal is to maximize coverage area while remaining fully connected
        Thus, reward is 0 if graph is not fully connected, else it is coverage area (approximated by robot_robot dist as true coverage area is hard to compute).
        """
        coverage_rew = estimate_area(state.p_pos, state.sensing_rads, key)

        # vectorized version of: for i in range(self.num_agents) for j in range(self.num_agents) f(i, j)
        agent_i = jnp.repeat(jnp.arange(self.num_agents), self.num_agents, axis=0)
        agent_j = jnp.tile(jnp.arange(self.num_agents), self.num_agents)

        def sensing_overlap(i: int, j: int):
            sensing_rads_sum = state.sensing_rads[i] + state.sensing_rads[j]
            delta_pos = state.p_pos[i] - state.p_pos[j]
            dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
            return (dist < sensing_rads_sum)

        adjacency_list = jax.vmap(sensing_overlap)(agent_i, agent_j).reshape(self.num_agents, self.num_agents)

        # penalty for disconnected graphs
        # TODO: add collision penalty?
        global_rew = jnp.where(is_connected(adjacency_list), coverage_rew, -1.0)

        # all agents rewarded w/ global team rew
        rew = {
            a: global_rew
            for i, a in enumerate(self.agents)
        }
        return rew

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """overriding superclass to reset capabilities"""
        # NOTE: copy-pasted from simple.py, bad practice

        key_a, key_l, key_c = jax.random.split(key, num=3)

        p_pos = jnp.concatenate(
            [
                jax.random.uniform(
                    key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0
                ),
                jax.random.uniform(
                    key_l, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0
                ),
            ]
        )

        # randomly sample N_agents' capabilities from the possible agent pool (hence w/out replacement)
        selected_agents = jax.random.choice(key_c, self.agent_rads.shape[0], shape=(self.num_agents,), replace=False)

        agent_rads = self.agent_rads[selected_agents]
        agent_accels = self.agent_accels[selected_agents]
        sensing_rads = self.sensing_rads[selected_agents]

        # unless a test distribution is provided and this is a test_env
        if self.test_env_flag and self.test_team is not None:
            agent_rads = jnp.array(self.test_team["agent_rads"])
            agent_accels = jnp.array(self.test_team["agent_accels"])
            sensing_rads = jnp.array(self.test_team["sensing_rads"])

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            sensing_rads=sensing_rads,
            c=jnp.zeros((self.num_agents, self.dim_c)),
            accel=agent_accels,
            rad=jnp.concatenate(
                # NOTE: here, must define landmark rad as well, by default landmarks are 0.05
                [agent_rads, jnp.full((self.num_landmarks), 0.05)]
            ),
            done=jnp.full((self.num_agents), False),
            step=0,
        )

        return self.get_obs(state), state
