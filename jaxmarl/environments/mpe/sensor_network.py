import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, List
from functools import partial
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from gymnax.environments.spaces import Box


class SensorNetworkMPE(SimpleMPE):
    def __init__(
        self,
        sensing_rads: List[float],
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
        self.dim_capabilities = num_agents * self.num_capabilities

        observation_spaces = {
            i:Box(-jnp.inf, jnp.inf, (4+(num_agents-1)*2+self.dim_capabilities,)) 
            for i in agents
        }

        colour = [AGENT_COLOUR] * num_agents

        # Env specific parameters
        # TODO: implement random sampling, may have to modify state again...
        self.sensing_rads = jnp.array(sensing_rads)
        assert (len(sensing_rads) == num_agents), f"len(sensing_rads) {len(sensing_rads)} does not match number of agents {num_agents}!"

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
            max_steps=50, # 25 default
            **kwargs,
        )

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        @partial(jax.vmap, in_axes=(0))
        def _common_stats(aidx: int):
            """Values needed in all observations"""

            # Zero out unseen agents with other_mask
            other_pos = state.p_pos[: self.num_agents] - state.p_pos[aidx]

            # use jnp.roll to remove ego agent from other_pos and other_vel arrays
            other_pos = jnp.roll(other_pos, shift=self.num_agents - aidx - 1, axis=0)[
                : self.num_agents - 1
            ]
            comm = jnp.roll(
                state.c[: self.num_agents], shift=self.num_agents - aidx - 1, axis=0
            )[: self.num_agents - 1]

            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)
            comm = jnp.roll(comm, shift=aidx, axis=0)

            return other_pos, comm

        other_pos, comm = _common_stats(self.agent_range)

        def _obs(aidx: int):
            original_obs = [
                state.p_vel[aidx].flatten(),  # 2
                state.p_pos[aidx].flatten(),  # 2
                other_pos[aidx].flatten(),  # N-1, 2
            ]

            # add full team's capabilities to obs
            all_agent_cap = self.sensing_rads.reshape(self.num_capabilities, self.num_agents).ravel(order='F')
            # roll s.t. ego agent appears first, e.g. [ego_agent, other agents...]
            ego_capabilities = [jnp.roll(all_agent_cap, -aidx*self.num_capabilities)]

            # zero-out capabilities for non-capability-aware baselines
            if not self.capability_aware:
                ego_capabilities = [jnp.zeros((self.dim_capabilities))]

            return jnp.concatenate(
                original_obs + ego_capabilities
            )

        obs = {a: _obs(i) for i, a in enumerate(self.agents)}
        return obs

    def rewards(self, state: State) -> Dict[str, float]:
        # vectorized version of: for i in range(self.num_agents) for j in range(self.num_agents) f(i, j)
        agent_i = jnp.repeat(jnp.arange(self.num_agents), self.num_agents, axis=0)
        agent_j = jnp.tile(jnp.arange(self.num_agents), self.num_agents)

        # approximation for connectedness: tally all robot-robot sensor connections
        # > is better
        def sensing_overlap(i: int, j: int):
            sensing_rads_sum = self.sensing_rads[i] + self.sensing_rads[j]
            delta_pos = state.p_pos[i] - state.p_pos[j]
            dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
            return (dist < sensing_rads_sum) & (i != j)

        all_overlap = jax.vmap(sensing_overlap)(agent_i, agent_j)
        overlap_rew = jnp.sum(all_overlap)

        # approximation for covered area: sum all robot-robot distances
        # > is better
        def robot_robot_dist(i: int, j: int):
            delta_pos = state.p_pos[i] - state.p_pos[j]
            dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
            return dist

        all_dists = jax.vmap(robot_robot_dist)(agent_i, agent_j)
        coverage_rew = jnp.sum(all_dists)

        # jax.debug.print("overlap rew {} coverage_rew {}", overlap_rew, coverage_rew)
        global_rew = 0.02 * overlap_rew + 0.0005 * coverage_rew

        # all agents rewarded w/ global team rew
        rew = {
            a: global_rew
            for i, a in enumerate(self.agents)
        }
        return rew
