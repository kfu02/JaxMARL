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
        num_agents=3,
        num_landmarks=2,
        action_type=DISCRETE_ACT,
        capability_aware=True,
        num_capabilities=2,
        **kwargs,
    ):
        agents = ["agent_{}".format(i) for i in range(num_agents)]
        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        self.capability_aware = capability_aware
        self.num_capabilities = num_capabilities

        observation_spaces = {
            i:Box(-jnp.inf, jnp.inf, (num_agents*(2+2+num_capabilities) + 2*num_landmarks,)) 
            for i in agents
        }

        self.colour = [AGENT_COLOUR] * num_agents + [OBS_COLOUR] * num_landmarks

        # Env specific parameters
        self.test_team = kwargs["test_team"] if "test_team" in kwargs else None
        # self.agent_spawns = jnp.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        self.agent_spawns = jnp.array([[0.6, 0.0], [0.0, 0.0], [-0.6, 0.0]])
        self.landmark_spawns = jnp.array([[0.0, -0.5], [0.0, 0.5]])
        # Parameters
        # NOTE: rad now passed in, necessity for SimpleSpread modifications
        collide = jnp.concatenate(
            [jnp.full((num_agents+num_landmarks), False)]
        )

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            observation_spaces=observation_spaces,
            dim_c=0, # no comm
            colour=self.colour,
            # NOTE: rad now passed in, necessity for SimpleSpread modifications
            # rad=rad,
            collide=collide,
            **kwargs,
        )

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

            other_cap = jnp.stack([
                state.accel.flatten(), state.rad[:self.num_agents].flatten(), # landmark rad is included in state.rad
            ], axis=-1)
            # jax.debug.print("other cap {} accel {} rad {}", other_cap, state.accel, state.rad[:self.num_agents])

            ego_cap = other_cap[aidx, :]
            # roll to remove ego agent
            other_cap = jnp.roll(other_cap, shift=self.num_agents - aidx - 1, axis=0)[:self.num_agents-1, :]

            # zero-out capabilities for non-capability-aware baselines
            if not self.capability_aware:
                other_cap = jnp.full(other_cap.shape, -1e3)
                ego_cap = jnp.full(ego_cap.shape, -1e3)

            # give agents the pos of all landmarks
            fire_obs = self.landmark_spawns

            obs = jnp.concatenate([
                # ego agent attributes, then, teammate attr
                # for each of pos/vel/cap, in same order, in matching order
                ego_pos.flatten(),  # 2
                other_pos.flatten(),  # N-1, 2
                ego_vel.flatten(),  # 2
                other_vel.flatten(),  # N-1, 2
                ego_cap.flatten(),  # n_cap
                other_cap.flatten(),  # N-1, n_cap
                fire_obs.flatten(), # 2, 2
            ])
            # jax.debug.print("pos {} , vel {}, cap {}, obs {}", state.p_pos.flatten(), state.p_vel.flatten(), state.sensing_rads.flatten(), obs)

            return obs

        obs = {a: _obs(i) for i, a in enumerate(self.agents)}
        return obs

    def rewards(self, state: State, key) -> Dict[str, float]:
        """
        Goal is to put out fires as quickly as possible.

        Fires are put out when the sum of all robot firefighting capacity on the fire is greater than the strength of the fire.
        """
        global_rew = 0

        # reward team when fires are being put out, penalize otherwise
        def _agent_in_range(agent_i: int, landmark_pos, landmark_rad):
            delta_pos = state.p_pos[agent_i] - landmark_pos
            dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
            return (dist < landmark_rad)

        agent_rads = state.rad[:self.num_agents]
        landmark_rads = state.rad[self.num_agents:]
        for i, landmark_pos in enumerate(self.landmark_spawns):
            landmark_rad = landmark_rads[i]

            # TODO could vectorize the outer loop as well
            # take the sum of all agents ff on this landmark
            agents_on_landmark = jax.vmap(_agent_in_range, in_axes=[0, None, None])(self.agent_range, landmark_pos, landmark_rad)
            firefighting_level = jnp.where(agents_on_landmark, agent_rads, 0)

            # if ff is enough, reward the team, otherwise penalize it
            enough_firefight = jnp.sum(firefighting_level) >= landmark_rads[i]
            global_rew = jnp.where(enough_firefight, global_rew + 1, global_rew - 1)

        # reward each agent for getting closer to one of the two landmarks
        def _dist_to_landmarks(agent_pos):
            dist_to_landmarks = jnp.linalg.norm(agent_pos - self.landmark_spawns, axis=1)
            return dist_to_landmarks

        def _agent_rew(agent_i):
            agent_pos = state.p_pos[agent_i]
            dists = _dist_to_landmarks(agent_pos)
            return -jnp.min(dists)

        rew = {
            a: global_rew + _agent_rew(i)
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
                self.agent_spawns,
                self.landmark_spawns,
            ]
        )

        # randomly sample N_agents' capabilities from the possible agent pool (hence w/out replacement)
        selected_agents = jax.random.choice(key_c, self.num_agents, shape=(self.num_agents,), replace=False)

        agent_rads = self.agent_rads[selected_agents]
        agent_accels = self.agent_accels[selected_agents]

        # unless a test distribution is provided and this is a test_env
        if self.test_env_flag and self.test_team is not None:
            agent_rads = jnp.array(self.test_team["agent_rads"])
            agent_accels = jnp.array(self.test_team["agent_accels"])

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            sensing_rads=None,
            c=jnp.zeros((self.num_agents, self.dim_c)),
            accel=agent_accels,
            rad=jnp.concatenate(
                # NOTE: here, must define landmark rad as well, by default landmarks are 0.05
                [agent_rads, jnp.array([0.19, 0.29])]
            ),
            done=jnp.full((self.num_agents), False),
            step=0,
        )

        return self.get_obs(state), state
