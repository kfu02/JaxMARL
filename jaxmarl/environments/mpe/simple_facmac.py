import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from jaxmarl.environments.mpe.simple import State, SimpleMPE
from gymnax.environments.spaces import Box, Discrete
from jaxmarl.environments.mpe.default_params import *


SimpleFacmacMPE3a = lambda: SimpleFacmacMPE(num_good_agents=1, num_adversaries=3, num_landmarks=2,
                            view_radius=1.5, score_function="min")
SimpleFacmacMPE6a = lambda: SimpleFacmacMPE(num_good_agents=2, num_adversaries=6, num_landmarks=4,
                            view_radius=1.5, score_function="min")
SimpleFacmacMPE9a = lambda: SimpleFacmacMPE(num_good_agents=3, num_adversaries=9, num_landmarks=6,
                            view_radius=1.5, score_function="min")

class SimpleFacmacMPE(SimpleMPE):
    def __init__(
        self,
        num_good_agents=1,
        num_adversaries=3,
        num_landmarks=2,
        view_radius=1.5,  # set -1 to deactivate
        score_function="min",
        capability_aware=True,
        num_capabilities=2,
        **kwargs,
    ):
        dim_c = 2  # NOTE follows code rather than docs
        action_type = DISCRETE_ACT # CONTINUOUS_ACT
        view_radius = view_radius if view_radius != -1 else 999999

        self.capability_aware = capability_aware
        self.num_capabilities = num_capabilities
        self.dim_capabilities = num_adversaries * num_capabilities

        num_agents = num_good_agents + num_adversaries
        self.num_landmarks = num_landmarks
        num_entities = num_agents + self.num_landmarks

        self.num_good_agents, self.num_adversaries = num_good_agents, num_adversaries

        self.adversaries = ["adversary_{}".format(i) for i in range(num_adversaries)]
        self.good_agents = ["agent_{}".format(i) for i in range(num_good_agents)]
        agents = self.adversaries + self.good_agents

        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        colour = (
            [(255, 155, 155)] * num_adversaries # predator color
            + [(115, 255, 115)] * num_good_agents # prey color
            + [(155, 155, 155)] * num_landmarks # obstacle color
        )

        # Parameters
        # TODO: make max_speed a cap too? would require modifying simple.py, and here (as done with agent_accel/rad)
        max_speed = jnp.concatenate(
            [
                jnp.full((self.num_adversaries), 1.0),
                jnp.full((self.num_good_agents), 1.3),
                jnp.full((self.num_landmarks), 0.0),
            ]
        )
        collide = jnp.full((num_entities,), True)

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            dim_c=dim_c,
            colour=colour,
            # NOTE: modified via reset(), see below
            # rad=rad,
            # accel=accel,
            max_speed=max_speed,
            collide=collide,
            **kwargs,
        )

        # Overwrite action and observation spaces (by default, will include the prey which is heuristically controlled)
        self.observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (16+self.dim_capabilities,)) for i in self.adversaries
        }
        self.action_spaces = {i: Discrete(5) for i in self.adversaries}

        # Introduce partial observability by limiting the agents' view radii
        self.view_radius = jnp.concatenate(
            [
                jnp.full((num_agents), view_radius),
                jnp.full((num_landmarks), 0.0),
            ]
        )

        self.score_function = score_function

    def rewards(self, state: State) -> Dict[str, float]:
        @partial(jax.vmap, in_axes=(0, None))
        def _collisions(agent_idx: int, other_idx: int):
            return jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                agent_idx,
                other_idx,
                state,
            )

        c = _collisions(
            jnp.arange(self.num_good_agents) + self.num_adversaries,
            jnp.arange(self.num_adversaries),
        )  # [agent, adversary, collison]

        def _good(aidx: int, collisions: chex.Array):
            # in the original SimpleTag, the prey is also controlled by a policy, but here since it's a heuristic, return 0
            return 0
            # rew = -10 * jnp.sum(collisions[aidx])
            #
            # mr = jnp.sum(self.map_bounds_reward(jnp.abs(state.p_pos[aidx])))
            # rew -= mr
            # return rew

        ad_rew = 10 * jnp.sum(c)

        rew = {a: ad_rew for a in self.adversaries}
        rew.update(
            {
                a: _good(i + self.num_adversaries, c)
                for i, a in enumerate(self.good_agents)
            }
        )
        return rew

    def _prey_policy(self, key: chex.PRNGKey, state: State, aidx: int):
        """
        greedily move to the corner furthest from the closest predator at all steps

        returns an int, like policies would
        """
        # world bounds are -1/+1
        corners = jnp.array([[-0.9, -0.9], [0.9, 0.9], [0.9, -0.9], [0.9, 0.9]])

        # compute dists of all corners to all adversaries
        # here, dist[i, j] = dist(adversary_i, corner_j)
        adv_pos = state.p_pos[:self.num_adversaries]
        delta_pos = corners[None, :, :] - adv_pos[:, None, :]
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos), axis=2))

        # find the corner with the furthest closest adversary
        # axis=0 reduces over adversaries dim
        closest_adv_each_corner = jnp.min(dist, axis=0)
        best_corner = corners[jnp.argmax(closest_adv_each_corner)]

        # pick the discrete action which moves towards best corner
        dir_to_corner = best_corner - state.p_pos[aidx]
        dir_to_corner /= jnp.linalg.norm(dir_to_corner)

        action_vectors = jnp.array([[0,0], [-1,0], [+1,0], [0,-1], [0,+1]])
        action_vectors *= state.accel[aidx]
        dot = jnp.dot(action_vectors, dir_to_corner)
        best_action = jnp.argmax(dot)
        return best_action

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        # add prey actions manually (actions from policies only control adversaries, but MPE expects actions for all agents)
        for i in range(self.num_good_agents):
            actions[f"agent_{i}"] = self._prey_policy(key, state, self.num_adversaries+i) # num_adversaries come first in p_pos/accel
        u, c = self.set_actions(state, actions)

        if (
            c.shape[1] < self.dim_c
        ):  # This is due to the MPE code carrying around 0s for the communication channels, and due to added prey
            c = jnp.concatenate(
                [c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))], axis=1
            )

        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u)

        key_c = jax.random.split(key, self.num_agents)
        c = self._apply_comm_action(key_c, c, self.c_noise, self.silent)
        done = jnp.full((self.num_agents), state.step >= self.max_steps)

        state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            c=c,
            done=done,
            step=state.step + 1,
        )

        reward = self.rewards(state)
        obs = self.get_obs(state)

        info = {}

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        @partial(jax.vmap, in_axes=(0))
        def _common_stats(aidx):
            """Values needed in all observations"""

            landmark_pos = (
                    state.p_pos[self.num_agents:] - state.p_pos[aidx]
            )  # Landmark positions in agent reference frame

            # Zero out unseen agents with other_mask
            other_pos = state.p_pos[: self.num_agents] - state.p_pos[aidx]
            other_vel = state.p_vel[: self.num_agents]

            # use jnp.roll to remove ego agent from other_pos and other_vel arrays
            other_pos = jnp.roll(other_pos, shift=self.num_agents - aidx - 1, axis=0)[
                        : self.num_agents - 1
                        ]
            other_vel = jnp.roll(other_vel, shift=self.num_agents - aidx - 1, axis=0)[
                        : self.num_agents - 1
                        ]

            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)
            other_vel = jnp.roll(other_vel, shift=aidx, axis=0)

            # mask out entities and other agents not in view radius of agent
            landmark_mask = jnp.sqrt(jnp.sum(landmark_pos ** 2)) > self.view_radius[aidx]
            landmark_pos = jnp.where(landmark_mask, 0.0, landmark_pos)

            other_mask = jnp.sqrt(jnp.sum(other_pos ** 2)) > self.view_radius[aidx]
            other_pos = jnp.where(other_mask, 0.0, other_pos)
            other_vel = jnp.where(other_mask, 0.0, other_vel)

            return landmark_pos, other_pos, other_vel

        landmark_pos, other_pos, other_vel = _common_stats(self.agent_range)

        def _good(aidx): # prey
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 5, 2
                    other_pos[aidx].flatten(),  # 5, 2
                    jnp.zeros_like(other_vel[aidx, -1:].flatten())  # 2 - ensure same dimensionality for all agents!
                ]
            )

        # TODO: add cap to obs
        def _adversary(aidx): # predator
            adversary_cap = jnp.stack([
                state.accel[:self.num_adversaries].flatten(), state.rad[:self.num_adversaries].flatten(),
            ], axis=-1)

            ego_cap = adversary_cap[aidx, :]
            # roll to remove ego agent
            other_cap = jnp.roll(adversary_cap, shift=self.num_adversaries - aidx - 1, axis=0)[:self.num_adversaries-1, :]

            # mask out capabilities for non-capability-aware baselines
            if not self.capability_aware:
                other_cap = jnp.full(other_cap.shape, -1e3)
                ego_cap = jnp.full(ego_cap.shape, -1e3)

            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 5, 2
                    other_pos[aidx].flatten(),  # 5, 2
                    other_vel[aidx, -1:].flatten(),  # 2
                    # NOTE: caps must go last for hypernet logic
                    ego_cap.flatten(),  # n_cap
                    other_cap.flatten(),  # N-1, n_cap
                ]
            )

        obs = {a: _adversary(i) for i, a in enumerate(self.adversaries)}
        obs.update(
            {a: _good(i + self.num_adversaries) for i, a in enumerate(self.good_agents)}
        )
        return obs

    def get_world_state(self, state: State):
        @partial(jax.vmap, in_axes=(0))
        def _common_stats(aidx):
            """Values needed in all observations"""

            landmark_pos = (
                    state.p_pos[self.num_agents:] - state.p_pos[aidx]
            )  # Landmark positions in agent reference frame

            # Zero out unseen agents with other_mask
            other_pos = state.p_pos[: self.num_agents] - state.p_pos[aidx]
            other_vel = state.p_vel[: self.num_agents]

            # use jnp.roll to remove ego agent from other_pos and other_vel arrays
            other_pos = jnp.roll(other_pos, shift=self.num_agents - aidx - 1, axis=0)[
                        : self.num_agents - 1
                        ]
            other_vel = jnp.roll(other_vel, shift=self.num_agents - aidx - 1, axis=0)[
                        : self.num_agents - 1
                        ]

            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)
            other_vel = jnp.roll(other_vel, shift=aidx, axis=0)

            return landmark_pos, other_pos, other_vel

        landmark_pos, other_pos, other_vel = _common_stats(self.agent_range)

        def _good(aidx):
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 5, 2
                    other_pos[aidx].flatten(),  # 5, 2
                ]
            )

        def _adversary(aidx):
            return jnp.concatenate(
                [
                    state.p_vel[aidx].flatten(),  # 2
                    state.p_pos[aidx].flatten(),  # 2
                    landmark_pos[aidx].flatten(),  # 5, 2
                    other_pos[aidx].flatten(),  # 5, 2
                    other_vel[aidx, -1:].flatten(),  # 2
                ]
            )

        obs = {a: _adversary(i) for i, a in enumerate(self.adversaries)}
        obs.update(
            {a: _good(i + self.num_adversaries) for i, a in enumerate(self.good_agents)}
        )
        return obs

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Overriding superclass simple.py"""
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
        selected_agents = jax.random.choice(key_c, self.agent_range, shape=(self.num_agents,), replace=False)

        # unless a test distribution is provided and this is a test_env
        # TODO: fix test time capabilities (see simple_fire)
        # if self.test_env_flag and self.test_capabilities is not None:
        #     team_capabilities = jnp.asarray(self.test_capabilities)

        agent_rads = self.agent_rads[selected_agents]
        agent_accels = self.agent_accels[selected_agents]

        rad = jnp.concatenate(
            [
                # "adversaries" = predators (controlled by our policy)
                # jnp.full((self.num_adversaries), 0.075),
                agent_rads,
                # "good agents" = prey (controlled by pretrained policy)
                jnp.full((self.num_good_agents), 0.05),
                jnp.full((self.num_landmarks), 0.2),
            ]
        )

        accel = jnp.concatenate(
            [
                # "adversaries" = predators (controlled by our policy)
                # jnp.full((self.num_adversaries), 3.0),
                agent_accels,
                # "good agents" = prey (controlled by pretrained policy)
                jnp.full((self.num_good_agents), 4.0),
            ]
        )

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            accel=accel,
            rad=rad,
            done=jnp.full((self.num_agents), False),
            step=0,
        )

        return self.get_obs(state), state

if __name__ == "__main__":
    env = SimpleFacmacMPE(0)
    vec_step_env = jax.jit(env.step_env)
    jax.jit(env.step_env)
    import jaxmarl
    env = jaxmarl.make("MPE_simple_pred_prey_v1")
    get_obs = jax.jit(env.get_obs)

    num_envs = 128
    rng = jax.random.PRNGKey(30)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, num_envs)
    obsv, env_state = jax.vmap(env.reset)(reset_rng)

    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, num_envs)
    env_act = jnp.zeros((num_envs, 1))
    obsv, env_state, reward, done, info = jax.vmap(env.step)(rng_step, env_state, env_act)
    pass
