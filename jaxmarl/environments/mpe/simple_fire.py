import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, List
from functools import partial
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from gymnax.environments.spaces import Box

class SimpleFireMPE(SimpleMPE):
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
            i:Box(-jnp.inf, jnp.inf, (num_agents*(2+2+num_capabilities) + num_landmarks*(2+1),)) 
            for i in agents
        }

        self.colour = [AGENT_COLOUR] * num_agents + [OBS_COLOUR] * num_landmarks

        # Env specific parameters
        self.test_team = kwargs["test_team"] if "test_team" in kwargs else None
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
            def shift_array(arr, i):
                """
                Assuming arr is 2D, moves row i to the front
                """
                i = i % arr.shape[0]
                first_part = arr[i:]
                second_part = arr[:i]
                return jnp.concatenate([first_part, second_part])

            # move ego_pos to front of agent_pos, then remove
            agent_pos = state.p_pos[:self.num_agents, :]
            other_pos = shift_array(agent_pos, aidx)
            ego_pos = other_pos[0]
            other_pos = other_pos[1:]
            rel_other_pos = other_pos - ego_pos # and transform to relative pos

            ego_vel = state.p_vel[aidx, :]

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

            # give agents the pos and rad of all landmarks (fires)
            landmark_p_pos = state.p_pos[self.num_agents:]
            # transform to relative pos
            rel_landmark_p_pos = landmark_p_pos - ego_pos
            landmark_rads = state.rad[self.num_agents:]

            obs = jnp.concatenate([
                ego_pos.flatten(),  # 2
                rel_other_pos.flatten(),  # N-1, 2
                ego_vel.flatten(),  # 2
                rel_landmark_p_pos.flatten(), # 2, 2
                landmark_rads.flatten(), # 1, 2
                # NOTE: caps must go last for hypernet logic
                ego_cap.flatten(),  # n_cap
                other_cap.flatten(),  # N-1, n_cap
            ])

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
        landmark_p_pos = state.p_pos[self.num_agents:]
        for i, landmark_pos in enumerate(landmark_p_pos):
            landmark_rad = landmark_rads[i]

            # take the sum of all agents ff on this landmark
            agents_on_landmark = jax.vmap(_agent_in_range, in_axes=[0, None, None])(self.agent_range, landmark_pos, landmark_rad)
            firefighting_level = jnp.sum(jnp.where(agents_on_landmark, agent_rads, 0))

            # dense rew for firefighting
            enough_firefight = firefighting_level >= landmark_rads[i]
            # NOTE: reward based on how much of fire is covered, but cap at 0
            # (since !enough_firefight means ff_level < landmark_rads, this second term is always < 0)
            ff_rew = jnp.where(enough_firefight, 0, 2*(firefighting_level-landmark_rads[i]))

            # only add reward if this fire is valid (rad > 0)
            global_rew = jnp.where(landmark_rad > 0, global_rew+ff_rew, global_rew)

        # normalize global rew based on how many active fires there are
        active_fires = jnp.count_nonzero(landmark_rads > 0)
        global_rew /= active_fires

        # reward each agent for getting closer to one of the landmarks
        def _dist_to_landmarks(agent_pos):
            dist_to_landmarks = jnp.linalg.norm(agent_pos - landmark_p_pos, axis=1)
            return dist_to_landmarks

        def _agent_rew(agent_i):
            agent_pos = state.p_pos[agent_i]
            dists = _dist_to_landmarks(agent_pos)
            # TODO: set rew to 0 if agent is within landmark rad
            return -jnp.min(dists)

        rew = {
            a: global_rew + (0.01 * _agent_rew(i))
            for i, a in enumerate(self.agents)
        }
        return rew

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Overriding superclass simple.py"""
        # NOTE: copy-pasted from simple.py, bad practice

        key_a, key_l, key_c, key_fr = jax.random.split(key, num=4)

        # spawn landmarks (fires) s.t. they don't overlap
        landmark_rads = jax.random.uniform(key_fr, (self.num_landmarks,), minval=0.10, maxval=0.40)

        def _spawn_one_fire(carry, _):
            key_l, existing_fires, fire_index = carry

            # spawn a new random fire
            key_l, key_fs = jax.random.split(key_l)
            new_fire = jax.random.uniform(
                key_fs, (1, 2), minval=-1.0, maxval=+1.0
            )

            # it's valid if it doesn't intersect with any existing fires (that fire's rad + new fire rad)
            dists = jnp.linalg.norm(existing_fires - new_fire, axis=1)
            new_fire_valid = jnp.all(dists > (landmark_rads + landmark_rads[fire_index]))

            new_fire = jnp.reshape(new_fire, (2,))
            new_fire_added = existing_fires.at[fire_index, :].set(new_fire)

            # if new fire spawn is valid, add it to the fire list, and incr the fire index
            new_fires = jax.lax.cond(
                new_fire_valid, 
                lambda: new_fire_added, # T
                lambda: existing_fires, # F
            )
            fire_index = jax.lax.cond(
                new_fire_valid,
                lambda: fire_index+1,
                lambda: fire_index,
            )
            return (key_l, new_fires, fire_index), None

        # use jax.lax.scan to spawn N more fires, each of which does not collide with any prev
        key_l, key_fs = jax.random.split(key_l)
        # set the init spawns to be far from the actual spawns, s.t. array shape can be maintained in the carry
        init_fires = jax.random.uniform(
            key_fs, (self.num_landmarks, 2), minval=-100.0, maxval=-100.0
        )
        initial_state = (key_l, init_fires, 0)
        MAX_ITERS = 10*self.num_landmarks # try ~10 times for each fire, each time
        (key_l, landmark_p_pos, _), _ = jax.lax.scan(_spawn_one_fire, initial_state, None, length=MAX_ITERS)

        """
        # randomly decide to spawn between 1 and NUM_LANDMARKS fires (NUM_LANDMARKS = MAX_FIRES)
        # then mask out landmark_rads/landmark_p_pos as needed
        key_l, key_num_fires = jax.random.split(key_l)
        num_fires = jax.random.randint(key_num_fires, (), 1, self.num_landmarks+1)

        # this mask dictates which fires to be masked out
        # and works by index-wise comparing indices array with RNG num_fires from above
        # e.g. indices=[0, 1, 2], num_fires=2 -> mask=[T, T, F]
        # num_to_mask = (self.num_landmarks - num_fires)
        num_to_mask = 0
        mask = jnp.arange(self.num_landmarks) < num_to_mask
        landmark_rads = jnp.where(mask, -10, landmark_rads)

        # expand mask to correctly match p_pos shape
        mask = jnp.stack([mask, mask], axis=1)
        landmark_p_pos = jnp.where(mask, -10, landmark_p_pos)
        """

        p_pos = jnp.concatenate(
            [
                # spawn agents in the same spot (center), like a real fire depot
                jnp.zeros((self.num_agents, 2)),
                # OLD: spawn randomly, in same range as fires
                # jax.random.uniform(
                #     key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0
                # ),
                landmark_p_pos,
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
            c=jnp.zeros((self.num_agents, self.dim_c)),
            accel=agent_accels,
            rad=jnp.concatenate(
                # NOTE: here, must define landmark rad as well, by default landmarks are 0.05
                [agent_rads, landmark_rads]
            ),
            done=jnp.full((self.num_agents), False),
            step=0,
        )

        return self.get_obs(state), state
