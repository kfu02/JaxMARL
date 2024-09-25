import jax
import jax.numpy as jnp
import chex
from functools import partial
from gymnax.environments.spaces import Box
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from typing import Tuple, Dict

class SimpleTransportMPE(SimpleMPE):
    def __init__(
        self,
        num_agents=4,
        action_type=DISCRETE_ACT,
        capability_aware=True,
        num_capabilities=2,
        **kwargs,
    ):
        agents = ["agent_{}".format(i) for i in range(num_agents)]

        # 3 landmarks, [concrete depo, lumber depo, construction site]
        self.num_landmarks = 3
        landmarks = ["landmark_{}".format(i) for i in range(3)]

        self.num_agents = num_agents
        self.capability_aware = capability_aware
        self.num_capabilities = num_capabilities
        self.dim_capabilities = num_agents * num_capabilities
        self.test_team = kwargs.get("test_team", None)

        # observation dimensions
        pos_dim = num_agents * 2
        vel_dim = 2  # for ego agent
        material_depot_dim = 2 * 2 # 2 materials, 2 positions
        construction_site_dim = 2
        payload_dim = 1

        # initialize observation space for each agent
        observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (pos_dim + vel_dim + material_depot_dim + construction_site_dim + payload_dim + self.dim_capabilities))
            for i in agents
        }

        # overriden in reset to reflect max capability
        self.colour = [(115, 243, 115)] * num_agents + [(255, 64, 64)] * len(landmarks)

        # reward shaping terms
        self.concrete_pickup_reward = kwargs.get("concrete_pickup_reward", 0.25)
        self.lumber_pickup_reward = kwargs.get("lumber_pickup_reward", 0.25)
        self.dropoff_reward = kwargs.get("dropoff_reward", 0.75)

        # no collisions in this env
        collide = jnp.concatenate(
            [jnp.full((num_agents+len(landmarks)), False)]
        )

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=len(landmarks),
            landmarks=landmarks,
            action_type=action_type,
            observation_spaces=observation_spaces,
            dim_c=0, # no comm
            colour=self.colour,
            # NOTE: modified via reset(), see below
            # rad=rad,
            collide=collide,
            **kwargs,
        )
    
    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        """
        Override simple.py step_env to update payload information
        """
        u, c = self.set_actions(state, actions)
        if (
            c.shape[1] < self.dim_c
        ):  # This is due to the MPE code carrying around 0s for the communication channels
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

        #######################################################################################
        # Start modified step
        # only update payloads after rewards are applied (I think this makes sense?)
        #######################################################################################
        # get distance between agents and landmarks
        agent_p_pos = state.p_pos[:self.num_agents]
        landmark_p_pos = state.p_pos[self.num_agents:]
        relative_positions = agent_p_pos[:, None, :] - landmark_p_pos[None, :, :]
        dists = jnp.linalg.norm(relative_positions, axis=-1)

        # get mask of agents within landmark radius
        landmark_rads = state.rad[self.num_agents:]
        mask = dists <= landmark_rads

        def io_callback(x, i):
            if x[0][0].item() == 1.0:
                print(f"payload {i}: {x[0][0].item()}")
        # update payload for agents on concrete depot
        payload = jnp.where(jnp.bitwise_and(mask[:, 0].reshape(-1,1), (state.payload == 0.)), state.capacity[:, 0].reshape(-1,1), state.payload)
        # jax.debug.callback(io_callback, payload, 0)

        # update payload for agents on lumber depot
        payload = jnp.where(jnp.bitwise_and(mask[:, 1].reshape(-1,1), (state.payload == 0.)), state.capacity[:, 1].reshape(-1,1), payload)
        # jax.debug.callback(io_callback, payload, 1)

        # reset payload for agents on construction site
        payload = jnp.where(jnp.bitwise_and(mask[:, 2].reshape(-1,1), (state.payload > 0.)), jnp.zeros_like(payload), payload)
        # jax.debug.callback(io_callback, payload, 2)

        state = state.replace(
            payload=payload,
        )
        #######################################################################################
        # End modified step
        #######################################################################################

        obs = self.get_obs(state)

        info = {}

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info

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

            # agent positions, move ego_pos to front of agent_pos, then remove
            agent_pos = state.p_pos[:self.num_agents, :]
            other_pos = shift_array(agent_pos, aidx)
            ego_pos = other_pos[0]
            other_pos = other_pos[1:]
            rel_other_pos = other_pos - ego_pos # and transform to relative pos

            # ego agent velocities
            ego_vel = state.p_vel[aidx, :]

            # agent capabilities, separate ego capability from other agents capability
            other_cap = state.capacity
            ego_cap = other_cap[aidx, :]
            other_cap = jnp.roll(other_cap, shift=self.num_agents - aidx - 1, axis=0)[:self.num_agents-1, :]
            
            # mask out capabilities for non-capability-aware baselines
            if not self.capability_aware:
                other_cap = jnp.full(other_cap.shape, MASK_VAL)
                ego_cap = jnp.full(ego_cap.shape, MASK_VAL)

            # relative position of all landmarks
            landmark_p_pos = state.p_pos[self.num_agents:]
            rel_landmark_p_pos = landmark_p_pos - ego_pos

            # current payload
            payload = state.payload[aidx, :]
            
            obs = jnp.concatenate([
                ego_pos.flatten(),  # 2
                rel_other_pos.flatten(),  # N-1, 2
                ego_vel.flatten(),  # 2
                rel_landmark_p_pos.flatten(), # 2, 2
                payload.flatten(), # 1
                # NOTE: caps must go last for hypernet logic
                ego_cap.flatten(),  # n_cap
                other_cap.flatten(),  # N-1, n_cap
            ])

            return obs

        obs = {a: _obs(i) for i, a in enumerate(self.agents)}
        return obs
    
    def rewards(self, state: State) -> Dict[str, float]:
        """
        Reward agents according to objective of maximizing delivered materials.
        """

        def _load_concrete_rew(agent_i):
            """
            Reward agent for loading concrete if payload is empty.
            """
            agent_pos = state.p_pos[agent_i]
            concrete_depot_pos = state.p_pos[-3]
            dist = jnp.array([jnp.linalg.norm(agent_pos - concrete_depot_pos)])
            return jnp.bitwise_and(dist <= state.rad[-3], state.payload[agent_i] == 0)
        
        def _load_lumber_rew(agent_i):
            """
            Reward agent for loading lumber if payload is empty.
            """
            agent_pos = state.p_pos[agent_i]
            lumber_depot_pos = state.p_pos[-2]
            dist = jnp.array([jnp.linalg.norm(agent_pos - lumber_depot_pos)])
            return jnp.bitwise_and(dist <= state.rad[-2], state.payload[agent_i] == 0)

        def _dropoff_rew(agent_i):
            """
            Reward agent for dropping of materials if payload is non empty
            """
            agent_pos = state.p_pos[agent_i]
            construction_site_pos = state.p_pos[-1]
            dist = jnp.array([jnp.linalg.norm(agent_pos - construction_site_pos)])
            return jnp.bitwise_and(dist <= state.rad[-1], state.payload[agent_i] > 0) * state.payload[agent_i]      

        rew = {
            a: (self.concrete_pickup_reward * _load_concrete_rew(i) + self.lumber_pickup_reward * _load_lumber_rew(i) + self.dropoff_reward * _dropoff_rew(i))[0]
            for i, a in enumerate(self.agents)
        }
        return rew
    
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """
        Override reset in simple.py to fix the location of the depots and construction site.
        """

        key_a, key_l = jax.random.split(key)

        p_pos = jnp.concatenate(
            [
                jax.random.uniform(
                    key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0
                ),
                jnp.array(
                    [
                        [-1.0, 1.0],
                        [1.0, 1.0],
                        [0.0, -1.0],
                    ]
                ),
            ]
        )

        # randomly sample N_agents' capabilities from the possible agent pool (hence w/out replacement)
        selected_agents = jax.random.choice(key_a, self.agent_range, shape=(self.num_agents,), replace=False)
        
        agent_rads = jnp.array([0.2, 0.2, 0.2, 0.2]) # self.agent_rads[selected_agents]
        agent_accels = jnp.array([5, 5, 5, 5]) # self.agent_accels[selected_agents]
        agent_capacities = jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]) # self.agent_capacities[selected_agents]

        # if a test distribution is provided and this is a test_env, override capacities
        # NOTE: also add other capabilities here?
        if self.test_env_flag and self.test_team is not None:
            agent_capacities = jnp.array(self.test_team["agent_capacities"])

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            accel=agent_accels,
            rad=jnp.concatenate(
                # NOTE: here, must define landmark rad as well, by default landmarks are 0.30
                [agent_rads, jnp.full((self.num_landmarks), 0.30)]
            ),
            done=jnp.full((self.num_agents), False),
            step=0,
            payload=jnp.zeros((self.num_agents, 1)),
            capacity=agent_capacities,
        )

        return self.get_obs(state), state
