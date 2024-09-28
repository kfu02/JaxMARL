import jax
import jax.numpy as jnp

def total_variational_distance(p, q):
    """
    Get the distance between two categorical distributions
    """

    return 0.5 * jnp.sum(jnp.abs(p - q), axis=-1)

def snd(rollouts, hiddens, policy, dim_c):
    """
    Calculate system neural diversity metric
    """

    batch_size, timsteps, n_agents, obs_dim = rollouts.shape

    def mask_obs(agent_i):
        """
        For agents j /= i, mask the non-capability observation of j with i
        """

        obs = rollouts[..., :-dim_c]
        cap = rollouts[..., -dim_c:]
        obs_i = obs[:, :, agent_i, :]
        obs_i = obs_i[:, :, None, :] # add extra dim for broadcasting
        obs_masked = jnp.tile(obs_i, (1, 1, n_agents, 1))
        
        return jnp.concatenate((obs_masked, cap), axis=-1)
    
    def get_policy_outputs(agent_i):
        """
        Using mask obs, get policy outputs for masked observations simulating generating
        outputs for the same observation conditioned on different capabilities
        """
        obs = mask_obs(agent_i)
        
        # duplicate hidden state for i across each agent
        hs = hiddens[:, :, agent_i, :]
        hs = hs[:, :, None, :] # add extra dim for broadcasting
        hs = jnp.tile(hs, (1, 1, n_agents, 1))

        # trivally set dones to 0
        dones = jnp.zeros_like(hs)
        
        return policy(hiddens, (obs, hs))


    def distance_vector(agent_i):
        """
        Using get policy outputs, compute the distance between the distributions outputted by
        corresponding observations, add to flatten into a vector containing the added distances
        between i and each other agent 
        """
        qvals_i = get_policy_outputs(agent_i) # [batch_size, timesteps, n_agents, action_dim]

        # convert qvals to categorical distributions
        categorical_i = qvals_i / jnp.sum(qvals_i, axis=-1, keepdims=True)

        # get pairwise distance between agent i and all other agents
        def tvd_for_agent_j(j):
            return total_variational_distance(categorical_i[:, :, agent_i, :], categorical_i[:, :, j, :])
        tvd_all_agents = jax.vmap(tvd_for_agent_j)(jnp.arange(n_agents))  # [n_agents, batch_size, timesteps]

        return tvd_all_agents  # [n_agents, batch_size, timesteps]

    # get distance matrix
    dist_matrix = jax.vmap(distance_vector)(jnp.arange(n_agents))  # [n_agents, n_agents, batch_size, timesteps]

    # average the distance matrix over the batch size and timesteps
    dist_matrix_avg = jnp.mean(dist_matrix, axis=(2, 3))  # [n_agents, n_agents]

    # compute the final SND metric
    snd_value = (2 / (n_agents * (n_agents - 1))) * jnp.sum(dist_matrix_avg) / 2# divide by 2 to account for double counting distances

    return snd_value

def dummy_homogenous_policy(hiddens, x):
    """
    dummy policy for testing, all qvals are 1
    """

    batch_size, timesteps, n_agents, _ = hiddens.shape

    return jnp.ones((batch_size, timesteps, n_agents, 5))

def dummy_homogenous_policy_2(hiddens, x):
    """
    dummy policy for testing, qvals for agent 0 are all 1's, agent 1 are all 2's, etc.
    """
    batch_size, timesteps, n_agents, obs_dim = hiddens.shape

    qvals = jnp.arange(1, n_agents + 1) 
    qvals = jnp.tile(qvals, (batch_size, timesteps, 1))
    qvals = qvals[:, :, :, None]  # add dim for tiling

    return jnp.tile(qvals, (1, 1, 1, 4))

def dummy_heterogeneous_policy(hiddens, x):
    """
    dummy policy for testing, qvals are all distinct
    """
    batch_size, timesteps, n_agents, obs_dim = hiddens.shape

    qvals = jnp.eye(n_agents)
    qvals = qvals[None, None, :, :] # add extra dims for tiling

    return jnp.tile(qvals, (batch_size, timesteps, 1, 1))

def main():
    """
    sanity checks
    """

    batch_size = 2
    timesteps = 3
    n_agents = 4
    obs_dim = 5
    dim_c = 2

    # ignorable observations and hidden states
    same_obs = jnp.ones((batch_size, timesteps, n_agents, obs_dim))
    hiddens = jnp.ones((batch_size, timesteps, n_agents, 8))

    # get snd for homogenous policy 1
    snd_value = snd(same_obs, hiddens, dummy_homogenous_policy, dim_c)
    assert jnp.isclose(snd_value, 0.0), f"Test failed: SND expected be 0 but got {snd_value}"
    print(f"YAY: SND = {snd_value} for homogenous_policy_1")

    # get snd homoenous policy 2
    snd_value = snd(same_obs, hiddens, dummy_homogenous_policy_2, dim_c)
    assert jnp.isclose(snd_value, 0.0), f"Test failed: SND expected be 0 but got {snd_value}"
    print(f"YAY: SND = {snd_value} for homogenous_policy_2")

    # get snd heterogeneous policy
    snd_value = snd(same_obs, hiddens, dummy_heterogeneous_policy, dim_c)
    assert jnp.isclose(snd_value, 1.0), f"Test failed: SND expected be 1 but got {snd_value}"
    print(f"YAY: SND = {snd_value} for homogenous_policy_2")

if __name__ == "__main__":
    main()