import jax.numpy as jnp

def total_variational_distance(p, q):
    """
    Get the distance between two categorical distributions
    """

    return 0.5 * jnp.sum(jnp.abs(p - q), axis=-1)

def snd(rollouts, policy):

    batch_size, timsteps, n_agents, obs_dim = rollouts.shape

    def mask_obs(agent_i):
        """
        For agents j /= i, mask the non-capability observation of j with i
        """
    
    def get_policy_outputs(agent_i)
        """
        Using mask obs, get policy outputs for masked observations simulating generating
        outputs for the same observation conditioned on different capabilities
        """
        obs = mask_obs(obs, agent_i)

    def distance_vector(agent_i)
        """
        Using get policy outputs, compute the distance between the distributions outputted by
        corresponding observations, add to flatten into a vector containing the added distances
        between i and each other agent 
        """
        qvals = get_policy_outputs(obs, agent_i)

    dist_matrix = jax.vmap(distance_vector)(jnp.arange(n_agents))
    dist_matrix = dist_matrix /= (batch_size * timesteps, n_agents)

    snd = (2 / (n_agents * (n_agents - 1))) * jnp.triu(snd_matrix).sum()