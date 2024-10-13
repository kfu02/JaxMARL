# # Sample list of new teams (replace this with your actual list)
# teams = [
#     [[0.9, 0.1], [0.7, 0.3], [1.0, 0.0], [0.0, 1.0]],
#     [[0.9, 0.1], [0.7, 0.3], [0.0, 1.0], [0.2, 0.8]],
#     [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.7, 0.3]],
#     [[1.0, 0.0], [0.0, 1.0], [0.1, 0.9], [0.3, 0.7]],
#     [[0.6, 0.4], [0.3, 0.7], [0.7, 0.3], [0.0, 1.0]],

#     [[0.2, 0.8], [0.6, 0.4], [0.8, 0.2], [1.0, 0.0]],
#     [[0.0, 1.0], [0.3, 0.7], [0.4, 0.6], [0.7, 0.3]],
#     [[0.0, 1.0], [0.6, 0.4], [0.8, 0.2], [1.0, 0.0]],
#     [[0.2, 0.8], [0.6, 0.4], [0.7, 0.3], [1.0, 0.0]],
#     [[0.3, 0.7], [0.4, 0.6], [0.6, 0.4], [0.9, 0.1]],
#     [[0.1, 0.9], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1]],
#     [[0.1, 0.9], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]],
#     [[0.1, 0.9], [0.3, 0.7], [0.4, 0.6], [0.8, 0.2]],
#     [[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [1.0, 0.0]],
#     [[0.1, 0.9], [0.3, 0.7], [0.4, 0.6], [0.7, 0.3]],
# ]

# # Dictionary to track team occurrences
# team_occurrences = {}

# # Check for duplicates
# for i, team in enumerate(teams):
#     # Convert the team to a sorted tuple
#     sorted_team = tuple(sorted(tuple(agent) for agent in team))
    
#     # Track occurrences of each team
#     if sorted_team in team_occurrences:
#         team_occurrences[sorted_team].append(team)
#         print(i)
#         print(team)
#     else:
#         team_occurrences[sorted_team] = [team]

# # Print duplicates
# print("Duplicate teams:")
# for team, occurrences in team_occurrences.items():
#     if len(occurrences) > 1:
#         print(f"Team {occurrences[0]} is duplicated {len(occurrences)} times.")


import numpy as np

# Define the agent values
# agent_values = np.array([
#     [0.05, 0.95], [0.15, 0.85], [0.25, 0.75], [0.35, 0.65], 
#     [0.45, 0.55], [0.55, 0.45], [0.65, 0.35], [0.75, 0.25], 
#     [0.85, 0.15], [0.95, 0.05]
# ])

# agent_values = np.array([
#     [0.0, 0.5], [0.1, 0.4], [0.2, 0.3], [0.3, 0.2], [0.4, 0.1], [0.5, 0.0]
# ])

agent_values = np.array([
    [0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5],
    [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1.0, 0.0]
])

# Number of teams and team size
num_teams = 10
team_size = 4

# To store unique teams
teams = set()

# Helper function to create a sorted tuple of the team to ensure uniqueness
def create_team():
    team_indices = np.random.choice(agent_values.shape[0], team_size, replace=False)
    team = agent_values[team_indices].tolist()
    team = tuple(sorted(tuple(agent) for agent in team))
    return team

# Generate 10 unique teams
while len(teams) < num_teams:
    team_tuple = create_team()
    print(team_tuple)
    teams.add(team_tuple)

# Convert back to array for easier handling
teams = np.array([list([list(agent) for agent in team]) for team in teams])

# Print the teams
for i, team in enumerate(teams):
    team = [list(agent) for agent in team]
    print(f"- {team}")