import torch

def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> torch.Tensor:
    """
    Evaluate state-value function for a policy on a 5x5 gridworld.

    Args:
        policy: dict mapping (row, col) to action probability dicts
        gamma: discount factor
        threshold: convergence threshold
    Returns:
        5x5 torch.Tensor of floats
    """
    # Your code here
    size_grid = 5

    V = torch.zeros((size_grid, size_grid), dtype=torch.float32)

    def is_terminal(i, j):
        return (i == 0 and j == 0) or (i == 0 and j == size_grid -1) or (i == size_grid - 1 and j == 0) or (i == size_grid - 1 and j == size_grid - 1)

    def get_next_state(i, j, action):
        if action.lower() == 'up':
            return (max(i - 1, 0), j)  # tidak bisa keluar grid
        elif action.lower() == 'down':
            return (min(i + 1, size_grid - 1), j)
        elif action.lower() == 'left':
            return (i, max(j - 1, 0))
        elif action.lower() == 'right':
            return (i, min(j + 1, size_grid - 1))

    while True:
        delta = torch.tensor(0.0, dtype=torch.float32)
        new_v = torch.clone(V)
        for row in range(size_grid):
            for col in range(size_grid):
                if is_terminal(row, col):
                    continue

                v = 0.0

                for action, prob in policy[(row, col)].items():
                    new_i, new_j = get_next_state(row, col, action)

                    reward = -1

                    v += prob * (reward + gamma * V[new_i, new_j])

                new_v[row, col] = v

                delta = torch.max(delta, torch.abs(v - V[row, col]))

        V = new_v

        if delta.item() < threshold:
            break

    return V
