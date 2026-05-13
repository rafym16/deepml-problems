import torch

def ucb_action(counts: torch.Tensor, values: torch.Tensor, t: int, c: float) -> int:
    """
    Choose an action using the UCB1 formula.
    Args:
      counts (torch.Tensor): Number of times each action has been chosen
      values (torch.Tensor): Average reward of each action
      t (int): Current timestep (starts from 1)
      c (float): Exploration coefficient
    Returns:
      int: Index of action to select
    """
    # TODO: Implement the UCB action selection
    tensor_t = torch.tensor(t)
    tensor_c = torch.tensor(c)
    ucb = torch.argmax(
        values + (
            tensor_c * torch.sqrt(
            torch.log(tensor_t)/counts
        )
        )
    ).item()
    return ucb