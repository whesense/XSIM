import torch

def remove_prefix(
        state_dict: dict[str, torch.Tensor],
        prefix: str
) -> dict[str, torch.Tensor]:
    new_state = {}

    for key in state_dict.keys():
        if key.startswith(prefix):
             new_state[key[len(prefix) :]] = state_dict[key]

    return new_state
