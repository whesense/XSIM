import pandas as pd
import numpy as np
import torch


def df_to_tensor(df: pd.DataFrame, key: str) -> torch.Tensor:
    arr = np.stack(df[key].values)
    return torch.as_tensor(arr)

