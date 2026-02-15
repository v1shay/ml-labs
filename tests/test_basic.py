import pandas as pd
import torch
import ml_labs
import numpy as np

def test_pandas_basic():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    assert df.shape == (2, 2)
    assert df['a'].sum() == 3

def test_torch_basic():
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert torch.sum(t).item() == 6.0

def test_ml_labs_import():
    from ml_labs.core.types import DatasetModality
    assert DatasetModality.TABULAR == "tabular"

def test_numpy_basic():
    a = np.array([1, 2, 3])
    assert a.sum() == 6
