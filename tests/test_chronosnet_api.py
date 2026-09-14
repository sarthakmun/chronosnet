"""
Unit tests for ChronosNet API, metadata, and core forecaster components.
"""
import pytest
import pandas as pd
import numpy as np
import chronosnet
from chronosnet import ChronosNet, NeuralProphet, split_df, set_random_seed


def test_chronosnet_metadata():
    """Verify package version, author and maintainer metadata."""
    assert chronosnet.__version__ == "1.0.0"
    assert chronosnet.__author__ == "Sarthak Mun"
    assert chronosnet.__email__ == "sarthak.mun03@gmail.com"


def test_chronosnet_model_initialization():
    """Test ChronosNet model construction and configuration."""
    model = ChronosNet(
        n_forecasts=3,
        n_lags=5,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=2,
        batch_size=16,
    )
    assert model.config_model.n_forecasts == 3
    assert model.config_ar.n_lags == 5
    assert model.config_train.epochs == 2
    assert model.config_train.batch_size == 16


def test_split_df():
    """Test synthetic time series generation and dataframe splitting."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    values = np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100)
    df = pd.DataFrame({"ds": dates, "y": values, "ID": "series_1"})

    df_train, df_val = split_df(df, n_lags=0, n_forecasts=1, valid_p=0.2)
    assert len(df_train) == 80
    assert len(df_val) == 20
    assert "ds" in df_train.columns
    assert "y" in df_train.columns


def test_set_random_seed():
    """Test reproducibility seed setting."""
    set_random_seed(42)
