"""
ChronosNet: Deep Neural Time-Series Forecasting & Interpretable AR-Net Engine.
Engineered by Sarthak Mun (IIT Kharagpur) <sarthak.mun03@gmail.com>.
"""

__version__ = "1.0.0"
__author__ = "Sarthak Mun"
__email__ = "sarthak.mun03@gmail.com"

# Re-export core forecasting components
from neuralprophet.forecaster import NeuralProphet
from neuralprophet.torch_prophet import TorchProphet
from neuralprophet.df_utils import add_quarter_condition, add_weekday_condition, split_df
from neuralprophet.uncertainty import uncertainty_evaluate
from neuralprophet.utils import load, save, set_log_level, set_random_seed

# Alias ChronosNet to NeuralProphet for intuitive branded usage
ChronosNet = NeuralProphet

__all__ = [
    "ChronosNet",
    "NeuralProphet",
    "TorchProphet",
    "split_df",
    "add_quarter_condition",
    "add_weekday_condition",
    "uncertainty_evaluate",
    "load",
    "save",
    "set_log_level",
    "set_random_seed",
    "__version__",
    "__author__",
    "__email__",
]
