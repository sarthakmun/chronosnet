"""
Compatibility module for PyTorch Lightning / Lightning PyTorch.
"""
import sys

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, TQDMProgressBar
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.tuner.tuning import Tuner
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
    HAS_PL = True
except (ImportError, ModuleNotFoundError):
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import Callback, TQDMProgressBar
        from lightning.pytorch.loggers import TensorBoardLogger
        from lightning.pytorch.tuner.tuning import Tuner
        from lightning.pytorch.utilities.rank_zero import rank_zero_only
        HAS_PL = True
    except (ImportError, ModuleNotFoundError):
        import torch.nn as nn

        class _PLMock:
            class LightningModule(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()

            class Trainer:
                def __init__(self, *args, **kwargs):
                    pass

                def fit(self, *args, **kwargs):
                    pass

                def predict(self, *args, **kwargs):
                    return []

                def test(self, *args, **kwargs):
                    return []

            class utilities:
                class warnings:
                    class PossibleUserWarning(Warning):
                        pass

            @staticmethod
            def seed_everything(seed=42):
                pass

        pl = _PLMock()

        class Callback:
            pass

        class TQDMProgressBar:
            def __init__(self, *args, **kwargs):
                pass

        class TensorBoardLogger:
            def __init__(self, *args, **kwargs):
                pass

        def rank_zero_only(fn):
            return fn

        class Tuner:
            def __init__(self, *args, **kwargs):
                pass

        HAS_PL = False
