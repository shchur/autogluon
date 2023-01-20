import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.ensemble import AbstractTimeSeriesEnsembleModel

logger = logging.getLogger(__name__)


class WeightedQuantileAdjustmentModel(nn.Module):
    """Each model has 2 weights that adjust the width of the prediction intervals around the median."""

    def __init__(self, num_models: int, quantile_levels: List[float]):
        super().__init__()
        assert 0.5 in quantile_levels
        self.num_models = num_models
        quantile_levels = list(sorted(quantile_levels))
        self.column_names = [str(q) for q in quantile_levels]
        self.median_index = self.column_names.index("0.5")
        self.lr = 1e-2
        self.max_epochs = 4000
        self.min_weight_threshold = 0.02

        self.register_buffer("quantile_levels", torch.tensor(quantile_levels, dtype=torch.float32))
        self.logit_weights = nn.Parameter(torch.zeros(self.num_models))
        self.log_scales = nn.Parameter(torch.zeros(2))
        # 0 for lower quantiles, 1 for upper quantiles
        self.register_buffer(
            "scale_selector", torch.tensor([q <= 0.5 for q in self.quantile_levels], dtype=torch.long)
        )

    def get_sparsified_weights(self) -> np.ndarray:
        """Return normalized weights assigned to the base models."""
        weights = self.logit_weights.softmax(dim=-1).cpu().detach().numpy()
        weights[weights < self.min_weight_threshold] = 0.0
        weights = weights / weights.sum()
        return weights

    def forward(self, stacked_predictions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_predictions = stacked_predictions @ weights  # [num_timesteps, num_quantiles]
        median = weighted_predictions[:, [self.median_index]]  # [num_timesteps, 1]
        scale = self.log_scales.exp()[self.scale_selector]  # [num_quantiles]
        return weighted_predictions * scale + (1 - scale) * median

    def quantile_loss(self, predictions: torch.Tensor, target: torch.Tensor):
        # predictions: shape [num_timestemps, num_columns]
        # target: shape [num_timestemps, 1]
        loss = ((target - predictions) * ((target <= predictions).float() - self.quantile_levels)).abs().sum(dim=0)
        return 2 * (loss / target.abs().sum()).mean()

    def fit(self, predictions: List[TimeSeriesDataFrame], labels: pd.Series, time_limit: Optional[int] = None):
        assert all(labels.index.equals(pred.index) for pred in predictions)
        stacked_predictions = self.prepare_inputs(predictions)
        target = torch.tensor(labels.values, dtype=torch.float32).view(-1, 1)
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        for epoch in range(self.max_epochs):
            preds = self(stacked_predictions, self.logit_weights.softmax(dim=-1))
            loss = self.quantile_loss(preds, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 250 == 0:
                print(
                    f"Epoch {epoch:4d}: loss = {loss.item():.3f}, {self.logit_weights.softmax(dim=-1).detach().numpy().round(2)}"
                )

    def prepare_inputs(self, predictions: List[TimeSeriesDataFrame]) -> torch.Tensor:
        assert all(predictions[0].index.equals(pred.index) for pred in predictions)
        return torch.stack(
            [torch.tensor(pred[self.column_names].values, dtype=torch.float32) for pred in predictions], dim=-1
        )

    def predict(self, predictions: List[TimeSeriesDataFrame]) -> TimeSeriesDataFrame:
        stacked_predictions = self.prepare_inputs(predictions)
        weights = self.get_sparsified_weights()
        nonzero_weights = torch.tensor(weights[weights != 0])
        preds = self(stacked_predictions, weights=nonzero_weights).detach().cpu().numpy()
        df = pd.DataFrame(
            preds,
            columns=self.column_names,
            index=predictions[0].index,
        )
        df["mean"] = df["0.5"]
        return TimeSeriesDataFrame(df)


class TimeSeriesWeightedEnsemble(AbstractTimeSeriesEnsembleModel):
    """Constructs a simple weighted ensemble using gradient descent."""

    def __init__(self, name: str, eval_metric: str, target: str, **kwargs):
        super().__init__(name=name, eval_metric=eval_metric, target=target, **kwargs)
        self.model_to_weight: Dict[str, float] = {}
        self.wqa: WeightedQuantileAdjustmentModel = None

    def _fit_ensemble(
        self,
        predictions: Dict[str, TimeSeriesDataFrame],
        data: TimeSeriesDataFrame,
        time_limit: Optional[int] = None,
        **kwargs,
    ):
        logger.info(f"Fitting {self.__class__.__name__} ensemble")
        predictions_list = list(predictions.values())
        self.wqa = WeightedQuantileAdjustmentModel(
            num_models=len(predictions_list), quantile_levels=self.quantile_levels
        )
        self.wqa.fit(
            predictions=predictions_list,
            labels=data.slice_by_timestep(-self.prediction_length, None)[self.target],
            time_limit=time_limit,
        )
        self.model_to_weight = {}
        for model_name, weight in zip(predictions.keys(), self.wqa.get_sparsified_weights()):
            if weight != 0:
                self.model_to_weight[model_name] = weight
        print(f"WQA weights: {self.model_to_weight}")

    @property
    def model_names(self) -> List[str]:
        return list(self.model_to_weight.keys())

    def predict(self, data: Dict[str, TimeSeriesDataFrame], **kwargs) -> TimeSeriesDataFrame:
        if set(data.keys()) != set(self.model_names):
            raise ValueError(
                f"Set of models given for prediction in {self.name} differ from those provided during initialization."
            )
        failed_models = [model_name for (model_name, model_pred) in data.items() if model_pred is None]
        if len(failed_models) == len(data):
            raise RuntimeError(f"All input models failed during prediction, {self.name} cannot predict.")
        if len(failed_models) > 0:
            logger.warning(
                f"Following models failed during prediction: {failed_models}. "
                f"{self.name} will set the weight of these models to zero and re-normalize the weights when predicting."
            )
        # Make sure that all predictions have same shape
        assert len(set(pred.shape for pred in data.values() if pred is not None)) == 1
        model_preds = [data[model_name] for model_name in self.model_names]
        return self.wqa.predict(model_preds)
