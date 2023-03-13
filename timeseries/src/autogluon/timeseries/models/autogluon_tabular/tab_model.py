import logging
import re
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str
from joblib.parallel import Parallel, delayed

import autogluon.core as ag
from autogluon.tabular import TabularPredictor
from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame, ITEMID, TIMESTAMP
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel

logger = logging.getLogger(__name__)


class AbstractTabularModel(AbstractTimeSeriesModel):
    TIMESERIES_METRIC_TO_TABULAR_METRIC = {
        "MASE": "mean_absolute_error",
        "MAPE": "mean_absolute_percentage_error",
        "sMAPE": "mean_absolute_percentage_error",
        "mean_wQuantileLoss": "mean_absolute_error",
        "MSE": "mean_squared_error",
        "RMSE": "root_mean_squared_error",
    }

    default_tabular_hyperparameters = {
        "XGB": {},
        "CAT": {},
        "GBM": {},
        "LR": {"proc.impute_strategy": "constant"},
        "NN_TORCH": {"proc.impute_strategy": "constant"},
    }

    features_are_masked: bool

    def __init__(
        self,
        freq: Optional[str] = None,
        prediction_length: int = 1,
        path: Optional[str] = None,
        name: Optional[str] = None,
        eval_metric: str = None,
        hyperparameters: Dict[str, Any] = None,
        **kwargs,  # noqa
    ):
        name = name or re.sub(r"Model$", "", self.__class__.__name__)  # TODO: look name up from presets
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs,
        )
        self._target_lag_indices: np.array = None
        self._known_covariates_lag_indices: np.array = None
        self._past_covariates_lag_indices: np.array = None
        self._time_features: List[Callable] = None
        self._available_features: pd.Index = None
        self.quantile_adjustments: Dict[str, float] = {}

        self.tabular_predictor = TabularPredictor(
            path=self.path,
            label=self.target,
            problem_type=ag.constants.REGRESSION,
            eval_metric=self.TIMESERIES_METRIC_TO_TABULAR_METRIC.get(self.eval_metric),
        )

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: int = None,
        **kwargs,
    ) -> None:
        self._check_fit_params()
        start_time = time.time()
        if self.tabular_predictor._learner.is_fit:
            raise AssertionError(f"{self.name} predictor has already been fit!")
        verbosity = kwargs.get("verbosity", 2)
        self._target_lag_indices = np.array(get_lags_for_frequency(train_data.freq), dtype=np.int64)
        self._past_covariates_lag_indices = self._target_lag_indices
        self._known_covariates_lag_indices = np.concatenate([[0], self._target_lag_indices])
        self._time_features = time_features_from_frequency_str(train_data.freq)

        train_data, _ = self._apply_scaling(train_data)
        train_df = self._get_features_dataframe(
            train_data,
            static_features=train_data.static_features,
            masked=self.features_are_masked,
        )
        # Remove features that are completely missing in the training set
        train_df.dropna(axis=1, how="all", inplace=True)
        self._available_features = train_df.columns

        model_params = self._get_model_params()
        tabular_hyperparameters = model_params.get("tabular_hyperparameters", self.default_tabular_hyperparameters)
        max_train_size = model_params.get("max_train_size", 1_000_000)

        if len(train_df) > max_train_size:
            train_df = train_df.sample(max_train_size)
        logger.debug(f"Generated training dataframe with shape {train_df.shape}")

        if val_data is not None:
            if val_data.freq != train_data.freq:
                raise ValueError(
                    f"train_data and val_data must have the same freq (received {train_data.freq} and {val_data.freq})"
                )
            val_data, _ = self._apply_scaling(val_data)
            val_df = self._get_features_dataframe(
                val_data,
                static_features=val_data.static_features,
                masked=self.features_are_masked,
                last_k_values=self.prediction_length,
            )
            val_df = val_df[self._available_features]

            if len(val_df) > max_train_size:
                val_df = val_df.sample(max_train_size)

            logger.debug(f"Generated validation dataframe with shape {val_df.shape}")
        else:
            logger.warning(
                f"No val_data was provided to {self.name}. "
                "TabularPredictor will generate a validation set without respecting the temporal ordering."
            )
            val_df = None

        time_elapsed = time.time() - start_time
        autogluon_logger = logging.getLogger("autogluon")
        logging_level = autogluon_logger.level
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tabular_predictor.fit(
                train_data=train_df,
                tuning_data=val_df,
                time_limit=time_limit - time_elapsed if time_limit else None,
                hyperparameters=tabular_hyperparameters,
                verbosity=verbosity - 2,
            )
        residuals = train_df[self.target] - self.tabular_predictor.predict(train_df)
        for q in self.quantile_levels:
            self.quantile_adjustments[q] = np.quantile(residuals, q)
        # Logger level is changed inside .fit(), restore to the initial value
        autogluon_logger.setLevel(logging_level)

    def _get_features_dataframe(
        self,
        data: pd.DataFrame,
        static_features: pd.DataFrame,
        last_k_values: Optional[int] = None,
        masked: bool = False,
        num_cpus: int = -1,
    ) -> pd.DataFrame:
        """Generate a feature matrix used by TabularPredictor.

        Parameters
        ----------
        data : TimeSeriesDataFrame
            Dataframe containing features derived from time index & past time series values, as well as the target.
        last_k_values: int, optional
            If given, features will be generated only for the last `last_k_values` timesteps of each time series.
        """

        def apply_mask(array: np.ndarray, num_hidden: np.ndarray, lag_indices: np.ndarray) -> pd.DataFrame:
            """Apply a mask that mimics the situation at prediction time when target/covariates are unknown during the
            forecast horizon.

            Parameters
            ----------
            array
                Array to mask, shape [N, len(lag_indices)]
            num_hidden
                Number of entries hidden in each row, shape [N]
            lag_indices
                Lag indices used to construct the dataframe

            Returns
            -------
            masked_array
                Array with the masking applied, shape [N, D * len(lag_indices)]


            For example, given the following inputs

            array = [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
            num_hidden = [6, 0, 1]
            lag_indices = [1, 2, 5, 10]
            num_columns = 1

            The resulting masked output will be

            masked_array = [
                [NaN, NaN, NaN, 1],
                [1, 1, 1, 1],
                [NaN, 1, 1, 1],
            ]

            """
            mask = num_hidden[:, None] >= lag_indices[None]  # shape [len(num_hidden), len(lag_indices)]
            array[mask] = np.nan
            return array

        def get_lags(
            ts: np.ndarray,
            lag_indices: np.ndarray,
            prediction_length: int,
            last_k_values: Optional[int] = None,
            mask: bool = False,
        ) -> np.ndarray:
            """Generate the matrix of lag features for a single time series.

            Parameters
            ----------
            ts
                Array with target or covariate values, shape [N]
            lag_indices
                Array with the lag indices to use for feature generation.
            prediction_length
                Length of the forecast horizon.
            last_k_values
                Maximum number of rows to include in the feature matrix.
                If last_k_values < len(ts), the lag features will be generated only
                for the *last* last_k_values entries of ts.
            mask
                If True, a mask will be applied to some entries of the feature matrix,
                mimicking the behavior at prediction time, when the ts values are not
                known during the forecast horizon.

            Returns
            -------
            features
                Array with lag features, shape [min(N, last_k_values), len(lag_indices)]
            """
            num_rows = len(ts) if last_k_values is None else min(last_k_values, len(ts))
            features = np.full([num_rows, len(lag_indices)], fill_value=np.nan)
            for i in range(1, num_rows + 1):
                target_idx = len(ts) - i
                selected_lags = lag_indices[lag_indices <= target_idx]
                features[num_rows - i, np.arange(len(selected_lags))] = ts[target_idx - selected_lags]
            if mask:
                num_windows = (len(ts) - 1) // prediction_length
                # We don't hide any past values for the first `remainder` values, otherwise the features will be all empty
                remainder = len(ts) - num_windows * prediction_length
                num_hidden = np.concatenate([np.zeros(remainder), np.tile(np.arange(prediction_length), num_windows)])
                features = apply_mask(features, num_hidden[-num_rows:], lag_indices)
            return features

        def get_lag_features(
            all_series: List[np.ndarray],
            lag_indices: np.ndarray,
            prediction_length: int,
            last_k_values: int,
            mask: bool,
            name: str,
        ):
            """Generate lag features for all time series in the dataset.

            See the docstring of get_lags for the description of the parameters.
            """
            # TODO: Expose n_jobs to the user as a hyperparameter
            if num_cpus == 1:
                lags_per_item = [
                    get_lags(
                        ts,
                        lag_indices,
                        prediction_length=prediction_length,
                        last_k_values=last_k_values,
                        mask=mask,
                    )
                    for ts in all_series
                ]
            else:
                lags_per_item = Parallel(n_jobs=num_cpus)(
                    delayed(get_lags)(
                        ts,
                        lag_indices,
                        prediction_length=prediction_length,
                        last_k_values=last_k_values,
                        mask=mask,
                    )
                    for ts in all_series
                )
            features = np.concatenate(lags_per_item)
            return pd.DataFrame(features, columns=[f"{name}_lag_{idx}" for idx in lag_indices])

        df = pd.DataFrame(data)

        all_series = [ts for _, ts in df.droplevel(TIMESTAMP).groupby(level=ITEMID, sort=False)]

        feature_dfs = []
        for column_name in df.columns:
            if column_name == self.target:
                mask = masked
                lag_indices = self._target_lag_indices
            elif column_name in self.metadata.past_covariates_real:
                mask = masked
                lag_indices = self._past_covariates_lag_indices
            elif column_name in self.metadata.known_covariates_real:
                mask = False
                lag_indices = self._known_covariates_lag_indices
            else:
                raise ValueError(f"Unexpected column {column_name} is not among target or covariates.")

            feature_dfs.append(
                get_lag_features(
                    [ts[column_name].to_numpy() for ts in all_series],
                    lag_indices=lag_indices,
                    prediction_length=self.prediction_length,
                    last_k_values=last_k_values,
                    mask=mask,
                    name=column_name,
                )
            )

        target_with_index = df[self.target]
        if last_k_values is not None:
            target_with_index = target_with_index.groupby(level=ITEMID, sort=False).tail(last_k_values)
        feature_dfs.append(target_with_index.reset_index(drop=True))

        timestamps = target_with_index.index.get_level_values(level=TIMESTAMP)
        feature_dfs.append(
            pd.DataFrame({time_feat.__name__: time_feat(timestamps) for time_feat in self._time_features})
        )

        features = pd.concat(feature_dfs, axis=1)

        if static_features is not None:
            features.index = target_with_index.index.get_level_values(level=ITEMID)
            features = pd.merge(features, static_features, how="left", on=ITEMID, suffixes=(None, "_static_feat"))

        features.reset_index(inplace=True, drop=True)
        return features

    def predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        normalized_data, scale_per_item = self._apply_scaling(data)
        predictions = self._predict_core(normalized_data, known_covariates=known_covariates)
        predictions = self._undo_scaling(predictions, scale_per_item)
        return self._postprocess_predictions(predictions)

    def _predict_core(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
    ) -> TimeSeriesDataFrame:
        raise NotImplementedError

    def _apply_scaling(self, data: TimeSeriesDataFrame) -> Tuple[TimeSeriesDataFrame, pd.Series]:
        # scale_per_item = data[self.target].abs().groupby(level=ITEMID, sort=False).mean()
        # normalized_data = data.copy()
        # normalized_data[self.target] = normalized_data[self.target] / scale_per_item
        # return normalized_data, scale_per_item
        return data, None

    def _undo_scaling(self, normalized_data: TimeSeriesDataFrame, scale_per_item: pd.Series) -> TimeSeriesDataFrame:
        data = normalized_data
        # for col in data.columns:
        #     data[col] = data[col] * scale_per_item
        return data

    def _extend_index(
        self, data: TimeSeriesDataFrame, known_covariates: Optional[TimeSeriesDataFrame] = None
    ) -> TimeSeriesDataFrame:
        """Add self.prediction_length many time steps with dummy values to each timeseries in the dataset."""

        def extend_single_time_series(group):
            offset = pd.tseries.frequencies.to_offset(data.freq)
            cutoff = group.index.get_level_values(TIMESTAMP)[-1]
            new_index = pd.date_range(cutoff + offset, freq=offset, periods=self.prediction_length).rename(TIMESTAMP)
            new_values = np.full([self.prediction_length], fill_value=np.nan)
            new_df = pd.DataFrame(new_values, index=new_index, columns=[self.target])
            return pd.concat([group.droplevel(ITEMID), new_df])

        if known_covariates is not None:
            extended_data = pd.concat([data, known_covariates])
        else:
            extended_data = data.groupby(level=ITEMID, sort=False).apply(extend_single_time_series)

        extended_data.static_features = data.static_features
        return extended_data

    def _postprocess_predictions(self, predictions: pd.DataFrame) -> TimeSeriesDataFrame:
        for q in self.quantile_levels:
            predictions[str(q)] = predictions["mean"] + self.quantile_adjustments[q]
        return predictions


class AutoregressiveTabularModel(AbstractTabularModel):
    """Predict future values one by one by calling tabular_predictor.predict at each time step."""

    features_are_masked = False

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: int = None,
        **kwargs,
    ) -> None:
        train_data = self._apply_differencing(train_data)
        train_data.groupby(level="item_id", sort=False).diff().abs().groupby(level="item_id", sort=False).mean()
        if val_data is not None:
            val_data = self._apply_differencing(val_data)
        return super()._fit(train_data=train_data, val_data=val_data, time_limit=time_limit, **kwargs)

    def _apply_differencing(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        # diffs = data.groupby(level=ITEMID, sort=False).diff(1)
        # # Keep the first value of each series to restore the original values afterwards
        # return diffs.fillna(data)
        return data

    def _undo_differencing(self, data_extended: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        # return data_extended.groupby(level=ITEMID, sort=False).cumsum()
        return data_extended

    def _predict_core(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
    ) -> TimeSeriesDataFrame:
        data = self._apply_differencing(data)
        data_extended = self._extend_index(data, known_covariates=known_covariates)
        last_index_per_item = data_extended.num_timesteps_per_item().cumsum() - 1
        for step in reversed(range(self.prediction_length)):
            if step == 0:
                data_up_to_current_step = data_extended
            else:
                data_up_to_current_step = data_extended.slice_by_timestep(None, -step)
            features = self._get_features_dataframe(
                data_up_to_current_step,
                static_features=data.static_features,
                masked=self.features_are_masked,
                last_k_values=1,
                num_cpus=1,
            )
            features = features[self._available_features]
            predictions_for_step = self.tabular_predictor.predict(features)
            data_extended[self.target].iloc[last_index_per_item - step] = predictions_for_step.values

        data_extended = self._undo_differencing(data_extended)
        predictions = data_extended.slice_by_timestep(-self.prediction_length, None)[self.target].rename("mean")
        return TimeSeriesDataFrame(predictions.to_frame())


class BatchTabularModel(AbstractTabularModel):
    """Predict all future values simultaneously with a single model."""

    features_are_masked = True

    def _predict_core(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
    ) -> TimeSeriesDataFrame:
        data_extended = self._extend_index(data, known_covariates=known_covariates)
        features = self._get_features_dataframe(
            data_extended,
            static_features=data.static_features,
            masked=self.features_are_masked,
            last_k_values=self.prediction_length,
        )
        features = features[self._available_features]

        predictions = self.tabular_predictor.predict(features).rename("mean").to_frame()

        preds_index = data_extended.slice_by_timestep(-self.prediction_length, None).index
        predictions.set_index(preds_index, inplace=True)
        return TimeSeriesDataFrame(predictions)


class MultiOutputTabularModel(AbstractTabularModel):
    """Train multiple tabular predictors - one for each time step in the forecast horizon."""

    pass
