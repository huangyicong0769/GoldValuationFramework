from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor


LOGGER = logging.getLogger(__name__)


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


class EnsemblePipeline:
    def __init__(self, models: list[Pipeline], weights: np.ndarray) -> None:
        self.models = models
        self.weights = weights

    def predict(self, X):  # noqa: N803
        preds = []
        for model in self.models:
            preds.append(np.asarray(model.predict(X), dtype=float))
        stacked = np.vstack(preds)
        return np.average(stacked, axis=0, weights=self.weights)


class FeatureWeighter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        gold_weight: float = 0.4,
        liquidity_weight: float = 1.8,
        debt_weight: float = 1.8,
        liquidity_component_weight: float = 1.3,
    ) -> None:
        self.gold_weight = gold_weight
        self.liquidity_weight = liquidity_weight
        self.debt_weight = debt_weight
        self.liquidity_component_weight = liquidity_component_weight
        self.feature_names_: list[str] | None = None
        self.weights_: np.ndarray | None = None

    def set_feature_names(self, feature_names: list[str]) -> "FeatureWeighter":
        self.feature_names_ = list(feature_names)
        return self

    def fit(self, X, y=None):  # noqa: N803
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        self.weights_ = self._build_weights()
        return self

    def transform(self, X):  # noqa: N803
        if self.weights_ is None:
            return X
        if isinstance(X, pd.DataFrame):
            return X.mul(self.weights_, axis=1)
        return X * self.weights_

    def _build_weights(self) -> np.ndarray | None:
        if not self.feature_names_:
            return None
        weights = np.ones(len(self.feature_names_), dtype=float)
        liquidity_components = {
            "M2SL",
            "TOTLL",
            "WALCL",
            "WRESBAL",
            "RRPONTSYD",
            "BOGMBASE",
            "GFDEBTN",
        }
        for idx, name in enumerate(self.feature_names_):
            base = name.split("_")[0].upper()
            lname = name.lower()
            if lname.startswith("gold"):
                weights[idx] = min(weights[idx], self.gold_weight)
            if lname.startswith("liquidity_index"):
                weights[idx] = max(weights[idx], self.liquidity_weight)
            if lname.startswith("debt_index"):
                weights[idx] = max(weights[idx], self.debt_weight)
            if base in liquidity_components:
                weights[idx] = max(weights[idx], self.liquidity_component_weight)
        return weights


@dataclass
class ModelResult:
    prediction: float
    last_actual: float
    mae: float
    rmse: float
    train_size: int
    test_size: int
    train_start: datetime | None
    train_end: datetime | None
    test_start: datetime | None
    test_end: datetime | None


@dataclass
class MultiTargetModelResult:
    predictions: dict[str, float]
    last_actuals: dict[str, float]
    metrics: dict[str, tuple[float, float]]
    train_size: int
    test_size: int
    train_start: datetime | None
    train_end: datetime | None
    test_start: datetime | None
    test_end: datetime | None


class GoldPriceModel:
    def __init__(
        self,
        pipeline: Pipeline | None = None,
        target_column: str = "gold",
        train_window: int | None = None,
    ) -> None:
        self.model = pipeline or Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ]
        )
        self.feature_top_n = 20
        self.mi_top_n = 60
        self.corr_threshold = 0.9
        self.half_life = 126  # trading days
        self.train_window = train_window
        self.target_transform = "none"
        self._min_price = 1e-6
        self.feature_names_: list[str] | None = None
        self.target_column = target_column
        self.refit_full_after_eval = True

    @staticmethod
    def _candidate_models(random_state: int = 42) -> list[tuple[str, Pipeline]]:
        candidates: list[tuple[str, Pipeline]] = [
            (
                "linear",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("weighter", FeatureWeighter()),
                        ("reg", LinearRegression()),
                    ]
                ),
            )
        ]

        for alpha in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
            candidates.append(
                (
                    f"ridge_{alpha}",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("weighter", FeatureWeighter()),
                            ("reg", Ridge(alpha=alpha, random_state=random_state)),
                        ]
                    ),
                )
            )

        for alpha, l1_ratio in ((0.0005, 0.1), (0.001, 0.2), (0.005, 0.3)):
            candidates.append(
                (
                    f"elasticnet_{alpha}_{l1_ratio}",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("weighter", FeatureWeighter()),
                            (
                                "reg",
                                ElasticNet(
                                    alpha=alpha,
                                    l1_ratio=l1_ratio,
                                    random_state=random_state,
                                    max_iter=5000,
                                ),
                            ),
                        ]
                    ),
                )
            )

        for epsilon in (1.1, 1.2, 1.25, 1.35, 1.5):
            candidates.append(
                (
                    f"huber_{epsilon}",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("weighter", FeatureWeighter()),
                            ("reg", HuberRegressor(epsilon=epsilon, max_iter=1000)),
                        ]
                    ),
                )
            )

        candidates.extend(
            [
                (
                    "gbr",
                    Pipeline(
                        [
                            ("scaler", "passthrough"),
                            ("weighter", FeatureWeighter()),
                            (
                                "reg",
                                GradientBoostingRegressor(
                                    random_state=random_state,
                                    n_estimators=400,
                                    learning_rate=0.04,
                                    max_depth=3,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "rf",
                    Pipeline(
                        [
                            ("scaler", "passthrough"),
                            ("weighter", FeatureWeighter()),
                            (
                                "reg",
                                RandomForestRegressor(
                                    n_estimators=500,
                                    max_depth=10,
                                    min_samples_leaf=2,
                                    random_state=random_state,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "extra_trees",
                    Pipeline(
                        [
                            ("scaler", "passthrough"),
                            ("weighter", FeatureWeighter()),
                            (
                                "reg",
                                ExtraTreesRegressor(
                                    n_estimators=600,
                                    max_depth=10,
                                    min_samples_leaf=2,
                                    random_state=random_state,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "hgb",
                    Pipeline(
                        [
                            ("scaler", "passthrough"),
                            ("weighter", FeatureWeighter()),
                            (
                                "reg",
                                HistGradientBoostingRegressor(
                                    max_depth=3,
                                    learning_rate=0.05,
                                    max_iter=500,
                                    l2_regularization=0.1,
                                    random_state=random_state,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "xgb",
                    Pipeline(
                        [
                            ("scaler", "passthrough"),
                            ("weighter", FeatureWeighter()),
                            (
                                "reg",
                                XGBRegressor(
                                    n_estimators=600,
                                    learning_rate=0.05,
                                    max_depth=4,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    reg_alpha=0.0,
                                    reg_lambda=1.0,
                                    objective="reg:squarederror",
                                    random_state=random_state,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )

        return candidates

    def train_and_predict(self, data: pd.DataFrame) -> ModelResult:
        LOGGER.info("Train model on %s rows", len(data))
        best_successful: list[tuple[Pipeline, float, float]] = []
        best_model: Pipeline | None = None
        best_mae = float("inf")
        best_rmse = float("inf")
        best_features: pd.DataFrame | None = None
        best_target: pd.Series | None = None
        best_train_size = 0
        best_test_size = 0
        best_train_start: datetime | None = None
        best_train_end: datetime | None = None
        best_test_start: datetime | None = None
        best_test_end: datetime | None = None
        best_top_n = self.feature_top_n
        best_transform = self.target_transform

        for transform in ("none", "log"):
            self.target_transform = transform
            for top_n in (30, 60, 90):
                self.feature_top_n = top_n
                features, target = self._prepare_features(data, horizon=1)

                if len(features) < 50:
                    continue

                x_train, x_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, shuffle=False
                )

                sample_weight = self._sample_weights(x_train)

                successful: list[tuple[Pipeline, float, float]] = []
                local_best_model: Pipeline | None = None
                local_best_mae = float("inf")
                local_best_rmse = float("inf")
                for name, candidate in self._candidate_models():
                    try:
                        weighter = candidate.named_steps.get("weighter")
                        if weighter is not None:
                            weighter.set_feature_names(list(features.columns))
                        candidate.fit(x_train, y_train, reg__sample_weight=sample_weight)
                        preds = np.asarray(candidate.predict(x_test))
                        y_test_eval, preds_eval = self._inverse_transform_target(
                            y_test, preds
                        )
                        mae = float(mean_absolute_error(y_test_eval, preds_eval))
                        rmse = float(np.sqrt(mean_squared_error(y_test_eval, preds_eval)))
                        LOGGER.info(
                            "Model %s (top_n=%s, target=%s) -> MAE %.4f RMSE %.4f",
                            name,
                            top_n,
                            transform,
                            mae,
                            rmse,
                        )
                        successful.append((candidate, rmse, mae))
                        if rmse < local_best_rmse:
                            local_best_rmse = rmse
                            local_best_mae = mae
                            local_best_model = candidate
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("Model %s failed: %s", name, exc)

                if local_best_model is None or not successful:
                    continue

                if local_best_rmse < best_rmse:
                    best_rmse = local_best_rmse
                    best_mae = local_best_mae
                    best_model = local_best_model
                    best_successful = successful
                    best_features = features
                    best_target = target
                    best_train_size = len(x_train)
                    best_test_size = len(x_test)
                    best_train_start = (
                        pd.Timestamp(x_train.index.min()).to_pydatetime()
                        if not x_train.empty
                        else None
                    )
                    best_train_end = (
                        pd.Timestamp(x_train.index.max()).to_pydatetime()
                        if not x_train.empty
                        else None
                    )
                    best_test_start = (
                        pd.Timestamp(x_test.index.min()).to_pydatetime()
                        if not x_test.empty
                        else None
                    )
                    best_test_end = (
                        pd.Timestamp(x_test.index.max()).to_pydatetime()
                        if not x_test.empty
                        else None
                    )
                    best_top_n = top_n
                    best_transform = transform

        if best_model is None or best_features is None or not best_successful:
            raise RuntimeError("All candidate models failed")

        self.feature_top_n = best_top_n
        self.target_transform = best_transform

        if self.refit_full_after_eval and best_target is not None:
            sample_weight_full = self._sample_weights(best_features)
            refit_successful: list[tuple[Pipeline, float, float]] = []
            for candidate, rmse, mae in best_successful:
                try:
                    weighter = candidate.named_steps.get("weighter")
                    if weighter is not None:
                        weighter.set_feature_names(list(best_features.columns))
                    candidate.fit(
                        best_features,
                        best_target,
                        reg__sample_weight=sample_weight_full,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Refit model failed: %s", exc)
                refit_successful.append((candidate, rmse, mae))
            best_successful = refit_successful

        rmses = np.array([item[1] for item in best_successful], dtype=float)
        weights = 1.0 / (rmses + 1e-6)
        weights = weights / weights.sum()
        ensemble = EnsemblePipeline([item[0] for item in best_successful], weights)

        self.model = ensemble
        self.feature_names_ = list(best_features.columns)
        setattr(self.model, "_feature_names", list(best_features.columns))
        setattr(self.model, "_target_transform", self.target_transform)
        setattr(best_model, "_feature_names", list(best_features.columns))
        setattr(best_model, "_target_transform", self.target_transform)
        setattr(self.model, "_best_single", best_model)
        last_features = best_features.iloc[[-1]]
        raw_pred = float(np.asarray(self.model.predict(last_features))[0])
        prediction = float(self._inverse_transform_prediction(raw_pred))
        last_actual = float(data[self.target_column].iloc[-1])

        return ModelResult(
            prediction=prediction,
            last_actual=last_actual,
            mae=best_mae,
            rmse=best_rmse,
            train_size=best_train_size,
            test_size=best_test_size,
            train_start=best_train_start,
            train_end=best_train_end,
            test_start=best_test_start,
            test_end=best_test_end,
        )

    def train_and_predict_horizon(self, data: pd.DataFrame, horizon: int) -> ModelResult | None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        features, target = self._prepare_features(data, horizon=horizon)

        if len(features) < 50:
            return None

        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, shuffle=False
        )
        sample_weight = self._sample_weights(x_train)

        successful: list[tuple[Pipeline, float, float]] = []
        best_model: Pipeline | None = None
        best_mae = float("inf")
        best_rmse = float("inf")
        candidates = [
            (name, model)
            for name, model in self._candidate_models()
            if name.startswith(("linear", "ridge", "elasticnet", "huber"))
        ]
        for name, candidate in candidates:
            try:
                weighter = candidate.named_steps.get("weighter")
                if weighter is not None:
                    weighter.set_feature_names(list(features.columns))
                candidate.fit(x_train, y_train, reg__sample_weight=sample_weight)
                preds = np.asarray(candidate.predict(x_test))
                y_test_eval, preds_eval = self._inverse_transform_target(y_test, preds)
                mae = float(mean_absolute_error(y_test_eval, preds_eval))
                rmse = float(np.sqrt(mean_squared_error(y_test_eval, preds_eval)))
                successful.append((candidate, rmse, mae))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_mae = mae
                    best_model = candidate
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Horizon model %s failed: %s", name, exc)

        if best_model is None or not successful:
            return None

        rmses = np.array([item[1] for item in successful], dtype=float)
        weights = 1.0 / (rmses + 1e-6)
        weights = weights / weights.sum()
        ensemble = EnsemblePipeline([item[0] for item in successful], weights)

        self.model = ensemble
        self.feature_names_ = list(features.columns)
        setattr(self.model, "_feature_names", list(features.columns))
        setattr(self.model, "_target_transform", self.target_transform)
        if best_model is not None:
            setattr(best_model, "_feature_names", list(features.columns))
            setattr(best_model, "_target_transform", self.target_transform)
        setattr(self.model, "_best_single", best_model)
        last_features = features.iloc[[-1]]
        raw_pred = float(np.asarray(self.model.predict(last_features))[0])
        prediction = float(self._inverse_transform_prediction(raw_pred))
        last_actual = float(data[self.target_column].iloc[-1])

        train_start = (
            pd.Timestamp(x_train.index.min()).to_pydatetime() if not x_train.empty else None
        )
        train_end = (
            pd.Timestamp(x_train.index.max()).to_pydatetime() if not x_train.empty else None
        )
        test_start = (
            pd.Timestamp(x_test.index.min()).to_pydatetime() if not x_test.empty else None
        )
        test_end = (
            pd.Timestamp(x_test.index.max()).to_pydatetime() if not x_test.empty else None
        )

        return ModelResult(
            prediction=prediction,
            last_actual=last_actual,
            mae=best_mae,
            rmse=best_rmse,
            train_size=len(x_train),
            test_size=len(x_test),
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

    def predict_latest(self, data: pd.DataFrame, meta: dict) -> ModelResult:
        LOGGER.info("Reuse cached model for prediction")
        last_features = data.drop(columns=[self.target_column]).iloc[[-1]]
        feature_names = None
        if hasattr(self.model, "_feature_names"):
            feature_names = list(getattr(self.model, "_feature_names"))
        elif self.feature_names_:
            feature_names = list(self.feature_names_)
        if feature_names:
            last_features = last_features.reindex(columns=feature_names, fill_value=0.0)
        raw_pred = float(np.asarray(self.model.predict(last_features))[0])
        prediction = float(self._inverse_transform_prediction(raw_pred))
        last_actual = float(data[self.target_column].iloc[-1])
        return ModelResult(
            prediction=prediction,
            last_actual=last_actual,
            mae=float(meta.get("mae", 0.0)),
            rmse=float(meta.get("rmse", 0.0)),
            train_size=int(meta.get("train_size", 0)),
            test_size=int(meta.get("test_size", 0)),
            train_start=_parse_datetime(meta.get("train_start")),
            train_end=_parse_datetime(meta.get("train_end")),
            test_start=_parse_datetime(meta.get("test_start")),
            test_end=_parse_datetime(meta.get("test_end")),
        )

    def _prepare_features(self, data: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
        if self.target_column not in data.columns:
            raise KeyError(f"Target column {self.target_column} not found in data")
        target = data[self.target_column].shift(-horizon)
        features = data.drop(columns=[self.target_column]).iloc[:-horizon]
        target = target.iloc[:-horizon]
        if self.target_transform == "log":
            target = pd.Series(
                np.log(target.clip(lower=self._min_price)),
                index=target.index,
            )

        if self.train_window and len(features) > self.train_window:
            features = features.tail(self.train_window)
            target = target.tail(self.train_window)

        if features.empty:
            return features, target

        numeric_features = features.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            stds = numeric_features.std()
            numeric_features = numeric_features.loc[:, stds > 1e-8]

        correlations = numeric_features.apply(lambda col: col.corr(target))
        correlations = correlations.abs().dropna()
        if not correlations.empty:
            order = np.argsort(correlations.values)[::-1]
            correlations = correlations.iloc[order]

        selected = list(correlations.head(self.feature_top_n).index)

        must_keep = [
            "liquidity_index",
            "debt_index",
            "neutral_balance",
            "WALCL",
            "M2SL",
            "GFDEBTN",
            "TOTLL",
        ]
        for name in must_keep:
            if name in features.columns and name not in selected:
                selected.append(name)

        if selected:
            filtered = features[selected]
        else:
            filtered = features

        if filtered.shape[1] > 1:
            corr_matrix = filtered.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            drop_cols = [
                column for column in upper.columns if any(upper[column] > self.corr_threshold)
            ]
            filtered = filtered.drop(columns=drop_cols)

        filtered = self._select_by_mutual_info(filtered, target, top_n=self.mi_top_n)

        return filtered, target

    def train_and_predict_multi(
        self, data: pd.DataFrame, target_columns: list[str]
    ) -> MultiTargetModelResult:
        LOGGER.info("Train multi-target model on %s rows", len(data))
        targets = [col for col in target_columns if col in data.columns]
        if not targets:
            raise ValueError("No valid target columns found in data")
        target_df = data[targets].shift(-1).iloc[:-1]
        features = data.drop(columns=targets).iloc[:-1]

        if self.target_transform == "log":
            target_df = np.log(target_df.clip(lower=self._min_price))
            target_df = pd.DataFrame(target_df, index=features.index, columns=targets)

        if self.train_window and len(features) > self.train_window:
            features = features.tail(self.train_window)
            target_df = target_df.tail(self.train_window)

        if len(features) < 50:
            raise ValueError("Not enough samples to train model")

        numeric_features = features.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            stds = numeric_features.std()
            numeric_features = numeric_features.loc[:, stds > 1e-8]

        correlations = pd.Series(0.0, index=numeric_features.columns, dtype=float)
        for target_col in targets:
            corr_series = numeric_features.corrwith(target_df[target_col]).abs()
            correlations = correlations.add(corr_series, fill_value=0.0)
        if not correlations.empty:
            correlations = correlations / float(len(targets))
            order = np.argsort(correlations.to_numpy())[::-1]
            correlations = correlations.iloc[order]

        selected = list(correlations.head(self.feature_top_n).index)

        must_keep = [
            "liquidity_index",
            "debt_index",
            "neutral_balance",
            "WALCL",
            "M2SL",
            "GFDEBTN",
            "TOTLL",
        ]
        for name in must_keep:
            if name in features.columns and name not in selected:
                selected.append(name)

        if selected:
            filtered = features[selected]
        else:
            filtered = features

        if filtered.shape[1] > 1:
            corr_matrix = filtered.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            drop_cols = [
                column for column in upper.columns if any(upper[column] > self.corr_threshold)
            ]
            filtered = filtered.drop(columns=drop_cols)

        filtered = self._select_by_mutual_info_multi(filtered, target_df, top_n=self.mi_top_n)

        x_train, x_test, y_train, y_test = train_test_split(
            filtered, target_df, test_size=0.2, shuffle=False
        )
        sample_weight = self._sample_weights(x_train)

        best_model: Pipeline | None = None
        best_rmse = float("inf")
        best_mae = float("inf")
        best_multi: Pipeline | None = None
        best_successful: list[tuple[Pipeline, float, float]] = []

        for name, candidate in self._candidate_models():
            try:
                weighter = candidate.named_steps.get("weighter")
                if weighter is not None:
                    weighter.set_feature_names(list(filtered.columns))
                steps = []
                for step_name, step in candidate.steps:
                    if step_name == "reg":
                        step = MultiOutputRegressor(step)
                    steps.append((step_name, step))
                multi_candidate = Pipeline(steps)
                multi_candidate.fit(x_train, y_train, reg__sample_weight=sample_weight)
                preds = np.asarray(multi_candidate.predict(x_test))
                if self.target_transform == "log":
                    y_test_eval = np.exp(y_test.to_numpy())
                    preds_eval = np.exp(preds)
                else:
                    y_test_eval = y_test.to_numpy()
                    preds_eval = preds

                rmses = []
                maes = []
                for idx in range(len(targets)):
                    mae = float(mean_absolute_error(y_test_eval[:, idx], preds_eval[:, idx]))
                    rmse = float(np.sqrt(mean_squared_error(y_test_eval[:, idx], preds_eval[:, idx])))
                    rmses.append(rmse)
                    maes.append(mae)
                avg_rmse = float(np.mean(rmses))
                avg_mae = float(np.mean(maes))
                LOGGER.info("Multi model %s -> MAE %.4f RMSE %.4f", name, avg_mae, avg_rmse)
                best_successful.append((multi_candidate, avg_rmse, avg_mae))
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_mae = avg_mae
                    best_model = multi_candidate
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Multi model %s failed: %s", name, exc)

        if best_model is None or not best_successful:
            raise RuntimeError("All candidate models failed")

        rmses = np.array([item[1] for item in best_successful], dtype=float)
        weights = 1.0 / (rmses + 1e-6)
        weights = weights / weights.sum()
        ensemble = EnsemblePipeline([item[0] for item in best_successful], weights)

        self.model = ensemble
        self.feature_names_ = list(filtered.columns)
        setattr(self.model, "_feature_names", list(filtered.columns))
        setattr(self.model, "_target_transform", self.target_transform)
        if best_model is not None:
            setattr(best_model, "_feature_names", list(filtered.columns))
            setattr(best_model, "_target_transform", self.target_transform)
        setattr(self.model, "_best_single", best_model)

        last_features = filtered.iloc[[-1]]
        raw_pred = np.asarray(self.model.predict(last_features))
        if raw_pred.ndim > 1:
            raw_pred = raw_pred[0]
        if self.target_transform == "log":
            raw_pred = np.exp(raw_pred)
        prediction_map = {name: float(raw_pred[idx]) for idx, name in enumerate(targets)}
        last_actuals = {name: float(data[name].iloc[-1]) for name in targets}

        metrics: dict[str, tuple[float, float]] = {}
        best_single = getattr(self.model, "_best_single", None)
        if isinstance(best_single, Pipeline):
            best_reg = best_single.named_steps.get("reg")
        else:
            best_reg = None
        if isinstance(best_reg, MultiOutputRegressor) and isinstance(best_single, Pipeline):
            best_preds = np.asarray(best_single.predict(x_test))
            if self.target_transform == "log":
                y_test_eval = np.exp(y_test.to_numpy())
                best_preds = np.exp(best_preds)
            else:
                y_test_eval = y_test.to_numpy()
            for idx, name in enumerate(targets):
                mae = float(mean_absolute_error(y_test_eval[:, idx], best_preds[:, idx]))
                rmse = float(np.sqrt(mean_squared_error(y_test_eval[:, idx], best_preds[:, idx])))
                metrics[name] = (mae, rmse)
        else:
            for name in targets:
                metrics[name] = (best_mae, best_rmse)

        if self.refit_full_after_eval:
            sample_weight_full = self._sample_weights(filtered)
            refit_successful: list[tuple[Pipeline, float, float]] = []
            for candidate, rmse, mae in best_successful:
                try:
                    weighter = candidate.named_steps.get("weighter")
                    if weighter is not None:
                        weighter.set_feature_names(list(filtered.columns))
                    candidate.fit(
                        filtered,
                        target_df,
                        reg__sample_weight=sample_weight_full,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Refit multi-target model failed: %s", exc)
                refit_successful.append((candidate, rmse, mae))
            best_successful = refit_successful

        return MultiTargetModelResult(
            predictions=prediction_map,
            last_actuals=last_actuals,
            metrics=metrics,
            train_size=len(x_train),
            test_size=len(x_test),
            train_start=(
                pd.Timestamp(x_train.index.min()).to_pydatetime() if not x_train.empty else None
            ),
            train_end=(
                pd.Timestamp(x_train.index.max()).to_pydatetime() if not x_train.empty else None
            ),
            test_start=(
                pd.Timestamp(x_test.index.min()).to_pydatetime() if not x_test.empty else None
            ),
            test_end=(
                pd.Timestamp(x_test.index.max()).to_pydatetime() if not x_test.empty else None
            ),
        )

    @staticmethod
    def _select_by_mutual_info(
        features: pd.DataFrame, target: pd.Series, top_n: int
    ) -> pd.DataFrame:
        if top_n is None or features.shape[1] <= top_n:
            return features
        numeric = features.select_dtypes(include=[np.number])
        if numeric.empty:
            return features
        aligned_target = target.loc[numeric.index]
        try:
            mi_scores = mutual_info_regression(
                numeric.fillna(0.0), aligned_target, random_state=42
            )
        except ValueError:
            return features
        order = np.argsort(mi_scores)[::-1]
        selected = list(numeric.columns[order[:top_n]])
        return features[selected]

    @staticmethod
    def _select_by_mutual_info_multi(
        features: pd.DataFrame, targets: pd.DataFrame, top_n: int
    ) -> pd.DataFrame:
        if top_n is None or features.shape[1] <= top_n:
            return features
        numeric = features.select_dtypes(include=[np.number])
        if numeric.empty:
            return features
        aligned_targets = targets.loc[numeric.index]
        mi_total = np.zeros(numeric.shape[1], dtype=float)
        for col in aligned_targets.columns:
            try:
                mi = mutual_info_regression(
                    numeric.fillna(0.0), aligned_targets[col], random_state=42
                )
            except ValueError:
                continue
            mi_total += mi
        if np.all(mi_total == 0):
            return features
        order = np.argsort(mi_total)[::-1]
        selected = list(numeric.columns[order[:top_n]])
        return features[selected]

    def predict_latest_multi(
        self, data: pd.DataFrame, meta: dict, target_columns: list[str]
    ) -> MultiTargetModelResult:
        LOGGER.info("Reuse cached multi-target model for prediction")
        targets = [col for col in target_columns if col in data.columns]
        if not targets:
            raise ValueError("No valid target columns found in data")
        last_features = data.drop(columns=targets).iloc[[-1]]
        feature_names = None
        if hasattr(self.model, "_feature_names"):
            feature_names = list(getattr(self.model, "_feature_names"))
        elif self.feature_names_:
            feature_names = list(self.feature_names_)
        if feature_names:
            last_features = last_features.reindex(columns=feature_names, fill_value=0.0)
        raw_pred = np.asarray(self.model.predict(last_features))
        if raw_pred.ndim > 1:
            raw_pred = raw_pred[0]
        if getattr(self.model, "_target_transform", None) == "log":
            raw_pred = np.exp(raw_pred)
        prediction_map = {name: float(raw_pred[idx]) for idx, name in enumerate(targets)}
        last_actuals = {name: float(data[name].iloc[-1]) for name in targets}

        metrics: dict[str, tuple[float, float]] = {}
        target_metrics = meta.get("target_metrics")
        if isinstance(target_metrics, dict):
            for name in targets:
                vals = target_metrics.get(name)
                if (
                    isinstance(vals, (list, tuple))
                    and len(vals) == 2
                    and all(isinstance(x, (int, float)) for x in vals)
                ):
                    metrics[name] = (float(vals[0]), float(vals[1]))
        if not metrics:
            fallback_mae = float(meta.get("mae", 0.0))
            fallback_rmse = float(meta.get("rmse", 0.0))
            for name in targets:
                metrics[name] = (fallback_mae, fallback_rmse)

        return MultiTargetModelResult(
            predictions=prediction_map,
            last_actuals=last_actuals,
            metrics=metrics,
            train_size=int(meta.get("train_size", 0)),
            test_size=int(meta.get("test_size", 0)),
            train_start=_parse_datetime(meta.get("train_start")),
            train_end=_parse_datetime(meta.get("train_end")),
            test_start=_parse_datetime(meta.get("test_start")),
            test_end=_parse_datetime(meta.get("test_end")),
        )

    def _inverse_transform_prediction(self, value: float) -> float:
        if self.target_transform == "log":
            return float(np.exp(value))
        return float(value)

    def _inverse_transform_target(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.target_transform == "log":
            return np.exp(y_true.to_numpy()), np.exp(y_pred)
        return y_true.to_numpy(), y_pred

    def _sample_weights(self, x_train: pd.DataFrame) -> np.ndarray:
        if x_train.empty:
            return np.array([])
        n = len(x_train)
        if self.half_life <= 0:
            return np.ones(n)
        decay = np.log(2) / self.half_life
        ages = np.arange(n - 1, -1, -1)
        weights = np.exp(-decay * ages)
        return weights
