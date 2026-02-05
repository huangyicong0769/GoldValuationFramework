from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import logging
import numpy as np
from typing import Any, cast

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from config import Settings, DEFAULT_MACRO_FACTORS, DEFAULT_MARKET_FACTORS, FACTOR_DESCRIPTIONS
from data_sources import FREDClient, GoldDataHub, MetalsLiveClient, StooqClient, TwelveDataClient
from factor_engine import FactorBuilder, FactorDiscovery
from model import EnsemblePipeline, GoldPriceModel, ModelResult, MultiTargetModelResult
from storage import Storage


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    data_rows: int
    model: ModelResult
    realtime_price: float | None
    realtime_time: datetime | None
    realtime_target_date: date | None
    realtime_target_horizon: int | None
    realtime_bias: float | None
    adjusted_prediction: float | None
    adjusted_diff: float | None
    adjusted_prediction_for_realtime: float | None
    adjusted_diff_for_realtime: float | None
    macro_factors: list[str]
    market_factors: list[str]
    last_data_time: datetime
    next_data_time: datetime
    frequency: str
    gold_history_source: str
    daily_target: float | None
    next_trading_date: date | None
    horizon_predictions: list[tuple[int, date, float]]
    model_top_features: list[tuple[str, float]]
    horizon_top_features: list[tuple[int, list[tuple[str, float]]]]
    last_close: float
    last_close_time: datetime
    target_column: str
    last_target: float


@dataclass
class MultiTargetPipelineResult:
    data_rows: int
    model: MultiTargetModelResult
    realtime_price: float | None
    realtime_time: datetime | None
    realtime_target_date: date | None
    realtime_target_horizon: int | None
    macro_factors: list[str]
    market_factors: list[str]
    last_data_time: datetime
    next_data_time: datetime
    frequency: str
    gold_history_source: str
    target_columns: list[str]
    last_targets: dict[str, float]
    last_close: float
    last_close_time: datetime
    adjusted_predictions: dict[str, float]
    adjusted_diffs: dict[str, float]
    adjusted_predictions_for_realtime: dict[str, float]
    adjusted_diffs_for_realtime: dict[str, float]
    horizon_predictions: dict[str, list[tuple[int, date, float]]]
    horizon_top_features: list[tuple[int, list[tuple[str, float]]]]


class GoldValuationPipeline:
    def __init__(self, settings: Settings) -> None:
        fred = (
            FREDClient(
                settings.fred_api_key,
                settings.http_timeout,
                cache_dir=settings.fred_cache_dir,
                cache_ttl_days=settings.fred_cache_ttl_days,
                retries=settings.http_retries,
                backoff=settings.http_backoff,
            )
            if settings.fred_api_key
            else None
        )
        twelve = (
            TwelveDataClient(
                settings.twelve_data_api_key,
                settings.http_timeout,
                cache_dir=settings.twelve_cache_dir,
                cache_ttl_minutes=settings.twelve_cache_ttl_minutes,
                retries=settings.http_retries,
                backoff=settings.http_backoff,
            )
            if settings.twelve_data_api_key
            else None
        )
        self.hub = GoldDataHub(
            fred=fred,
            stooq=StooqClient(settings.http_timeout, settings.http_retries, settings.http_backoff),
            metals=MetalsLiveClient(settings.http_timeout, settings.http_retries, settings.http_backoff),
            twelve=twelve,
            gold_history_source=settings.gold_history_source,
            gold_history_symbol=settings.twelve_data_symbol,
        )
        self.discovery = FactorDiscovery(macro_keywords=[])
        self.builder = FactorBuilder(self.hub)
        self.storage = Storage(settings.artifact_dir)
        self.symbol = settings.twelve_data_symbol
        self.market_factors = settings.market_factors or DEFAULT_MARKET_FACTORS
        self.target_column = settings.target_column
        self.target_columns = settings.target_columns or [settings.target_column]
        self.train_window = settings.train_window
        self.refit_full_after_eval = settings.refit_full_after_eval
        self.use_realtime_last_close = settings.use_realtime_last_close

    def run(
        self,
        start: date | None,
        end: date | None,
        frequency: str = "day",
        horizons: list[int] | None = None,
        target_override: str | None = None,
    ) -> PipelineResult:
        LOGGER.info("Pipeline started")
        frequency = frequency.lower()
        if frequency not in {"day", "1min"}:
            raise ValueError("frequency must be 'day' or '1min'")
        target_column = (target_override or self.target_column) if frequency == "day" else "gold"
        self.storage = Storage(self.storage.base_dir, f"{frequency}_{target_column}")
        macro_factors: list[str] = []
        market_factors: list[str] = []
        factors_changed = False
        cached_factors = self.storage.load_factors()
        if frequency == "day":
            cached_macro = (cached_factors or {}).get("macro_factors", DEFAULT_MACRO_FACTORS)
            cached_market = (cached_factors or {}).get("market_factors", self.market_factors)
            cached_macro_defaults = (cached_factors or {}).get("macro_defaults", DEFAULT_MACRO_FACTORS)
            cached_market_defaults = (cached_factors or {}).get("market_defaults", self.market_factors)

            macro_factors = cached_macro
            market_factors = cached_market

            macro_changed = set(cached_macro_defaults) != set(DEFAULT_MACRO_FACTORS)
            market_changed = set(cached_market_defaults) != set(self.market_factors)

            if macro_changed:
                LOGGER.info("Default macro factors changed, refresh macro cache")
                search_func = (
                    self.hub.fred.search_series
                    if self.hub.fred and self.discovery.macro_keywords
                    else None
                )
                macro_factors = self.discovery.discover_macro(
                    DEFAULT_MACRO_FACTORS, search_func=search_func
                )
                factors_changed = True

            if market_changed:
                LOGGER.info("Default market factors changed, refresh market cache")
                market_factors = self.discovery.discover_market(self.market_factors)
                factors_changed = True

            if not cached_factors or macro_changed or market_changed:
                self.storage.save_factors(
                    macro_factors,
                    market_factors,
                    macro_defaults=DEFAULT_MACRO_FACTORS,
                    market_defaults=self.market_factors,
                )
                LOGGER.info("Factors cached")
            else:
                LOGGER.info("Use cached factors")
        if frequency != "day":
            macro_factors = []
            market_factors = []
            LOGGER.info("Intraday mode: skip macro/market factors")

        meta_for_cache = self.storage.load_meta()

        cached_raw = self.storage.load_dataset()
        if factors_changed:
            LOGGER.info("Factors changed, rebuild dataset")
            cached_raw = None
        if cached_raw is not None and meta_for_cache:
            cached_source = meta_for_cache.get("gold_history_source")
            if cached_source and cached_source != self.hub.gold_history_source:
                LOGGER.info(
                    "Gold history source changed (%s -> %s), rebuild dataset",
                    cached_source,
                    self.hub.gold_history_source,
                )
                cached_raw = None
        data_end = end or date.today()
        raw = None
        rebuild_cache = False
        if cached_raw is not None:
            LOGGER.info("Use cached dataset with last date %s", cached_raw.index.max().date())
            cached_start = cached_raw.index.min().date()
            cached_end = cached_raw.index.max().date()
            desired_start = start or cached_start
            if data_end <= cached_end:
                if desired_start < cached_start:
                    LOGGER.info("Requested start before cache range, rebuild dataset")
                    rebuild_cache = True
                else:
                    raw = cached_raw.loc[pd.Timestamp(desired_start) : pd.Timestamp(data_end)]
            else:
                update_start = cached_end
                LOGGER.info("Incremental update from %s to %s", update_start, data_end)
                if frequency == "day":
                    fresh = self.builder.build_dataset(macro_factors, market_factors, update_start, data_end)
                else:
                    fresh = self.hub.fetch_gold_intraday(update_start, data_end, self.symbol).to_frame()
                raw = (
                    pd.concat([cached_raw, fresh])
                    .sort_index()
                    .loc[lambda frame: ~frame.index.duplicated(keep="last")]
                )
                raw = raw.ffill().dropna()
        if raw is None or rebuild_cache:
            if frequency == "day":
                raw = self.builder.build_dataset(macro_factors, market_factors, start, data_end)
            else:
                raw = self.hub.fetch_gold_intraday(start, data_end, self.symbol).to_frame()
        if raw is None:
            raise RuntimeError("Failed to build dataset")
        realtime_for_update: float | None = None
        realtime_time: datetime | None = None
        last_idx = raw.index.max()
        last_idx_time = pd.Timestamp(last_idx).to_pydatetime() if last_idx is not None else None
        if frequency == "day" and self.use_realtime_last_close:
            realtime_quote = self.hub.fetch_gold_spot_with_time(self.symbol)
            if realtime_quote is not None:
                realtime_for_update, realtime_time = realtime_quote
                if (
                    last_idx is not None
                    and realtime_time is not None
                    and last_idx_time is not None
                    and realtime_time.date() == last_idx_time.date()
                ):
                    for col in ("gold_open", "gold_high", "gold_low", "gold"):
                        if col in raw.columns:
                            raw.loc[last_idx, col] = realtime_for_update
        self.storage.save_dataset(raw)

        if frequency == "day":
            available_macro = [name for name in macro_factors if name in raw.columns]
            available_market = [name for name in market_factors if name in raw.columns]
            if available_macro != macro_factors or available_market != market_factors:
                LOGGER.info("Drop unavailable factors from cache")
                macro_factors = available_macro
                market_factors = available_market
                self.storage.save_factors(
                    macro_factors,
                    market_factors,
                    macro_defaults=DEFAULT_MACRO_FACTORS,
                    market_defaults=self.market_factors,
                )
        features = self.builder.build_features(raw, target_column=target_column)
        model_pipeline = self.storage.load_model()
        meta = self.storage.load_meta()
        last_close_time = pd.Timestamp(raw.index.max()).to_pydatetime()
        last_close = float(raw["gold"].iloc[-1]) if "gold" in raw.columns else float(raw[target_column].iloc[-1])
        last_target = float(raw[target_column].iloc[-1])
        last_data_time = last_close_time
        if frequency == "1min":
            next_data_time = last_data_time + pd.Timedelta(minutes=1)
        else:
            next_data_time = (pd.Timestamp(last_data_time) + pd.tseries.offsets.BDay(1)).to_pydatetime()
        reuse_ok = False
        if model_pipeline is not None and meta and meta.get("last_data_time") == last_data_time.isoformat():
            feature_columns = meta.get("feature_columns")
            cached_target = meta.get("target_column")
            if isinstance(feature_columns, list) and list(features.columns) == feature_columns:
                if cached_target == target_column:
                    reuse_ok = True
            else:
                LOGGER.info("Feature columns changed, retrain model")
        if reuse_ok:
            if not meta or any(
                meta.get(field) is None
                for field in ("train_start", "train_end", "test_start", "test_end")
            ):
                LOGGER.info("Model meta missing train/test range, retrain model")
                reuse_ok = False

        model_top_features: list[tuple[str, float]] = []
        if reuse_ok:
            model = GoldPriceModel(
                model_pipeline,
                target_column=target_column,
                train_window=self.train_window,
            )
            model.refit_full_after_eval = self.refit_full_after_eval
            assert meta is not None
            model_result = model.predict_latest(features, meta)
        else:
            LOGGER.info("Train new model")
            model = GoldPriceModel(
                target_column=target_column,
                train_window=self.train_window,
            )
            model.refit_full_after_eval = self.refit_full_after_eval
            model_result = model.train_and_predict(features)
            self.storage.save_model(model.model)
            self.storage.save_meta(
                last_data_time=last_data_time,
                mae=model_result.mae,
                rmse=model_result.rmse,
                train_size=model_result.train_size,
                test_size=model_result.test_size,
                gold_history_source=self.hub.gold_history_source,
                train_start=(
                    model_result.train_start.isoformat(timespec="seconds")
                    if model_result.train_start
                    else None
                ),
                train_end=(
                    model_result.train_end.isoformat(timespec="seconds")
                    if model_result.train_end
                    else None
                ),
                test_start=(
                    model_result.test_start.isoformat(timespec="seconds")
                    if model_result.test_start
                    else None
                ),
                test_end=(
                    model_result.test_end.isoformat(timespec="seconds")
                    if model_result.test_end
                    else None
                ),
                feature_columns=list(features.columns),
                target_column=target_column,
            )
        last_features = features.drop(columns=[target_column]).iloc[[-1]]
        feature_names = getattr(model.model, "feature_names_in_", None)
        if feature_names is not None:
            feature_names = list(feature_names)
            last_features = last_features.reindex(columns=feature_names, fill_value=0.0)
        model_top_features = _top_contributions(model.model, list(last_features.columns), last_features)
        LOGGER.info("Pipeline completed")
        realtime = realtime_for_update
        if realtime is None:
            realtime_quote = self.hub.fetch_gold_spot_with_time(self.symbol)
            if realtime_quote is not None:
                realtime, realtime_time = realtime_quote
        realtime_bias = None
        adjusted_prediction = None
        adjusted_diff = None
        realtime_target_date: date | None = None
        realtime_target_horizon: int | None = None
        adjusted_prediction_for_realtime: float | None = None
        adjusted_diff_for_realtime: float | None = None
        if realtime is not None and realtime_time is not None:
            realtime_date = realtime_time.date()
            target_date = next_data_time.date()
            if realtime_date >= target_date:
                days = int(np.busday_count(target_date, realtime_date)) + 1
                realtime_target_horizon = days
                realtime_target_date = realtime_date
                if realtime_target_horizon == 1:
                    realtime_matches_next = True
                else:
                    realtime_matches_next = False
            else:
                realtime_matches_next = False
        else:
            realtime_matches_next = False
        if realtime is not None and realtime_matches_next:
            realtime_bias = realtime - last_close
            volatility_pct = None
            if "gold_std7" in features.columns:
                vol_value = float(features["gold_std7"].iloc[-1])
                if last_close != 0:
                    volatility_pct = vol_value / abs(last_close)
            adjusted_prediction, adjusted_diff = _adjust_prediction_with_realtime(
                model_result.prediction, realtime, volatility_pct
            )
        daily_target = None
        next_trading_date = None
        horizon_predictions: list[tuple[int, date, float]] = []
        horizon_top_features: list[tuple[int, list[tuple[str, float]]]] = []
        if frequency == "1min":
            daily_series = raw["gold"].resample("1D").last().dropna()
            if len(daily_series) >= 50:
                daily_raw = daily_series.to_frame()
                daily_features = self.builder.build_features(daily_raw, target_column="gold")
                daily_model = GoldPriceModel(target_column="gold")
                daily_result = daily_model.train_and_predict(daily_features)
                daily_target = daily_result.prediction
                last_daily = daily_series.index.max().date()
                next_trading_date = (
                    pd.Timestamp(last_daily) + pd.tseries.offsets.BDay(1)
                ).date()
                if horizons:
                    for horizon in horizons:
                        horizon_model = GoldPriceModel(target_column="gold")
                        horizon_result = horizon_model.train_and_predict_horizon(
                            daily_features, horizon
                        )
                        if horizon_result is None:
                            continue
                        target_date = (
                            pd.Timestamp(last_daily) + pd.tseries.offsets.BDay(horizon)
                        ).date()
                        horizon_predictions.append(
                            (horizon, target_date, horizon_result.prediction)
                        )
        else:
            if horizons:
                horizon_predictions, horizon_top_features = _forecast_horizons_recursive(
                    model.model,
                    raw,
                    horizons,
                    self.builder,
                    target_column=target_column,
                )
        if realtime is not None and realtime_target_horizon and realtime_target_horizon > 1:
            target_prediction: float | None = None
            if horizon_predictions:
                for horizon, _, price in horizon_predictions:
                    if horizon == realtime_target_horizon:
                        target_prediction = price
                        break
            if target_prediction is None:
                horizon_predictions_rt, _ = _forecast_horizons_recursive(
                    model.model,
                    raw,
                    [realtime_target_horizon],
                    self.builder,
                    target_column=target_column,
                )
                if horizon_predictions_rt:
                    target_prediction = horizon_predictions_rt[0][2]
            if target_prediction is not None:
                adjusted_prediction_for_realtime, adjusted_diff_for_realtime = (
                    _adjust_prediction_with_realtime(target_prediction, realtime, None)
                )

        return PipelineResult(
            data_rows=len(features),
            model=model_result,
            realtime_price=realtime,
            realtime_time=realtime_time,
            realtime_target_date=realtime_target_date,
            realtime_target_horizon=realtime_target_horizon,
            realtime_bias=realtime_bias,
            adjusted_prediction=adjusted_prediction,
            adjusted_diff=adjusted_diff,
            adjusted_prediction_for_realtime=adjusted_prediction_for_realtime,
            adjusted_diff_for_realtime=adjusted_diff_for_realtime,
            macro_factors=macro_factors,
            market_factors=market_factors,
            last_data_time=last_data_time,
            next_data_time=next_data_time,
            frequency=frequency,
            gold_history_source=self.hub.gold_history_source,
            daily_target=daily_target,
            next_trading_date=next_trading_date,
            horizon_predictions=horizon_predictions,
            model_top_features=model_top_features,
            horizon_top_features=horizon_top_features,
            last_close=last_close,
            last_close_time=last_close_time,
            target_column=target_column,
            last_target=last_target,
        )

    def run_multi(
        self,
        start: date | None,
        end: date | None,
        frequency: str = "day",
        horizons: list[int] | None = None,
    ) -> MultiTargetPipelineResult:
        LOGGER.info("Pipeline started")
        frequency = frequency.lower()
        if frequency not in {"day", "1min"}:
            raise ValueError("frequency must be 'day' or '1min'")
        target_columns = self.target_columns or [self.target_column]
        if frequency != "day":
            target_columns = ["gold"]
        storage_key = f"{frequency}"
        self.storage = Storage(self.storage.base_dir, storage_key)
        macro_factors: list[str] = []
        market_factors: list[str] = []
        factors_changed = False
        cached_factors = self.storage.load_factors()
        if frequency == "day":
            cached_macro = (cached_factors or {}).get("macro_factors", DEFAULT_MACRO_FACTORS)
            cached_market = (cached_factors or {}).get("market_factors", self.market_factors)
            cached_macro_defaults = (cached_factors or {}).get("macro_defaults", DEFAULT_MACRO_FACTORS)
            cached_market_defaults = (cached_factors or {}).get("market_defaults", self.market_factors)

            macro_factors = cached_macro
            market_factors = cached_market

            macro_changed = set(cached_macro_defaults) != set(DEFAULT_MACRO_FACTORS)
            market_changed = set(cached_market_defaults) != set(self.market_factors)

            if macro_changed:
                LOGGER.info("Default macro factors changed, refresh macro cache")
                search_func = (
                    self.hub.fred.search_series
                    if self.hub.fred and self.discovery.macro_keywords
                    else None
                )
                macro_factors = self.discovery.discover_macro(
                    DEFAULT_MACRO_FACTORS, search_func=search_func
                )
                factors_changed = True

            if market_changed:
                LOGGER.info("Default market factors changed, refresh market cache")
                market_factors = self.discovery.discover_market(self.market_factors)
                factors_changed = True

            if not cached_factors or macro_changed or market_changed:
                self.storage.save_factors(
                    macro_factors,
                    market_factors,
                    macro_defaults=DEFAULT_MACRO_FACTORS,
                    market_defaults=self.market_factors,
                )
                LOGGER.info("Factors cached")
            else:
                LOGGER.info("Use cached factors")
        if frequency != "day":
            macro_factors = []
            market_factors = []
            LOGGER.info("Intraday mode: skip macro/market factors")

        cached_raw = self.storage.load_dataset()
        if factors_changed:
            LOGGER.info("Factors changed, rebuild dataset")
            cached_raw = None
        data_end = end or date.today()
        raw = None
        rebuild_cache = False
        if cached_raw is not None:
            LOGGER.info("Use cached dataset with last date %s", cached_raw.index.max().date())
            cached_start = cached_raw.index.min().date()
            cached_end = cached_raw.index.max().date()
            desired_start = start or cached_start
            if data_end <= cached_end:
                if desired_start < cached_start:
                    LOGGER.info("Requested start before cache range, rebuild dataset")
                    rebuild_cache = True
                else:
                    raw = cached_raw.loc[pd.Timestamp(desired_start) : pd.Timestamp(data_end)]
            else:
                update_start = cached_end
                LOGGER.info("Incremental update from %s to %s", update_start, data_end)
                if frequency == "day":
                    fresh = self.builder.build_dataset(macro_factors, market_factors, update_start, data_end)
                else:
                    fresh = self.hub.fetch_gold_intraday(update_start, data_end, self.symbol).to_frame()
                raw = (
                    pd.concat([cached_raw, fresh])
                    .sort_index()
                    .loc[lambda frame: ~frame.index.duplicated(keep="last")]
                )
                raw = raw.ffill().dropna()
        if raw is None or rebuild_cache:
            if frequency == "day":
                raw = self.builder.build_dataset(macro_factors, market_factors, start, data_end)
            else:
                raw = self.hub.fetch_gold_intraday(start, data_end, self.symbol).to_frame()
        if raw is None:
            raise RuntimeError("Failed to build dataset")
        realtime_for_update: float | None = None
        realtime_time: datetime | None = None
        last_idx = raw.index.max()
        last_idx_time = pd.Timestamp(last_idx).to_pydatetime() if last_idx is not None else None
        if frequency == "day" and self.use_realtime_last_close:
            realtime_quote = self.hub.fetch_gold_spot_with_time(self.symbol)
            if realtime_quote is not None:
                realtime_for_update, realtime_time = realtime_quote
                if (
                    last_idx is not None
                    and realtime_time is not None
                    and last_idx_time is not None
                    and realtime_time.date() == last_idx_time.date()
                ):
                    for col in ("gold_open", "gold_high", "gold_low", "gold"):
                        if col in raw.columns:
                            raw.loc[last_idx, col] = realtime_for_update
        self.storage.save_dataset(raw)

        if frequency == "day":
            available_macro = [name for name in macro_factors if name in raw.columns]
            available_market = [name for name in market_factors if name in raw.columns]
            if available_macro != macro_factors or available_market != market_factors:
                LOGGER.info("Drop unavailable factors from cache")
                macro_factors = available_macro
                market_factors = available_market
                self.storage.save_factors(
                    macro_factors,
                    market_factors,
                    macro_defaults=DEFAULT_MACRO_FACTORS,
                    market_defaults=self.market_factors,
                )

        valid_targets = [col for col in target_columns if col in raw.columns]
        features = self.builder.build_features(raw, target_columns=valid_targets)
        model_pipeline = self.storage.load_model()
        meta = self.storage.load_meta()
        last_close_time = pd.Timestamp(raw.index.max()).to_pydatetime()
        last_close = float(raw["gold"].iloc[-1]) if "gold" in raw.columns else float(raw[valid_targets[0]].iloc[-1])
        last_data_time = last_close_time
        if frequency == "1min":
            next_data_time = last_data_time + pd.Timedelta(minutes=1)
        else:
            next_data_time = (pd.Timestamp(last_data_time) + pd.tseries.offsets.BDay(1)).to_pydatetime()

        reuse_ok = False
        if model_pipeline is not None and meta and meta.get("last_data_time") == last_data_time.isoformat():
            feature_columns = meta.get("feature_columns")
            cached_targets = meta.get("target_columns")
            current_features = list(features.drop(columns=valid_targets).columns)
            if isinstance(feature_columns, list) and current_features == feature_columns:
                if isinstance(cached_targets, list) and cached_targets == valid_targets:
                    reuse_ok = True
            else:
                LOGGER.info("Feature columns changed, retrain model")
        if reuse_ok:
            if not meta or any(
                meta.get(field) is None
                for field in ("train_start", "train_end", "test_start", "test_end")
            ):
                LOGGER.info("Model meta missing train/test range, retrain model")
                reuse_ok = False

        if reuse_ok:
            model = GoldPriceModel(
                model_pipeline,
                train_window=self.train_window,
            )
            model.refit_full_after_eval = self.refit_full_after_eval
            assert meta is not None
            model_result = model.predict_latest_multi(features, meta, valid_targets)
        else:
            LOGGER.info("Train new model")
            model = GoldPriceModel(train_window=self.train_window)
            model.refit_full_after_eval = self.refit_full_after_eval
            model_result = model.train_and_predict_multi(features, valid_targets)
            self.storage.save_model(model.model)
            self.storage.save_meta(
                last_data_time=last_data_time,
                mae=float(np.mean([val[0] for val in model_result.metrics.values()])),
                rmse=float(np.mean([val[1] for val in model_result.metrics.values()])),
                train_size=model_result.train_size,
                test_size=model_result.test_size,
                gold_history_source=self.hub.gold_history_source,
                train_start=(
                    model_result.train_start.isoformat(timespec="seconds")
                    if model_result.train_start
                    else None
                ),
                train_end=(
                    model_result.train_end.isoformat(timespec="seconds")
                    if model_result.train_end
                    else None
                ),
                test_start=(
                    model_result.test_start.isoformat(timespec="seconds")
                    if model_result.test_start
                    else None
                ),
                test_end=(
                    model_result.test_end.isoformat(timespec="seconds")
                    if model_result.test_end
                    else None
                ),
                feature_columns=list(features.drop(columns=valid_targets).columns),
                target_columns=valid_targets,
                target_metrics=model_result.metrics,
            )

        last_targets = {name: float(raw[name].iloc[-1]) for name in valid_targets}
        realtime = realtime_for_update
        if realtime is None:
            realtime_quote = self.hub.fetch_gold_spot_with_time(self.symbol)
            if realtime_quote is not None:
                realtime, realtime_time = realtime_quote
        adjusted_predictions: dict[str, float] = {}
        adjusted_diffs: dict[str, float] = {}
        adjusted_predictions_for_realtime: dict[str, float] = {}
        adjusted_diffs_for_realtime: dict[str, float] = {}
        realtime_target_date: date | None = None
        realtime_target_horizon: int | None = None
        if realtime is not None and realtime_time is not None:
            realtime_date = realtime_time.date()
            target_date = next_data_time.date()
            if realtime_date >= target_date:
                days = int(np.busday_count(target_date, realtime_date)) + 1
                realtime_target_horizon = days
                realtime_target_date = realtime_date
                if realtime_target_horizon == 1:
                    realtime_matches_next = True
                else:
                    realtime_matches_next = False
            else:
                realtime_matches_next = False
        else:
            realtime_matches_next = False
        if realtime is not None and realtime_matches_next:
            for name in valid_targets:
                prediction = model_result.predictions.get(name)
                if prediction is None:
                    continue
                series = raw[name] if name in raw.columns else raw[valid_targets[0]]
                vol_value = float(series.tail(7).std()) if len(series) >= 7 else float(series.std())
                volatility_pct = vol_value / abs(last_close) if last_close != 0 else None
                adjusted_prediction, adjusted_diff = _adjust_prediction_with_realtime(
                    prediction, realtime, volatility_pct
                )
                adjusted_predictions[name] = adjusted_prediction
                adjusted_diffs[name] = adjusted_diff
        horizon_predictions: dict[str, list[tuple[int, date, float]]] = {}
        horizon_top_features: list[tuple[int, list[tuple[str, float]]]] = []
        if frequency == "day" and horizons:
            horizon_predictions, horizon_top_features = _forecast_horizons_recursive_multi(
                model.model,
                raw,
                horizons,
                self.builder,
                target_columns=valid_targets,
            )
        if realtime is not None and realtime_target_horizon and realtime_target_horizon > 1:
            target_predictions: dict[str, float] = {}
            if horizon_predictions:
                for name, predictions in horizon_predictions.items():
                    for horizon, _, price in predictions:
                        if horizon == realtime_target_horizon:
                            target_predictions[name] = price
                            break
            if not target_predictions:
                horizon_predictions_rt, _ = _forecast_horizons_recursive_multi(
                    model.model,
                    raw,
                    [realtime_target_horizon],
                    self.builder,
                    target_columns=valid_targets,
                )
                for name, predictions in horizon_predictions_rt.items():
                    if predictions:
                        target_predictions[name] = predictions[0][2]
            for name, price in target_predictions.items():
                adjusted_prediction, adjusted_diff = _adjust_prediction_with_realtime(
                    price, realtime, None
                )
                adjusted_predictions_for_realtime[name] = adjusted_prediction
                adjusted_diffs_for_realtime[name] = adjusted_diff
        LOGGER.info("Pipeline completed")

        return MultiTargetPipelineResult(
            data_rows=len(features),
            model=model_result,
            realtime_price=realtime,
            realtime_time=realtime_time,
            realtime_target_date=realtime_target_date,
            realtime_target_horizon=realtime_target_horizon,
            macro_factors=macro_factors,
            market_factors=market_factors,
            last_data_time=last_data_time,
            next_data_time=next_data_time,
            frequency=frequency,
            gold_history_source=self.hub.gold_history_source,
            target_columns=valid_targets,
            last_targets=last_targets,
            last_close=last_close,
            last_close_time=last_close_time,
            adjusted_predictions=adjusted_predictions,
            adjusted_diffs=adjusted_diffs,
            adjusted_predictions_for_realtime=adjusted_predictions_for_realtime,
            adjusted_diffs_for_realtime=adjusted_diffs_for_realtime,
            horizon_predictions=horizon_predictions,
            horizon_top_features=horizon_top_features,
        )


def _top_contributions(
    model_pipeline: Pipeline | EnsemblePipeline,
    feature_names: list[str],
    last_features: pd.DataFrame,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    try:
        if isinstance(model_pipeline, EnsemblePipeline):
            best_single = getattr(model_pipeline, "_best_single", None)
            if not isinstance(best_single, Pipeline):
                return []
            model_pipeline = best_single
        if hasattr(model_pipeline, "_best_single"):
            best_single = getattr(model_pipeline, "_best_single")
            if isinstance(best_single, Pipeline):
                model_pipeline = best_single
        model_pipeline = cast(Pipeline, model_pipeline)
        scaler = model_pipeline.named_steps.get("scaler")
        weighter = model_pipeline.named_steps.get("weighter")
        reg = model_pipeline.named_steps.get("reg")
        if scaler is None or reg is None:
            return []
        if hasattr(model_pipeline, "_feature_names"):
            feature_names = list(getattr(model_pipeline, "_feature_names"))
            last_features = last_features.reindex(columns=feature_names, fill_value=0.0)
        if hasattr(reg, "coef_"):
            if scaler == "passthrough":
                x_scaled = last_features
            else:
                x_scaled = scaler.transform(last_features)
            if weighter is not None:
                x_scaled = weighter.transform(x_scaled)
            if isinstance(x_scaled, pd.DataFrame):
                x_scaled = x_scaled.to_numpy()
            coefs = reg.coef_
            if hasattr(coefs, "ndim") and coefs.ndim > 1:
                coefs = coefs[0]
            contributions = coefs * x_scaled[0]
            pairs = list(zip(feature_names, contributions))
            pairs.sort(key=lambda item: abs(item[1]), reverse=True)
            return pairs[:top_n]
        if hasattr(reg, "feature_importances_"):
            importances = reg.feature_importances_
            pairs = list(zip(feature_names, importances))
            pairs.sort(key=lambda item: abs(item[1]), reverse=True)
            return pairs[:top_n]
        return []
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to compute contributions: %s", exc)
        return []


def _top_contributions_multi_average(
    model_pipeline: Pipeline | EnsemblePipeline,
    feature_names: list[str],
    last_features: pd.DataFrame,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    try:
        if isinstance(model_pipeline, EnsemblePipeline):
            best_single = getattr(model_pipeline, "_best_single", None)
            if not isinstance(best_single, Pipeline):
                return []
            model_pipeline = best_single
        if hasattr(model_pipeline, "_best_single"):
            best_single = getattr(model_pipeline, "_best_single")
            if isinstance(best_single, Pipeline):
                model_pipeline = best_single
        model_pipeline = cast(Pipeline, model_pipeline)
        scaler = model_pipeline.named_steps.get("scaler")
        weighter = model_pipeline.named_steps.get("weighter")
        reg = model_pipeline.named_steps.get("reg")
        if scaler is None or reg is None:
            return []
        if hasattr(model_pipeline, "_feature_names"):
            feature_names = list(getattr(model_pipeline, "_feature_names"))
            last_features = last_features.reindex(columns=feature_names, fill_value=0.0)
        if scaler == "passthrough":
            x_scaled = last_features
        else:
            x_scaled = scaler.transform(last_features)
        if weighter is not None:
            x_scaled = weighter.transform(x_scaled)
        if isinstance(x_scaled, pd.DataFrame):
            x_scaled = x_scaled.to_numpy()

        if isinstance(reg, MultiOutputRegressor):
            contribs_list: list[np.ndarray] = []
            for estimator in reg.estimators_:
                est: Any = estimator
                if hasattr(est, "coef_"):
                    coefs = est.coef_
                    if hasattr(coefs, "ndim") and coefs.ndim > 1:
                        coefs = coefs[0]
                    contribs_list.append(np.asarray(coefs) * x_scaled[0])
                elif hasattr(est, "feature_importances_"):
                    contribs_list.append(np.asarray(est.feature_importances_))
            if not contribs_list:
                return []
            mean_abs = np.mean([np.abs(arr) for arr in contribs_list], axis=0)
            pairs = list(zip(feature_names, mean_abs))
            pairs.sort(key=lambda item: abs(item[1]), reverse=True)
            return pairs[:top_n]

        if hasattr(reg, "coef_"):
            coefs = reg.coef_
            if hasattr(coefs, "ndim") and coefs.ndim > 1:
                coefs = coefs[0]
            contributions = np.asarray(coefs) * x_scaled[0]
            pairs = list(zip(feature_names, np.abs(contributions)))
            pairs.sort(key=lambda item: abs(item[1]), reverse=True)
            return pairs[:top_n]
        if hasattr(reg, "feature_importances_"):
            importances = np.asarray(reg.feature_importances_)
            pairs = list(zip(feature_names, np.abs(importances)))
            pairs.sort(key=lambda item: abs(item[1]), reverse=True)
            return pairs[:top_n]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to compute multi contributions: %s", exc)
        return []
    return []


def _forecast_horizons_recursive(
    model_pipeline: Pipeline | EnsemblePipeline,
    raw: pd.DataFrame,
    horizons: list[int],
    builder: FactorBuilder,
    target_column: str,
) -> tuple[list[tuple[int, date, float]], list[tuple[int, list[tuple[str, float]]]]]:
    if not horizons:
        return [], []
    max_horizon = max(horizons)
    horizon_set = set(horizons)
    horizon_predictions: list[tuple[int, date, float]] = []
    horizon_top_features: list[tuple[int, list[tuple[str, float]]]] = []

    temp_raw = raw.copy()
    current_date = pd.Timestamp(raw.index.max())
    for step in range(1, max_horizon + 1):
        features = builder.build_features(temp_raw, target_column=target_column)
        if features.empty:
            break
        last_features = features.drop(columns=[target_column]).iloc[[-1]]
        if hasattr(model_pipeline, "_feature_names"):
            feature_names = list(getattr(model_pipeline, "_feature_names"))
            last_features = last_features.reindex(columns=feature_names, fill_value=0.0)
        prediction = float(np.asarray(model_pipeline.predict(last_features))[0])
        if getattr(model_pipeline, "_target_transform", None) == "log":
            prediction = float(np.exp(prediction))
        current_date = current_date + pd.tseries.offsets.BDay(1)

        if step in horizon_set:
            horizon_predictions.append((step, current_date.date(), prediction))
            top = _top_contributions(model_pipeline, list(last_features.columns), last_features)
            horizon_top_features.append((step, top))

        last_row = temp_raw.iloc[-1].copy()
        last_row[target_column] = prediction
        temp_raw = pd.concat([temp_raw, pd.DataFrame([last_row], index=[current_date])])

    return horizon_predictions, horizon_top_features


def _adjust_prediction_with_realtime(
    prediction: float, realtime: float, volatility_pct: float | None
) -> tuple[float, float]:
    if realtime == 0:
        return prediction, prediction - realtime
    bias = realtime - prediction
    max_diff = 0.05 * abs(realtime)
    required_alpha = 0.0
    if abs(bias) > max_diff:
        required_alpha = 1 - (max_diff / abs(bias))
    if volatility_pct is None or volatility_pct <= 0:
        base_alpha = 0.75
    else:
        base_alpha = min(0.95, max(0.6, 0.6 + 3.0 * volatility_pct))
    alpha = min(1.0, max(base_alpha, required_alpha))
    adjusted = prediction + alpha * bias
    return adjusted, realtime - adjusted


def format_report(result: PipelineResult) -> str:
    lines = [
        "Gold Valuation Report",
        "",
        f"Data rows: {result.data_rows}",
        f"Gold history source: {result.gold_history_source}",
        f"Macro factors: {', '.join(result.macro_factors)}",
        f"Market factors: {', '.join(result.market_factors)}",
    ]
    target_desc = FACTOR_DESCRIPTIONS.get(result.target_column, "Not specified")
    lines.extend(
        [
            f"Frequency: {result.frequency}",
            f"Target column: {result.target_column}",
            f"Latest historical data time: {result.last_data_time.isoformat(timespec='seconds')}",
            f"Last close time: {result.last_close_time.isoformat(timespec='seconds')}",
            f"Forecast time (next period): {result.next_data_time.isoformat(timespec='seconds')}",
            f"Last close price: {result.last_close:.2f}",
            f"Last target value: {result.last_target:.2f}",
            f"Model next-period prediction: {result.model.prediction:.2f}",
            f"MAE: {result.model.mae:.4f}",
            f"RMSE: {result.model.rmse:.4f}",
        ]
    )
    if result.daily_target is not None and result.next_trading_date is not None:
        lines.extend(
            [
                f"Next trading day date: {result.next_trading_date.isoformat()}",
                f"Next trading day target price: {result.daily_target:.2f}",
            ]
        )
    if result.horizon_predictions:
        use_horizon_adjusted = result.realtime_price is not None
        lines.append(
            "Forward targets (adjusted):" if use_horizon_adjusted else "Forward targets:"
        )
        for horizon, target_date, price in result.horizon_predictions:
            display_price = price
            if use_horizon_adjusted and result.realtime_price is not None:
                display_price, _ = _adjust_prediction_with_realtime(
                    price, result.realtime_price, None
                )
            lines.append(
                f"- {horizon} trading days later ({target_date.isoformat()}): {display_price:.2f}"
            )
    if result.model_top_features:
        lines.append("Top impact factors (by absolute contribution):")
        for name, value in result.model_top_features:
            sign = "+" if value >= 0 else "-"
            lines.append(f"- {name}: {sign}{abs(value):.4f} | {_describe_feature(name)}")
    if result.horizon_top_features:
        lines.append("Forward forecast impact factors (by absolute contribution):")
        for horizon, top_features in result.horizon_top_features:
            if not top_features:
                continue
            lines.append(f"- Horizon {horizon}:")
            for name, value in top_features:
                sign = "+" if value >= 0 else "-"
                lines.append(f"  - {name}: {sign}{abs(value):.4f} | {_describe_feature(name)}")
        avg_scores: dict[str, float] = {}
        avg_counts: dict[str, int] = {}
        for _, top_features in result.horizon_top_features:
            for name, value in top_features:
                avg_scores[name] = avg_scores.get(name, 0.0) + abs(value)
                avg_counts[name] = avg_counts.get(name, 0) + 1
        averaged = [
            (name, avg_scores[name] / avg_counts[name])
            for name in avg_scores
            if avg_counts.get(name, 0) > 0
        ]
        averaged.sort(key=lambda item: item[1], reverse=True)
        if averaged:
            lines.append("Top average impact factors across horizons:")
            for name, value in averaged[:10]:
                lines.append(
                    f"- {name}: {value:.4f} | {_describe_feature(name)}"
                )
    if result.realtime_price is not None:
        diff = result.realtime_price - result.model.prediction
        lines.extend(
            [
                f"Realtime gold price: {result.realtime_price:.2f}",
                f"Realtime price time: {result.realtime_time.isoformat(timespec='seconds') if result.realtime_time else '-'}",
                f"Realtime - prediction diff (before adjustment): {diff:.2f}",
            ]
        )
        if result.adjusted_prediction is not None and result.adjusted_diff is not None:
            lines.append(f"Adjusted prediction from realtime bias: {result.adjusted_prediction:.2f}")
            lines.append(f"Realtime - prediction diff (after adjustment): {result.adjusted_diff:.2f}")
        if (
            result.adjusted_prediction_for_realtime is not None
            and result.adjusted_diff_for_realtime is not None
            and result.realtime_target_date is not None
            and result.realtime_target_horizon is not None
        ):
            lines.append(
                f"Realtime-adjusted target date: {result.realtime_target_date.isoformat()}"
            )
            lines.append(
                f"Realtime-adjusted target price: {result.adjusted_prediction_for_realtime:.2f}"
            )
            lines.append(
                f"Realtime - target diff (after adjustment): {result.adjusted_diff_for_realtime:.2f}"
            )
    lines.append("")
    lines.append("Target description:")
    lines.append(f"- {result.target_column}: {target_desc}")
    lines.append("Factor descriptions:")
    if result.macro_factors:
        lines.append("Macro factor descriptions:")
        for name in result.macro_factors:
            desc = FACTOR_DESCRIPTIONS.get(name, "Not specified")
            lines.append(f"- {name}: {desc}")
    if result.market_factors:
        lines.append("Market factor descriptions:")
        for name in result.market_factors:
            desc = FACTOR_DESCRIPTIONS.get(name, "Not specified")
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def format_report_multi(result: MultiTargetPipelineResult) -> str:
    lines = [
        "Gold Valuation Report",
        "",
        f"Data rows: {result.data_rows}",
        f"Gold history source: {result.gold_history_source}",
        f"Macro factors: {', '.join(result.macro_factors)}",
        f"Market factors: {', '.join(result.market_factors)}",
        f"Frequency: {result.frequency}",
        f"Target columns: {', '.join(result.target_columns)}",
        f"Latest historical data time: {result.last_data_time.isoformat(timespec='seconds')}",
        f"Last close time: {result.last_close_time.isoformat(timespec='seconds')}",
        f"Forecast time (next period): {result.next_data_time.isoformat(timespec='seconds')}",
        f"Latest OHLC (raw latest): O {result.last_targets.get('gold_open', float('nan')):.2f} | "
        f"H {result.last_targets.get('gold_high', float('nan')):.2f} | "
        f"L {result.last_targets.get('gold_low', float('nan')):.2f} | "
        f"C {result.last_targets.get('gold', float('nan')):.2f}",
        "",
        "Target summary:",
    ]
    use_adjusted = bool(result.adjusted_predictions)
    if use_adjusted:
        lines.append("| Target | Latest actual | Prediction (adjusted) | MAE | RMSE |")
    else:
        lines.append("| Target | Latest actual | Prediction | MAE | RMSE |")
    lines.append("|---|---|---|---|---|")
    for name in result.target_columns:
        prediction = result.model.predictions.get(name)
        if use_adjusted:
            prediction = result.adjusted_predictions.get(name, prediction)
        metrics = result.model.metrics.get(name)
        last_actual = result.model.last_actuals.get(name)
        if prediction is None or metrics is None or last_actual is None:
            continue
        mae, rmse = metrics
        lines.append(
            f"| {name} | {last_actual:.2f} | {prediction:.2f} | {mae:.4f} | {rmse:.4f} |"
        )
    if result.horizon_predictions:
        lines.append("")
        use_horizon_adjusted = result.realtime_price is not None
        lines.append(
            "Forward targets (adjusted):" if use_horizon_adjusted else "Forward targets:"
        )
        horizon_set: set[int] = set()
        horizon_dates: dict[int, date] = {}
        horizon_values: dict[tuple[str, int], float] = {}
        for name, predictions in result.horizon_predictions.items():
            for horizon, target_date, price in predictions:
                horizon_set.add(horizon)
                horizon_dates.setdefault(horizon, target_date)
                display_price = price
                if use_horizon_adjusted and result.realtime_price is not None:
                    display_price, _ = _adjust_prediction_with_realtime(
                        price, result.realtime_price, None
                    )
                horizon_values[(name, horizon)] = display_price
        if horizon_set:
            ordered_horizons = sorted(horizon_set)
            header = "| Trading days ahead | Date | " + " | ".join(result.target_columns) + " |"
            sep = "|---" + "|---" * (len(result.target_columns) + 1) + "|"
            lines.append(header)
            lines.append(sep)
            for horizon in ordered_horizons:
                date_str = horizon_dates.get(horizon)
                date_text = date_str.isoformat() if date_str else "-"
                row_values = []
                for name in result.target_columns:
                    value = horizon_values.get((name, horizon))
                    row_values.append(f"{value:.2f}" if value is not None else "-")
                lines.append(
                    "| "
                    + str(horizon)
                    + " | "
                    + date_text
                    + " | "
                    + " | ".join(row_values)
                    + " |"
                )
    if result.horizon_top_features:
        avg_scores: dict[str, float] = {}
        avg_counts: dict[str, int] = {}
        for _, top_features in result.horizon_top_features:
            for name, value in top_features:
                avg_scores[name] = avg_scores.get(name, 0.0) + abs(value)
                avg_counts[name] = avg_counts.get(name, 0) + 1
        averaged = [
            (name, avg_scores[name] / avg_counts[name])
            for name in avg_scores
            if avg_counts.get(name, 0) > 0
        ]
        averaged.sort(key=lambda item: item[1], reverse=True)
        if averaged:
            lines.append("")
            lines.append("Top average impact factors across horizons:")
            for name, value in averaged[:10]:
                lines.append(
                    f"- {name}: {value:.4f} | {_describe_feature(name)}"
                )
    if result.realtime_price is not None:
        lines.append("")
        lines.append(f"Realtime gold price: {result.realtime_price:.2f}")
        lines.append(
            f"Realtime price time: {result.realtime_time.isoformat(timespec='seconds') if result.realtime_time else '-'}"
        )
        if result.adjusted_predictions:
            lines.append("Bias-adjusted prediction from realtime price:")
            lines.append("| Target | Adjusted prediction | Realtime - prediction diff (after adjustment) |")
            lines.append("|---|---|---|")
            for name in result.target_columns:
                adjusted_prediction = result.adjusted_predictions.get(name)
                adjusted_diff = result.adjusted_diffs.get(name)
                if adjusted_prediction is None or adjusted_diff is None:
                    continue
                lines.append(
                    f"| {name} | {adjusted_prediction:.2f} | {adjusted_diff:.2f} |"
                )
        if result.adjusted_predictions_for_realtime and result.realtime_target_date:
            lines.append("")
            lines.append(
                f"Realtime-adjusted target date: {result.realtime_target_date.isoformat()}"
            )
            lines.append("Realtime-adjusted target prices:")
            lines.append("| Target | Adjusted prediction | Realtime - target diff (after adjustment) |")
            lines.append("|---|---|---|")
            for name in result.target_columns:
                adjusted_prediction = result.adjusted_predictions_for_realtime.get(name)
                adjusted_diff = result.adjusted_diffs_for_realtime.get(name)
                if adjusted_prediction is None or adjusted_diff is None:
                    continue
                lines.append(
                    f"| {name} | {adjusted_prediction:.2f} | {adjusted_diff:.2f} |"
                )
    lines.append("")
    lines.append("Target descriptions:")
    for name in result.target_columns:
        lines.append(f"- {name}: {FACTOR_DESCRIPTIONS.get(name, 'Not specified')}")
    lines.append("Factor descriptions:")
    if result.macro_factors:
        lines.append("Macro factor descriptions:")
        for name in result.macro_factors:
            desc = FACTOR_DESCRIPTIONS.get(name, "Not specified")
            lines.append(f"- {name}: {desc}")
    if result.market_factors:
        lines.append("Market factor descriptions:")
        for name in result.market_factors:
            desc = FACTOR_DESCRIPTIONS.get(name, "Not specified")
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def _forecast_horizons_recursive_multi(
    model_pipeline: Pipeline | EnsemblePipeline,
    raw: pd.DataFrame,
    horizons: list[int],
    builder: FactorBuilder,
    target_columns: list[str],
) -> tuple[
    dict[str, list[tuple[int, date, float]]],
    list[tuple[int, list[tuple[str, float]]]],
]:
    if not horizons:
        return {}, []
    max_horizon = max(horizons)
    horizon_set = set(horizons)
    horizon_predictions: dict[str, list[tuple[int, date, float]]] = {
        name: [] for name in target_columns
    }
    horizon_top_features: list[tuple[int, list[tuple[str, float]]]] = []

    temp_raw = raw.copy()
    current_date = pd.Timestamp(raw.index.max())
    for step in range(1, max_horizon + 1):
        features = builder.build_features(temp_raw, target_columns=target_columns)
        if features.empty:
            break
        last_features = features.drop(columns=target_columns).iloc[[-1]]
        if hasattr(model_pipeline, "_feature_names"):
            feature_names = list(getattr(model_pipeline, "_feature_names"))
            last_features = last_features.reindex(columns=feature_names, fill_value=0.0)
        pred = np.asarray(model_pipeline.predict(last_features))
        if pred.ndim > 1:
            pred = pred[0]
        if getattr(model_pipeline, "_target_transform", None) == "log":
            pred = np.exp(pred)
        current_date = current_date + pd.tseries.offsets.BDay(1)

        last_row = temp_raw.iloc[-1].copy()
        for idx, name in enumerate(target_columns):
            if idx < len(pred):
                last_row[name] = float(pred[idx])
        temp_raw = pd.concat([temp_raw, pd.DataFrame([last_row], index=[current_date])])

        if step in horizon_set:
            pred_map = {name: float(pred[idx]) for idx, name in enumerate(target_columns) if idx < len(pred)}
            for name, value in pred_map.items():
                horizon_predictions[name].append((step, current_date.date(), value))
            top = _top_contributions_multi_average(
                model_pipeline, list(last_features.columns), last_features
            )
            horizon_top_features.append((step, top))

    return horizon_predictions, horizon_top_features


def _describe_feature(feature_name: str) -> str:
    if feature_name in FACTOR_DESCRIPTIONS:
        return FACTOR_DESCRIPTIONS[feature_name]
    if feature_name.endswith("_pct"):
        base = feature_name[: -len("_pct")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} return"
    if feature_name.endswith("_ma7"):
        base = feature_name[: -len("_ma7")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 7-day mean"
    if feature_name.endswith("_ma30"):
        base = feature_name[: -len("_ma30")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 30-day mean"
    if feature_name.endswith("_ma5"):
        base = feature_name[: -len("_ma5")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 5-day mean"
    if feature_name.endswith("_ma21"):
        base = feature_name[: -len("_ma21")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 21-day mean"
    if feature_name.endswith("_ma60"):
        base = feature_name[: -len("_ma60")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 60-day mean"
    if feature_name.endswith("_std5"):
        base = feature_name[: -len("_std5")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 5-day volatility (std dev)"
    if feature_name.endswith("_std7"):
        base = feature_name[: -len("_std7")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 7-day volatility (std dev)"
    if feature_name.endswith("_std21"):
        base = feature_name[: -len("_std21")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 21-day volatility (std dev)"
    if feature_name.endswith("_std30"):
        base = feature_name[: -len("_std30")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 30-day volatility (std dev)"
    if feature_name.endswith("_lag1"):
        base = feature_name[: -len("_lag1")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 1-day lag"
    if feature_name.endswith("_lag2"):
        base = feature_name[: -len("_lag2")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 2-day lag"
    if feature_name.endswith("_lag5"):
        base = feature_name[: -len("_lag5")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 5-day lag"
    if feature_name.endswith("_lag10"):
        base = feature_name[: -len("_lag10")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 10-day lag"
    if feature_name.endswith("_lag21"):
        base = feature_name[: -len("_lag21")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 21-day lag"
    if feature_name.endswith("_ret1"):
        base = feature_name[: -len("_ret1")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 1-day return"
    if feature_name.endswith("_ret5"):
        base = feature_name[: -len("_ret5")]
        return f"{FACTOR_DESCRIPTIONS.get(base, base)} 5-day return"
    return f"{feature_name} indicator"
