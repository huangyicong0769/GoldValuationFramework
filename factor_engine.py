from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable
import logging

import numpy as np
import pandas as pd

from config import DEFAULT_MACRO_FACTORS, DEFAULT_MARKET_FACTORS
from data_sources import GoldDataHub


LOGGER = logging.getLogger(__name__)


@dataclass
class FactorDiscovery:
    macro_keywords: list[str]

    def discover_macro(
        self,
        default: Iterable[str] | None = None,
        search_func: Callable[[str, int], list[str]] | None = None,
        limit_per_keyword: int = 3,
    ) -> list[str]:
        discovered = list(default or DEFAULT_MACRO_FACTORS)
        if search_func:
            for keyword in self.macro_keywords:
                try:
                    discovered.extend(search_func(keyword, limit_per_keyword))
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Macro discovery failed for %s: %s", keyword, exc)
        deduped = list(dict.fromkeys(discovered))
        LOGGER.info("Macro factors discovered: %s", deduped)
        return deduped

    def discover_market(self, default: Iterable[str] | None = None) -> list[str]:
        discovered = list(default or DEFAULT_MARKET_FACTORS)
        LOGGER.info("Market factors discovered: %s", discovered)
        return discovered


@dataclass
class FactorBuilder:
    hub: GoldDataHub

    def build_dataset(
        self,
        macro_ids: Iterable[str],
        market_symbols: Iterable[str],
        start: date | None,
        end: date | None,
    ) -> pd.DataFrame:
        macro_ids = list(macro_ids)
        market_symbols = list(market_symbols)
        LOGGER.info("Build dataset: %s macro factors, %s market factors", len(macro_ids), len(market_symbols))
        macro = self.hub.fetch_macro_factors(macro_ids, start, end)
        market = self.hub.fetch_market_factors(market_symbols, start, end)
        gold = self.hub.fetch_gold_history(start, end)

        if isinstance(gold, pd.Series):
            gold_frame = gold.to_frame()
        else:
            gold_frame = gold.copy()
        if all(col in gold_frame.columns for col in ["gold_open", "gold_high", "gold_low", "gold"]):
            gold_frame = gold_frame.copy()
            gold_frame["gold_hl2"] = (gold_frame["gold_high"] + gold_frame["gold_low"]) / 2
            gold_frame["gold_hlc3"] = (
                gold_frame["gold_high"] + gold_frame["gold_low"] + gold_frame["gold"]
            ) / 3
            gold_frame["gold_ohlc4"] = (
                gold_frame["gold_open"]
                + gold_frame["gold_high"]
                + gold_frame["gold_low"]
                + gold_frame["gold"]
            ) / 4
            gold_frame["gold_oc2"] = (
                gold_frame["gold_open"] + gold_frame["gold"]
            ) / 2

        gold_index = gold_frame.index

        def _align_series(name: str, series: pd.Series) -> pd.Series:
            series = series.sort_index()
            if len(series) > 1:
                median_delta = series.index.to_series().diff().median()
                if isinstance(median_delta, pd.Timedelta) and median_delta > pd.Timedelta(days=1):
                    LOGGER.warning(
                        "Macro/market series %s appears non-daily (median delta %s)",
                        name,
                        median_delta,
                    )
            return series.reindex(gold_index).ffill()

        frames = [gold_frame]
        for name, series in {**macro, **market}.items():
            frames.append(_align_series(name, series).rename(name).to_frame())

        data = pd.concat(frames, axis=1).sort_index()
        data = data.dropna(axis=1, how="all")

        if not data.empty:
            last_index = data.index.max()
            cutoff = last_index - pd.Timedelta(days=120)
            keep_cols: list[str] = []
            for column in data.columns:
                last_valid = data[column].last_valid_index()
                if last_valid is None:
                    continue
                if last_valid >= cutoff:
                    keep_cols.append(column)
            data = data[keep_cols]

        data = data.ffill().dropna()

        return data

    @staticmethod
    def build_features(
        data: pd.DataFrame,
        target_column: str = "gold",
        target_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        LOGGER.info("Build features from %s rows", len(data))
        features = data.copy()
        target_cols = target_columns or [target_column]
        if any(col.startswith("gold") for col in target_cols):
            LOGGER.info("Skip gold-derived features to avoid autoregressive OHLC usage")

        derived_frames: list[pd.DataFrame] = []
        for column in data.columns:
            if column in target_cols or column.startswith("gold"):
                continue
            series = data[column]
            rolling_7 = series.rolling(window=7)
            rolling_30 = series.rolling(window=30)
            rolling_90 = series.rolling(window=90)
            ema_12 = series.ewm(span=12, adjust=False).mean()
            ema_26 = series.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            derived_frames.append(
                pd.DataFrame(
                    {
                        f"{column}_pct": series.pct_change(),
                        f"{column}_pct5": series.pct_change(5),
                        f"{column}_pct21": series.pct_change(21),
                        f"{column}_ma7": rolling_7.mean(),
                        f"{column}_ma30": rolling_30.mean(),
                        f"{column}_ma90": rolling_90.mean(),
                        f"{column}_std7": rolling_7.std(),
                        f"{column}_std30": rolling_30.std(),
                        f"{column}_std90": rolling_90.std(),
                        f"{column}_z30": (series - rolling_30.mean()) / rolling_30.std(),
                        f"{column}_z90": (series - rolling_90.mean()) / rolling_90.std(),
                        f"{column}_lag1": series.shift(1),
                        f"{column}_lag5": series.shift(5),
                        f"{column}_lag21": series.shift(21),
                        f"{column}_ema12": ema_12,
                        f"{column}_ema26": ema_26,
                        f"{column}_macd": macd,
                        f"{column}_macd_signal": macd.ewm(span=9, adjust=False).mean(),
                    },
                    index=data.index,
                )
            )
        if derived_frames:
            features = pd.concat([features] + derived_frames, axis=1)

        if "DGS10" in features.columns and "DGS2" in features.columns:
            curve = features["DGS10"] - features["DGS2"]
            features["yc_10y2y"] = curve
            features["yc_10y2y_chg"] = curve.diff()
            features["yc_10y2y_ma30"] = curve.rolling(window=30).mean()
            features["yc_10y2y_std30"] = curve.rolling(window=30).std()

        if "DGS10" in features.columns and "DGS3MO" in features.columns:
            features["yc_10y3m"] = features["DGS10"] - features["DGS3MO"]

        if "DGS10" in features.columns and "DGS5" in features.columns:
            features["yc_10y5y"] = features["DGS10"] - features["DGS5"]

        if "DGS30" in features.columns and "DGS10" in features.columns:
            features["yc_30y10y"] = features["DGS30"] - features["DGS10"]

        if "T10YIE" in features.columns:
            breakeven = features["T10YIE"]
            features["t10yie_chg"] = breakeven.diff()
            features["t10yie_ma30"] = breakeven.rolling(window=30).mean()
            features["t10yie_std30"] = breakeven.rolling(window=30).std()

        if "T5YIE" in features.columns:
            breakeven_5y = features["T5YIE"]
            features["t5yie_chg"] = breakeven_5y.diff()
            features["t5yie_ma30"] = breakeven_5y.rolling(window=30).mean()
            features["t5yie_std30"] = breakeven_5y.rolling(window=30).std()

        if "DGS10" in features.columns and "T10YIE" in features.columns:
            real_rate = features["DGS10"] - features["T10YIE"]
            features["real_rate_proxy"] = real_rate
            features["real_rate_proxy_ma30"] = real_rate.rolling(window=30).mean()
            features["real_rate_proxy_chg"] = real_rate.diff()

        if "DGS5" in features.columns and "T5YIE" in features.columns:
            features["real_rate_proxy_5y"] = features["DGS5"] - features["T5YIE"]

        equity_pct_cols = [
            col
            for col in ["spx_pct", "dji_pct", "ndx_pct"]
            if col in features.columns
        ]
        if equity_pct_cols:
            equity_avg = features[equity_pct_cols].mean(axis=1)
            features["equity_avg_pct"] = equity_avg
            features["equity_avg_vol30"] = equity_avg.rolling(window=30).std()

            if "vix_pct" in features.columns:
                features["risk_appetite"] = equity_avg - features["vix_pct"]

            if "usdidx_pct" in features.columns:
                features["equity_vs_usd"] = equity_avg - features["usdidx_pct"]

        if any(col.startswith("gold") for col in features.columns):
            keep_prefixes = ("gold_lag", "gold_ret", "gold_ma", "gold_std")
            drop_gold = [
                col
                for col in features.columns
                if col.startswith("gold")
                and col not in target_cols
                and not col.startswith(keep_prefixes)
            ]
            if drop_gold:
                features = features.drop(columns=drop_gold)

        gold_base_cols = [col for col in target_cols if col in data.columns]
        for col in gold_base_cols:
            series = data[col]
            features[f"{col}_lag1"] = series.shift(1)
            features[f"{col}_lag2"] = series.shift(2)
            features[f"{col}_lag5"] = series.shift(5)
            features[f"{col}_lag10"] = series.shift(10)
            features[f"{col}_lag21"] = series.shift(21)
            features[f"{col}_ret1"] = series.pct_change(1)
            features[f"{col}_ret5"] = series.pct_change(5)
            features[f"{col}_ma5"] = series.rolling(window=5).mean().shift(1)
            features[f"{col}_ma21"] = series.rolling(window=21).mean().shift(1)
            features[f"{col}_ma60"] = series.rolling(window=60).mean().shift(1)
            features[f"{col}_std5"] = series.rolling(window=5).std().shift(1)
            features[f"{col}_std21"] = series.rolling(window=21).std().shift(1)

        if isinstance(features.index, pd.DatetimeIndex):
            features["cal_month"] = features.index.month
            features["cal_quarter"] = features.index.quarter
            features["cal_dayofweek"] = features.index.dayofweek
            features["cal_weekofyear"] = features.index.isocalendar().week.astype(int)

        liquidity_components = [
            name
            for name in ["M2SL", "TOTLL", "WALCL", "WRESBAL", "RRPONTSYD", "BOGMBASE"]
            if name in features.columns
        ]
        if liquidity_components:
            component_frame = features[liquidity_components]
            rolling_mean = component_frame.rolling(window=252, min_periods=60).mean()
            rolling_std = component_frame.rolling(window=252, min_periods=60).std()
            zscores = (component_frame - rolling_mean) / rolling_std
            features["liquidity_index"] = zscores.mean(axis=1)

        if "GFDEBTN" in features.columns:
            debt = features["GFDEBTN"]
            debt_mean = debt.rolling(window=252, min_periods=60).mean()
            debt_std = debt.rolling(window=252, min_periods=60).std()
            features["debt_index"] = (debt - debt_mean) / debt_std

        if "liquidity_index" in features.columns and "debt_index" in features.columns:
            features["neutral_balance"] = (
                features["liquidity_index"] - features["debt_index"]
            )

        features = features.mask(np.isinf(features), pd.NA)
        features = features.ffill()

        nan_ratio = features.isna().mean()
        keep_cols = nan_ratio[nan_ratio <= 0.3].index.tolist()
        for col in target_cols:
            if col in features.columns and col not in keep_cols:
                keep_cols.append(col)
        features = features[keep_cols]
        features = features.dropna()
        LOGGER.info("Features kept: %s/%s columns", len(keep_cols), features.shape[1])
        return features
