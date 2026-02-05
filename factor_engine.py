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

        cols_to_process = [
            col for col in data.columns 
            if col not in target_cols and not col.startswith("gold")
        ]

        derived_frames: list[pd.DataFrame] = []
        std_safeguard_value = 1e-12
        if cols_to_process:
            subset = data[cols_to_process]

            rolling_7 = subset.rolling(window=7)
            rolling_30 = subset.rolling(window=30)
            rolling_90 = subset.rolling(window=90)

            rolling_7_mean = rolling_7.mean()
            rolling_30_mean = rolling_30.mean()
            rolling_90_mean = rolling_90.mean()
            rolling_7_std = rolling_7.std()
            rolling_30_std = rolling_30.std()
            rolling_90_std = rolling_90.std()

            rolling_30_std_safe = rolling_30_std.mask(
                np.isclose(rolling_30_std, 0),
                std_safeguard_value,
            )
            rolling_90_std_safe = rolling_90_std.mask(
                np.isclose(rolling_90_std, 0),
                std_safeguard_value,
            )

            ema_12 = subset.ewm(span=12, adjust=False).mean()
            ema_26 = subset.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9, adjust=False).mean()

            derived_frames = [
                subset.pct_change().add_suffix("_pct"),
                subset.pct_change(5).add_suffix("_pct5"),
                subset.pct_change(21).add_suffix("_pct21"),
                rolling_7_mean.add_suffix("_ma7"),
                rolling_30_mean.add_suffix("_ma30"),
                rolling_90_mean.add_suffix("_ma90"),
                rolling_7_std.add_suffix("_std7"),
                rolling_30_std.add_suffix("_std30"),
                rolling_90_std.add_suffix("_std90"),
                ((subset - rolling_30_mean) / rolling_30_std_safe).add_suffix("_z30"),
                ((subset - rolling_90_mean) / rolling_90_std_safe).add_suffix("_z90"),
                subset.shift(1).add_suffix("_lag1"),
                subset.shift(5).add_suffix("_lag5"),
                subset.shift(21).add_suffix("_lag21"),
                ema_12.add_suffix("_ema12"),
                ema_26.add_suffix("_ema26"),
                macd.add_suffix("_macd"),
                macd_signal.add_suffix("_macd_signal"),
            ]

        # Additional features collection to prevent fragmentation
        additional_features = []

        if "DGS10" in features.columns and "DGS2" in features.columns:
            curve = features["DGS10"] - features["DGS2"]
            additional_features.append(curve.rename("yc_10y2y"))
            additional_features.append(curve.diff().rename("yc_10y2y_chg"))
            additional_features.append(curve.rolling(window=30).mean().rename("yc_10y2y_ma30"))
            additional_features.append(curve.rolling(window=30).std().rename("yc_10y2y_std30"))

        if "DGS10" in features.columns and "DGS3MO" in features.columns:
            additional_features.append((features["DGS10"] - features["DGS3MO"]).rename("yc_10y3m"))

        if "DGS10" in features.columns and "DGS5" in features.columns:
            additional_features.append((features["DGS10"] - features["DGS5"]).rename("yc_10y5y"))

        if "DGS30" in features.columns and "DGS10" in features.columns:
            additional_features.append((features["DGS30"] - features["DGS10"]).rename("yc_30y10y"))

        if "T10YIE" in features.columns:
            breakeven = features["T10YIE"]
            additional_features.append(breakeven.diff().rename("t10yie_chg"))
            additional_features.append(breakeven.rolling(window=30).mean().rename("t10yie_ma30"))
            additional_features.append(breakeven.rolling(window=30).std().rename("t10yie_std30"))

        if "T5YIE" in features.columns:
            breakeven_5y = features["T5YIE"]
            additional_features.append(breakeven_5y.diff().rename("t5yie_chg"))
            additional_features.append(breakeven_5y.rolling(window=30).mean().rename("t5yie_ma30"))
            additional_features.append(breakeven_5y.rolling(window=30).std().rename("t5yie_std30"))

        if "DGS10" in features.columns and "T10YIE" in features.columns:
            real_rate = features["DGS10"] - features["T10YIE"]
            additional_features.append(real_rate.rename("real_rate_proxy"))
            additional_features.append(real_rate.rolling(window=30).mean().rename("real_rate_proxy_ma30"))
            additional_features.append(real_rate.diff().rename("real_rate_proxy_chg"))

        if "DGS5" in features.columns and "T5YIE" in features.columns:
            additional_features.append((features["DGS5"] - features["T5YIE"]).rename("real_rate_proxy_5y"))

        equity_pct_cols = [
            col
            for col in ["spx_pct", "dji_pct", "ndx_pct"]
            if col in features.columns
        ]
        if equity_pct_cols:
            equity_avg = features[equity_pct_cols].mean(axis=1)
            additional_features.append(equity_avg.rename("equity_avg_pct"))
            additional_features.append(equity_avg.rolling(window=30).std().rename("equity_avg_vol30"))

            if "vix_pct" in features.columns:
                additional_features.append((equity_avg - features["vix_pct"]).rename("risk_appetite"))

            if "usdidx_pct" in features.columns:
                additional_features.append((equity_avg - features["usdidx_pct"]).rename("equity_vs_usd"))

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
        if gold_base_cols:
            g_subset = data[gold_base_cols]
            additional_features.append(g_subset.shift(1).add_suffix("_lag1"))
            additional_features.append(g_subset.shift(2).add_suffix("_lag2"))
            additional_features.append(g_subset.shift(5).add_suffix("_lag5"))
            additional_features.append(g_subset.shift(10).add_suffix("_lag10"))
            additional_features.append(g_subset.shift(21).add_suffix("_lag21"))
            additional_features.append(g_subset.pct_change(1).add_suffix("_ret1"))
            additional_features.append(g_subset.pct_change(5).add_suffix("_ret5"))
            
            g_roll5 = g_subset.rolling(window=5)
            g_roll21 = g_subset.rolling(window=21)
            g_roll60 = g_subset.rolling(window=60)
            
            additional_features.append(g_roll5.mean().shift(1).add_suffix("_ma5"))
            additional_features.append(g_roll21.mean().shift(1).add_suffix("_ma21"))
            additional_features.append(g_roll60.mean().shift(1).add_suffix("_ma60"))
            additional_features.append(g_roll5.std().shift(1).add_suffix("_std5"))
            additional_features.append(g_roll21.std().shift(1).add_suffix("_std21"))

        if isinstance(features.index, pd.DatetimeIndex):
            cal_df = pd.DataFrame(
                {
                    "cal_month": features.index.month,
                    "cal_quarter": features.index.quarter,
                    "cal_dayofweek": features.index.dayofweek,
                    "cal_weekofyear": features.index.isocalendar().week.astype(int),
                },
                index=features.index,
            )
            additional_features.append(cal_df)

        liquidity_index = None
        debt_index = None
        
        liquidity_components = [
            name
            for name in ["M2SL", "TOTLL", "WALCL", "WRESBAL", "RRPONTSYD", "BOGMBASE"]
            if name in features.columns
        ]
        if liquidity_components:
            component_frame = features[liquidity_components]
            rolling_mean = component_frame.rolling(window=252, min_periods=60).mean()
            rolling_std = component_frame.rolling(window=252, min_periods=60).std()
            rolling_std_safe = rolling_std.mask(
                np.isclose(rolling_std, 0),
                std_safeguard_value,
            )
            zscores = (component_frame - rolling_mean) / rolling_std_safe
            liquidity_index = zscores.mean(axis=1)
            additional_features.append(liquidity_index.rename("liquidity_index"))

        if "GFDEBTN" in features.columns:
            debt = features["GFDEBTN"]
            debt_mean = debt.rolling(window=252, min_periods=60).mean()
            debt_std = debt.rolling(window=252, min_periods=60).std()
            debt_std_safe = debt_std.mask(
                np.isclose(debt_std, 0),
                std_safeguard_value,
            )
            debt_index = (debt - debt_mean) / debt_std_safe
            additional_features.append(debt_index.rename("debt_index"))

        if liquidity_index is not None and debt_index is not None:
            additional_features.append((liquidity_index - debt_index).rename("neutral_balance"))
            
        if derived_frames or additional_features:
            frames_to_concat: list[pd.DataFrame] = [features]
            if derived_frames:
                frames_to_concat.extend(derived_frames)
            if additional_features:
                normalized_additional = [
                    item.to_frame() if isinstance(item, pd.Series) else item
                    for item in additional_features
                ]
                frames_to_concat.extend(normalized_additional)
            features = pd.concat(frames_to_concat, axis=1)

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
