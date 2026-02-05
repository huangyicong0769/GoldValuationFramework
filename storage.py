from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import joblib


LOGGER = logging.getLogger(__name__)


@dataclass
class Storage:
    base_dir: str = "artifacts"
    namespace: str | None = None

    def __post_init__(self) -> None:
        root = Path(self.base_dir)
        if self.namespace:
            root = root / self.namespace
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def factors_path(self) -> Path:
        return self._root / "factors.json"

    @property
    def data_path(self) -> Path:
        return self._root / "raw_data.pkl"

    @property
    def model_path(self) -> Path:
        return self._root / "model.joblib"

    @property
    def meta_path(self) -> Path:
        return self._root / "model_meta.json"

    def load_factors(self) -> dict[str, list[str]] | None:
        if not self.factors_path.exists():
            return None
        return json.loads(self.factors_path.read_text(encoding="utf-8"))

    def save_factors(
        self,
        macro: list[str],
        market: list[str],
        macro_defaults: list[str] | None = None,
        market_defaults: list[str] | None = None,
    ) -> None:
        payload = {
            "macro_factors": macro,
            "market_factors": market,
        }
        if macro_defaults is not None:
            payload["macro_defaults"] = macro_defaults
        if market_defaults is not None:
            payload["market_defaults"] = market_defaults
        self.factors_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_dataset(self) -> pd.DataFrame | None:
        if not self.data_path.exists():
            return None
        return pd.read_pickle(self.data_path)

    def save_dataset(self, data: pd.DataFrame) -> None:
        data.to_pickle(self.data_path)

    def load_model(self) -> Any | None:
        if not self.model_path.exists():
            return None
        return joblib.load(self.model_path)

    def save_model(self, model: Any) -> None:
        joblib.dump(model, self.model_path)

    def load_meta(self) -> dict[str, Any] | None:
        if not self.meta_path.exists():
            return None
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def save_meta(
        self,
        last_data_time: datetime,
        mae: float,
        rmse: float,
        train_size: int,
        test_size: int,
        gold_history_source: str | None = None,
        train_start: str | None = None,
        train_end: str | None = None,
        test_start: str | None = None,
        test_end: str | None = None,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        target_columns: list[str] | None = None,
        target_metrics: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        payload = {
            "last_data_time": last_data_time.isoformat(),
            "mae": mae,
            "rmse": rmse,
            "train_size": train_size,
            "test_size": test_size,
            "gold_history_source": gold_history_source,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "feature_columns": feature_columns or [],
            "target_column": target_column,
            "target_columns": target_columns or [],
            "target_metrics": {
                key: [float(val[0]), float(val[1])] for key, val in (target_metrics or {}).items()
            },
        }
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
