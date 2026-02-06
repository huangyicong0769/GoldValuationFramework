from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable
from collections import deque
import logging
import io
from pathlib import Path
import hashlib
import random
import time
import html
import re
import time
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)


def _create_session(retries: int, backoff: float) -> requests.Session:
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@dataclass
class FREDClient:
    api_key: str
    timeout: int = 20
    cache_dir: str | None = None
    cache_ttl_days: int = 7
    retries: int = 3
    backoff: float = 0.5
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._session = _create_session(self.retries, self.backoff)

    def get_series(
        self,
        series_id: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.Series:
        cached = self._load_cache(series_id, start, end)
        if cached is not None:
            LOGGER.info("FRED cache hit for %s", series_id)
            return cached
        LOGGER.info("Fetch FRED series %s", series_id)
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start.isoformat()
        if end:
            params["observation_end"] = end.isoformat()

        url = "https://api.stlouisfed.org/fred/series/observations"
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        records: list[tuple[pd.Timestamp, float]] = []
        for obs in payload.get("observations", []):
            value = obs.get("value", ".")
            if value in (".", None, ""):
                continue
            try:
                records.append((pd.to_datetime(obs["date"]), float(value)))
            except (ValueError, KeyError):
                continue

        series = pd.Series({ts: val for ts, val in records}, name=series_id)
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()
        self._save_cache(series_id, start, end, series)
        return series

    def _cache_path(self, series_id: str, start: date | None, end: date | None) -> Path | None:
        if not self.cache_dir:
            return None
        key = f"{series_id}|{start}|{end}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        base = Path(self.cache_dir)
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{series_id}_{digest}.pkl"

    def _load_cache(self, series_id: str, start: date | None, end: date | None) -> pd.Series | None:
        path = self._cache_path(series_id, start, end)
        if not path or not path.exists():
            return None
        if self.cache_ttl_days > 0:
            age_seconds = time.time() - path.stat().st_mtime
            if age_seconds > self.cache_ttl_days * 86400:
                return None
        try:
            return pd.read_pickle(path)
        except Exception:  # noqa: BLE001
            return None

    def _save_cache(self, series_id: str, start: date | None, end: date | None, series: pd.Series) -> None:
        path = self._cache_path(series_id, start, end)
        if not path:
            return
        try:
            series.to_pickle(path)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Failed to write FRED cache for %s", series_id)

    def search_series(self, keyword: str, limit: int = 5) -> list[str]:
        params = {
            "search_text": keyword,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": limit,
        }
        url = "https://api.stlouisfed.org/fred/series/search"
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        series_ids = [item.get("id") for item in payload.get("seriess", []) if item.get("id")]
        return series_ids


@dataclass
class StooqClient:
    timeout: int = 20
    retries: int = 3
    backoff: float = 0.5
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._session = _create_session(self.retries, self.backoff)

    def get_daily_series(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.Series:
        LOGGER.info("Fetch stooq series %s", symbol)
        url = "https://stooq.com/q/d/l/"
        params = {
            "s": symbol.lower(),
            "i": "d",
        }
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        if frame.empty:
            raise ValueError(f"No data returned for {symbol}")

        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.sort_values("Date")
        if start:
            frame = frame[frame["Date"] >= pd.to_datetime(start)]
        if end:
            frame = frame[frame["Date"] <= pd.to_datetime(end)]

        series = frame.set_index("Date")["Close"].rename(symbol.lower())
        return series

    def get_daily_ohlc(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        LOGGER.info("Fetch stooq OHLC %s", symbol)
        url = "https://stooq.com/q/d/l/"
        params = {
            "s": symbol.lower(),
            "i": "d",
        }
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        if frame.empty:
            raise ValueError(f"No data returned for {symbol}")

        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.sort_values("Date")
        if start:
            frame = frame[frame["Date"] >= pd.to_datetime(start)]
        if end:
            frame = frame[frame["Date"] <= pd.to_datetime(end)]

        ohlc = frame.set_index("Date")[["Open", "High", "Low", "Close"]].rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
            }
        )
        return ohlc


@dataclass
class MetalsLiveClient:
    timeout: int = 20
    retries: int = 3
    backoff: float = 0.5
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._session = _create_session(self.retries, self.backoff)

    def get_spot_gold(self) -> float:
        price, _ = self.get_spot_gold_with_time()
        return price

    def get_spot_gold_with_time(self) -> tuple[float, datetime]:
        LOGGER.info("Fetch realtime gold spot")
        url = "https://api.metals.live/v1/spot/gold"
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not payload:
            raise ValueError("Empty response from metals.live")
        ts_raw = payload[0][0]
        price = float(payload[0][1])
        try:
            if isinstance(ts_raw, (int, float)):
                ts_value = float(ts_raw)
                if ts_value > 1e12:
                    ts_value = ts_value / 1000.0
                ts = datetime.utcfromtimestamp(ts_value)
            else:
                ts = pd.to_datetime(ts_raw).to_pydatetime()
        except Exception:  # noqa: BLE001
            ts = datetime.utcnow()
        return price, ts


@dataclass
class TwelveDataClient:
    api_key: str
    timeout: int = 20
    cache_dir: str | None = None
    cache_ttl_minutes: int = 15
    retries: int = 3
    backoff: float = 0.5
    rate_limit_per_minute: int = 8
    _session: requests.Session = field(init=False, repr=False)
    _request_timestamps: deque[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._session = _create_session(self.retries, self.backoff)
        self._request_timestamps = deque()

    def _throttle(self) -> None:
        if self.rate_limit_per_minute <= 0:
            return
        window = 60.0
        while True:
            now = time.monotonic()
            while self._request_timestamps and now - self._request_timestamps[0] >= window:
                self._request_timestamps.popleft()
            if len(self._request_timestamps) < self.rate_limit_per_minute:
                self._request_timestamps.append(now)
                return
            wait_seconds = window - (now - self._request_timestamps[0])
            if wait_seconds > 0:
                LOGGER.info("Twelve Data rate limit hit, sleeping %.2fs", wait_seconds)
                time.sleep(wait_seconds)

    def get_intraday_series(
        self,
        symbol: str,
        interval: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.Series:
        cached = self._load_cache(symbol, interval, start, end, "series")
        if isinstance(cached, pd.Series):
            LOGGER.info("Twelve Data cache hit for %s %s", symbol, interval)
            return cached
        LOGGER.info("Fetch Twelve Data intraday %s %s", symbol, interval)
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "order": "ASC",
        }
        if start:
            params["start_date"] = start.isoformat()
        if end:
            params["end_date"] = end.isoformat()

        self._throttle()
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if "values" not in payload:
            raise ValueError(payload.get("message", "No data returned"))

        records: list[tuple[pd.Timestamp, float]] = []
        for item in payload.get("values", []):
            ts = item.get("datetime")
            value = item.get("close")
            if not ts or value is None:
                continue
            try:
                records.append((pd.to_datetime(ts), float(value)))
            except ValueError:
                continue

        series = pd.Series({ts: val for ts, val in records}, name="gold")
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()
        self._save_cache(symbol, interval, start, end, "series", series)
        return series

    def get_daily_series(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.Series:
        return self.get_intraday_series(symbol, "1day", start, end)

    def get_daily_ohlc(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        cached = self._load_cache(symbol, "1day", start, end, "ohlc")
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            LOGGER.info("Twelve Data cache hit for OHLC %s", symbol)
            return cached

        LOGGER.info("Fetch Twelve Data OHLC %s", symbol)
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": "1day",
            "apikey": self.api_key,
            "order": "ASC",
        }
        if start:
            params["start_date"] = start.isoformat()
        if end:
            params["end_date"] = end.isoformat()

        self._throttle()
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if "values" not in payload:
            raise ValueError(payload.get("message", "No data returned"))

        records: list[tuple[pd.Timestamp, float, float, float, float]] = []
        for item in payload.get("values", []):
            ts = item.get("datetime")
            if not ts:
                continue
            try:
                records.append(
                    (
                        pd.to_datetime(ts),
                        float(item["open"]),
                        float(item["high"]),
                        float(item["low"]),
                        float(item["close"]),
                    )
                )
            except (ValueError, KeyError):
                continue

        ohlc = pd.DataFrame(
            records,
            columns=["datetime", "open", "high", "low", "close"],
        ).set_index("datetime")
        ohlc.index = pd.to_datetime(ohlc.index)
        ohlc = ohlc.sort_index()
        self._save_cache(symbol, "1day", start, end, "ohlc", ohlc)
        return ohlc

    def get_latest_price(self, symbol: str) -> float:
        cached = self._load_cache(symbol, "1min", None, None, "latest")
        if isinstance(cached, (int, float)):
            LOGGER.info("Twelve Data cache hit for latest %s", symbol)
            return float(cached)
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": "1min",
            "apikey": self.api_key,
            "order": "DESC",
            "outputsize": 1,
        }
        self._throttle()
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        values = payload.get("values")
        if not values:
            raise ValueError(payload.get("message", "No data returned"))
        latest = values[0]
        price = float(latest["close"])
        self._save_cache(symbol, "1min", None, None, "latest", price)
        return price

    def get_latest_quote(self, symbol: str) -> tuple[float, datetime]:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": "1min",
            "apikey": self.api_key,
            "order": "DESC",
            "outputsize": 1,
        }
        self._throttle()
        response = self._session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        values = payload.get("values")
        if not values:
            raise ValueError(payload.get("message", "No data returned"))
        latest = values[0]
        price = float(latest["close"])
        ts = pd.to_datetime(latest.get("datetime"))
        return price, ts.to_pydatetime()

    def _cache_path(
        self,
        symbol: str,
        interval: str,
        start: date | None,
        end: date | None,
        kind: str,
    ) -> Path | None:
        if not self.cache_dir:
            return None
        key = f"{symbol}|{interval}|{start}|{end}|{kind}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        base = Path(self.cache_dir)
        base.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace("/", "_")
        return base / f"{safe_symbol}_{interval}_{kind}_{digest}.pkl"

    def _load_cache(
        self,
        symbol: str,
        interval: str,
        start: date | None,
        end: date | None,
        kind: str,
    ) -> pd.Series | pd.DataFrame | float | None:
        path = self._cache_path(symbol, interval, start, end, kind)
        if not path or not path.exists():
            return None
        if self.cache_ttl_minutes > 0:
            age_seconds = time.time() - path.stat().st_mtime
            if age_seconds > self.cache_ttl_minutes * 60:
                return None
        try:
            payload = pd.read_pickle(path)
            if kind == "latest":
                if isinstance(payload, pd.Series) and not payload.empty:
                    return float(payload.iloc[0])
                if isinstance(payload, (int, float)):
                    return float(payload)
                return None
            return payload
        except Exception:  # noqa: BLE001
            return None

    def _save_cache(
        self,
        symbol: str,
        interval: str,
        start: date | None,
        end: date | None,
        kind: str,
        payload: pd.Series | pd.DataFrame | float,
    ) -> None:
        path = self._cache_path(symbol, interval, start, end, kind)
        if not path:
            return
        try:
            if isinstance(payload, (pd.Series, pd.DataFrame)):
                payload.to_pickle(path)
            else:
                pd.Series([payload]).to_pickle(path)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Failed to write Twelve Data cache for %s", symbol)


@dataclass
class GoldDataHub:
    fred: FREDClient | None
    stooq: StooqClient
    metals: MetalsLiveClient
    twelve: TwelveDataClient | None = None
    gold_history_source: str = "stooq"
    gold_history_symbol: str = "XAU/USD"

    def fetch_macro_factors(
        self,
        series_ids: Iterable[str],
        start: date | None,
        end: date | None,
    ) -> dict[str, pd.Series]:
        factors: dict[str, pd.Series] = {}
        if not self.fred:
            LOGGER.warning("FRED API key missing, skip macro factors: %s", list(series_ids))
            return factors
        for series_id in series_ids:
            try:
                factors[series_id] = self.fred.get_series(series_id, start, end)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to fetch %s: %s", series_id, exc)
        return factors

    def fetch_market_factors(
        self,
        symbols: Iterable[str],
        start: date | None,
        end: date | None,
    ) -> dict[str, pd.Series]:
        twelve_map = {
            "usdidx": ["UUP"],
            "spx": ["SPY"],
            "dji": ["DIA"],
            "ndx": ["QQQ"],
            "vix": ["VIXY"],
        }
        factors: dict[str, pd.Series] = {}
        for symbol in symbols:
            try:
                factors[symbol] = self.stooq.get_daily_series(symbol, start, end)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to fetch %s: %s", symbol, exc)
                if self.twelve:
                    candidates = twelve_map.get(symbol.lower(), [symbol])
                    for twelve_symbol in candidates:
                        try:
                            series = self.twelve.get_daily_series(twelve_symbol, start, end)
                            factors[symbol] = series.rename(symbol)
                            LOGGER.info("Market factor fallback to Twelve Data: %s", twelve_symbol)
                            break
                        except Exception as fallback_exc:  # noqa: BLE001
                            LOGGER.warning("Twelve Data fallback failed for %s: %s", symbol, fallback_exc)
        return factors

    def fetch_gold_history(
        self,
        start: date | None,
        end: date | None,
    ) -> pd.DataFrame:
        if self.gold_history_source == "twelve":
            if not self.twelve:
                LOGGER.warning("Twelve Data not configured, fallback to stooq for gold history")
                ohlc = self.stooq.get_daily_ohlc("xauusd", start, end)
            else:
                ohlc = self.twelve.get_daily_ohlc(self.gold_history_symbol, start, end)
        else:
            ohlc = self.stooq.get_daily_ohlc("xauusd", start, end)
        ohlc = ohlc.copy()
        ohlc.rename(
            columns={
                "open": "gold_open",
                "high": "gold_high",
                "low": "gold_low",
                "close": "gold",
            },
            inplace=True,
        )
        return ohlc

    def fetch_gold_intraday(
        self,
        start: date | None,
        end: date | None,
        symbol: str,
        interval: str = "1min",
    ) -> pd.Series:
        if not self.twelve:
            raise RuntimeError("TWELVE_DATA_API_KEY is required for intraday data")
        return self.twelve.get_intraday_series(symbol, interval, start, end)

    def fetch_gold_spot(self, symbol: str = "XAU/USD") -> float | None:
        quote = self.fetch_gold_spot_with_time(symbol)
        return quote[0] if quote else None

    def fetch_gold_spot_with_time(self, symbol: str = "XAU/USD") -> tuple[float, datetime] | None:
        try:
            return self.metals.get_spot_gold_with_time()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to fetch realtime gold price: %s", exc)
            if self.twelve:
                try:
                    price, ts = self.twelve.get_latest_quote(symbol)
                    LOGGER.info("Realtime fallback: Twelve Data %s", symbol)
                    return price, ts
                except Exception as fallback_exc:  # noqa: BLE001
                    LOGGER.warning("Fallback realtime price failed: %s", fallback_exc)
            return None


@dataclass(frozen=True)
class NewsSource:
    name: str
    category: str
    url: str
    priority: int = 0


@dataclass
class NewsClient:
    sources: list[NewsSource]
    timeout: int = 20
    cache_dir: str | None = None
    cache_ttl_hours: int = 24
    retries: int = 2
    backoff: float = 0.6
    min_interval_seconds: float = 1.5
    max_total_items: int = 120
    max_items_per_category: int = 40
    category_limits: dict | None = None
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._session = _create_session(self.retries, self.backoff)

    def fetch_news(self, limit_per_source: int = 12) -> dict:
        fetched_at = datetime.utcnow().isoformat()
        items: list[dict] = []
        for source in self.sources:
            cache_status = self._cache_status(source)
            if cache_status == "hit":
                cached = self._load_cache(source, ignore_ttl=True)
                if cached is not None:
                    LOGGER.info("News cache hit: %s %s", source.name, source.category)
                    items.extend(self._strip_output_fields(cached.get("items", []), remove_links=False))
                    continue
            elif cache_status == "expired":
                LOGGER.info("News cache expired: %s %s", source.name, source.category)
            else:
                LOGGER.info("News cache miss: %s %s", source.name, source.category)

            try:
                payload = self._fetch_source(source, limit_per_source)
                cached_items = self._attach_fetched_at(payload.get("items", []), payload.get("fetched_at"))
                cached_payload = dict(payload)
                cached_payload["items"] = cached_items
                self._save_cache(source, cached_payload)
                items.extend(self._strip_output_fields(payload.get("items", []), remove_links=False))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("News source failed (%s %s): %s", source.name, source.category, exc)
                stale = self._load_cache(source, ignore_ttl=True)
                if stale is not None:
                    LOGGER.info(
                        "News fallback cache used: %s %s (cached at %s)",
                        source.name,
                        source.category,
                        stale.get("fetched_at"),
                    )
                    items.extend(self._strip_output_fields(stale.get("items", []), remove_links=False))

            self._apply_min_interval()
        items = self._dedupe_items(items)
        items = self._limit_items(items)
        return {
            "fetched_at": fetched_at,
            "items": items,
        }

    def _apply_min_interval(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        jitter = random.uniform(0, 0.6)
        time.sleep(self.min_interval_seconds + jitter)

    def _fetch_source(self, source: NewsSource, limit_per_source: int) -> dict:
        headers = self._build_headers()
        response = self._session.get(source.url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        text = response.text
        items = self._parse_feed(text, source, limit_per_source)
        return {
            "fetched_at": datetime.utcnow().isoformat(),
            "source": source.name,
            "category": source.category,
            "items": items,
        }

    def _build_headers(self) -> dict:
        user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        ]
        return {
            "User-Agent": random.choice(user_agents),
            "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.1",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    def _parse_feed(self, content: str, source: NewsSource, limit_per_source: int) -> list[dict]:
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return []

        items: list[dict] = []

        channel = root.find("channel")
        if channel is not None:
            for item in channel.findall("item"):
                parsed = self._parse_rss_item(item, source)
                if parsed:
                    items.append(parsed)
                if len(items) >= limit_per_source:
                    break
            return items

        if root.tag.endswith("feed"):
            for entry in root.findall("{*}entry"):
                parsed = self._parse_atom_entry(entry, source)
                if parsed:
                    items.append(parsed)
                if len(items) >= limit_per_source:
                    break
        return items

    def _parse_rss_item(self, item: ET.Element, source: NewsSource) -> dict | None:
        title = self._get_text(item.find("title"))
        link = self._get_text(item.find("link"))
        pub_date = self._get_text(item.find("pubDate"))
        summary = self._get_text(item.find("description"))
        return self._build_item(source, title, link, pub_date, summary)

    def _parse_atom_entry(self, entry: ET.Element, source: NewsSource) -> dict | None:
        title = self._get_text(entry.find("{*}title"))
        link = None
        link_el = entry.find("{*}link")
        if link_el is not None:
            link = link_el.attrib.get("href") or link_el.text
        updated = self._get_text(entry.find("{*}updated"))
        summary = self._get_text(entry.find("{*}summary")) or self._get_text(entry.find("{*}content"))
        return self._build_item(source, title, link, updated, summary)

    def _build_item(
        self,
        source: NewsSource,
        title: str | None,
        link: str | None,
        published_raw: str | None,
        summary: str | None,
    ) -> dict | None:
        if not title:
            return None
        published_at = self._parse_published_at(published_raw)
        cleaned_summary = self._clean_text(summary)
        return {
            "source": source.name,
            "category": source.category,
            "priority": source.priority,
            "title": title.strip(),
            "link": link.strip() if link else None,
            "published_at": published_at,
            "summary": cleaned_summary,
        }

    @staticmethod
    def _get_text(element: ET.Element | None) -> str | None:
        if element is None:
            return None
        text = element.text or ""
        return text.strip() if text else None

    @staticmethod
    def _clean_text(text: str | None) -> str | None:
        if not text:
            return None
        cleaned = html.unescape(text)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if cleaned else None

    @staticmethod
    def _parse_published_at(value: str | None) -> str | None:
        if not value:
            return None
        try:
            return parsedate_to_datetime(value).isoformat()
        except Exception:
            pass
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except Exception:
            return None

    @staticmethod
    def _attach_fetched_at(items: list[dict], fetched_at: str | None) -> list[dict]:
        if not fetched_at:
            return items
        enriched: list[dict] = []
        for item in items:
            if isinstance(item, dict):
                payload = dict(item)
                payload["source_fetched_at"] = fetched_at
                enriched.append(payload)
            else:
                enriched.append(item)
        return enriched

    @staticmethod
    def _strip_output_fields(items: list[dict], remove_links: bool = False) -> list[dict]:
        stripped: list[dict] = []
        for item in items:
            if isinstance(item, dict):
                payload = dict(item)
                payload.pop("source_fetched_at", None)
                if remove_links:
                    payload.pop("link", None)
                stripped.append(payload)
            else:
                stripped.append(item)
        return stripped

    @staticmethod
    def strip_for_llm_payload(payload: dict | None) -> dict | None:
        if not payload or not isinstance(payload, dict):
            return payload
        items = payload.get("items", [])
        stripped_items = NewsClient._strip_output_fields(items, remove_links=True)
        sanitized = dict(payload)
        sanitized["items"] = stripped_items
        return sanitized

    def _cache_status(self, source: NewsSource) -> str:
        path = self._cache_path(source)
        if not path or not path.exists():
            return "miss"
        if self.cache_ttl_hours <= 0:
            return "hit"
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=self.cache_ttl_hours):
            return "expired"
        return "hit"

    @staticmethod
    def _dedupe_items(items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        deduped: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip().lower()
            source = (item.get("source") or "").strip().lower()
            published = (item.get("published_at") or "").strip()
            summary = (item.get("summary") or "").strip().lower()
            key = "|".join([source, title, published, summary])
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _limit_items(self, items: list[dict]) -> list[dict]:
        if not items:
            return items
        def _sort_key(item: dict) -> tuple[int, str]:
            priority = item.get("priority")
            try:
                priority_val = int(priority) if priority is not None else 0
            except Exception:
                priority_val = 0
            ts = item.get("published_at") or ""
            return (priority_val, ts)

        try:
            items_sorted = sorted(items, key=_sort_key, reverse=True)
        except Exception:
            items_sorted = items

        per_category_default = max(1, int(self.max_items_per_category))
        total_limit = max(1, int(self.max_total_items))

        buckets: dict[str, list[dict]] = {}
        for item in items_sorted:
            category = (item.get("category") or "unknown").strip().lower()
            buckets.setdefault(category, []).append(item)

        trimmed: list[dict] = []
        limits = self.category_limits or {}
        for category, group in buckets.items():
            limit = limits.get(category, per_category_default)
            try:
                limit_int = max(1, int(limit))
            except Exception:
                limit_int = per_category_default
            trimmed.extend(group[:limit_int])

        if len(trimmed) > total_limit:
            trimmed = sorted(trimmed, key=_sort_key, reverse=True)[:total_limit]
        return trimmed

    def _cache_path(self, source: NewsSource) -> Path | None:
        if not self.cache_dir:
            return None
        today = date.today().isoformat()
        key = f"{source.name}|{source.category}|{source.url}|{today}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        base = Path(self.cache_dir)
        base.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", source.name.lower())
        return base / f"{safe_name}_{source.category}_{digest}.json"

    def _load_cache(self, source: NewsSource, ignore_ttl: bool = False) -> dict | None:
        path = self._cache_path(source)
        if not path or not path.exists():
            return None
        if self.cache_ttl_hours > 0 and not ignore_ttl:
            age_seconds = time.time() - path.stat().st_mtime
            if age_seconds > self.cache_ttl_hours * 3600:
                return None
        try:
            import json

            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_cache(self, source: NewsSource, payload: dict) -> None:
        path = self._cache_path(source)
        if not path:
            return
        try:
            import json

            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            LOGGER.debug("Failed to write news cache for %s", source.name)
