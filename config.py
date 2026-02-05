from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


def _parse_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    fred_api_key: str | None = None
    fred_cache_dir: str = "artifacts/fred_cache"
    fred_cache_ttl_days: int = 7
    twelve_data_api_key: str | None = None
    twelve_data_symbol: str = "XAU/USD"
    twelve_cache_dir: str = "artifacts/twelve_cache"
    twelve_cache_ttl_minutes: int = 15
    gold_history_source: str = "twelve"
    market_factors: list[str] | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_user: str | None = None
    smtp_pass: str | None = None
    smtp_from: str | None = None
    smtp_to: str | None = None
    smtp_use_ssl: bool = True
    smtp_starttls: bool = False
    http_timeout: int = 30
    http_retries: int = 3
    http_backoff: float = 0.5
    artifact_dir: str = "artifacts"
    target_column: str = "gold_ohlc4"
    target_columns: list[str] | None = None
    train_window: int | None = None
    refit_full_after_eval: bool = True
    use_realtime_last_close: bool = True
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "gpt-4o"
    llm_min_interval_seconds: int = 8
    llm_lock_timeout_seconds: int = 300
    llm_max_output_tokens: int = 128000
    llm_max_input_chars: int = 128000
    llm_empty_retry_count: int = 2
    llm_empty_retry_delay_seconds: int = 5
    llm_report_language: str = "Chinese"


    @staticmethod
    def from_env() -> "Settings":
        load_dotenv(override=False)
        smtp_port_raw = os.getenv("SMTP_PORT")
        train_window_raw = os.getenv("TRAIN_WINDOW")
        refit_full_raw = os.getenv("REFIT_FULL_AFTER_EVAL", "1").strip().lower()
        refit_full_after_eval = refit_full_raw not in {"0", "false", "no", "off"}
        use_realtime_raw = os.getenv("USE_REALTIME_LAST_CLOSE", "1").strip().lower()
        use_realtime_last_close = use_realtime_raw not in {"0", "false", "no", "off"}
        return Settings(
            fred_api_key=os.getenv("FRED_API_KEY"),
            fred_cache_dir=os.getenv("FRED_CACHE_DIR", "artifacts/fred_cache"),
            fred_cache_ttl_days=int(os.getenv("FRED_CACHE_TTL_DAYS", "7")),
            twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY"),
            twelve_data_symbol=os.getenv("TWELVE_DATA_SYMBOL", "XAU/USD"),
            twelve_cache_dir=os.getenv("TWELVE_CACHE_DIR", "artifacts/twelve_cache"),
            twelve_cache_ttl_minutes=int(os.getenv("TWELVE_CACHE_TTL_MINUTES", "15")),
            gold_history_source=os.getenv("GOLD_HISTORY_SOURCE", "twelve"),
            market_factors=_parse_list(os.getenv("MARKET_FACTORS"), DEFAULT_MARKET_FACTORS),
            smtp_host=os.getenv("SMTP_HOST"),
            smtp_port=int(smtp_port_raw) if smtp_port_raw else None,
            smtp_user=os.getenv("SMTP_USER"),
            smtp_pass=os.getenv("SMTP_PASS"),
            smtp_from=os.getenv("SMTP_FROM"),
            smtp_to=os.getenv("SMTP_TO"),
            smtp_use_ssl=_parse_bool(os.getenv("SMTP_USE_SSL"), True),
            smtp_starttls=_parse_bool(os.getenv("SMTP_STARTTLS"), False),
            http_timeout=int(os.getenv("HTTP_TIMEOUT", "30")),
            http_retries=int(os.getenv("HTTP_RETRIES", "3")),
            http_backoff=float(os.getenv("HTTP_BACKOFF", "0.5")),
            artifact_dir=os.getenv("ARTIFACT_DIR", "artifacts"),
            target_column=os.getenv("TARGET_COLUMN", "gold_ohlc4"),
            target_columns=_parse_list(
                os.getenv("TARGET_COLUMNS"),
                [
                    "gold_open",
                    "gold_high",
                    "gold_low",
                    "gold",
                    "gold_hl2",
                    "gold_hlc3",
                    "gold_ohlc4",
                    "gold_oc2",
                ],
            ),
            train_window=int(train_window_raw) if train_window_raw else None,
            refit_full_after_eval=refit_full_after_eval,
            use_realtime_last_close=use_realtime_last_close,
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            llm_min_interval_seconds=int(os.getenv("LLM_MIN_INTERVAL_SECONDS", "8")),
            llm_lock_timeout_seconds=int(os.getenv("LLM_LOCK_TIMEOUT_SECONDS", "300")),
            llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "128000")),
            llm_max_input_chars=int(os.getenv("LLM_MAX_INPUT_CHARS", "128000")),
            llm_report_language=os.getenv("LLM_REPORT_LANGUAGE", "Chinese"),
            llm_empty_retry_count=int(os.getenv("LLM_EMPTY_RETRY_COUNT", "2")),
            llm_empty_retry_delay_seconds=int(os.getenv("LLM_EMPTY_RETRY_DELAY_SECONDS", "5")),
        )


DEFAULT_MACRO_FACTORS = [
    "DGS3MO",
    "DGS10",
    "DGS2",
    "DGS5",
    "DGS30",
    "T5YIE",
    "T10YIE",
    "CPIAUCSL",
    "CPILFESL",
    "PCEPI",
    "UNRATE",
    "PAYEMS",
    "ICSA",
    "FEDFUNDS",
    "SOFR",
    "M2SL",
    "TOTLL",
    "WALCL",
    "WRESBAL",
    "RRPONTSYD",
    "BOGMBASE",
    "GFDEBTN",
    "BAMLH0A0HYM2",
    "BAA10Y",
    "USSLIND",
    "USREC",
    "UMCSENT",
    "HOUST",
    "PERMIT",
    "SPCS20RSA",
    "DTWEXBGS",
    "DCOILWTICO",
]

DEFAULT_MARKET_FACTORS = [
    "usdidx",
    "spx",
    "dji",
    "ndx",
    "vix",
]

FACTOR_DESCRIPTIONS: dict[str, str] = {
    "DGS10": "US 10-year Treasury yield",
    "DGS3MO": "US 3-month Treasury yield",
    "DGS2": "US 2-year Treasury yield",
    "DGS5": "US 5-year Treasury yield",
    "DGS30": "US 30-year Treasury yield",
    "T5YIE": "5-year inflation expectations (breakeven)",
    "T10YIE": "10-year inflation expectations (breakeven)",
    "CPIAUCSL": "US CPI (all urban consumers)",
    "CPILFESL": "US core CPI (ex food & energy)",
    "PCEPI": "US PCE price index",
    "UNRATE": "US unemployment rate",
    "PAYEMS": "US nonfarm payrolls",
    "ICSA": "US initial jobless claims",
    "FEDFUNDS": "Federal funds target rate",
    "SOFR": "Secured Overnight Financing Rate (SOFR)",
    "M2SL": "US M2 money supply",
    "TOTLL": "Commercial bank loans and leases",
    "WALCL": "Federal Reserve balance sheet (total assets)",
    "WRESBAL": "Reserve balances of depository institutions",
    "RRPONTSYD": "Overnight reverse repo (ON RRP)",
    "BOGMBASE": "Monetary base (adjusted monetary base)",
    "GFDEBTN": "US federal debt held by the public",
    "BAMLH0A0HYM2": "US high-yield credit spread",
    "BAA10Y": "BAA corporate - 10-year Treasury spread",
    "USSLIND": "US leading economic index",
    "USREC": "US recession indicator (0/1)",
    "UMCSENT": "University of Michigan consumer sentiment",
    "HOUST": "US housing starts",
    "PERMIT": "US building permits",
    "SPCS20RSA": "S&P Case-Shiller 20-city home price index",
    "DTWEXBGS": "Broad real effective exchange rate (USD)",
    "DCOILWTICO": "WTI crude oil spot price",
    "usdidx": "US dollar index (USD)",
    "spx": "S&P 500 index",
    "dji": "Dow Jones Industrial Average",
    "ndx": "Nasdaq 100 index",
    "vix": "VIX volatility index",
    "gold_open": "Gold open price",
    "gold_high": "Gold high price",
    "gold_low": "Gold low price",
    "gold": "Gold close price",
    "gold_hl2": "Gold average price (high+low)/2",
    "gold_hlc3": "Gold average price (high+low+close)/3",
    "gold_ohlc4": "Gold average price (open+high+low+close)/4",
    "gold_oc2": "Gold average price (open+close)/2",
    "gold_range": "Gold intraday range (high-low)",
    "gold_body": "Gold candle body (close-open)",
    "gold_hl_pct": "Gold intraday range pct (high-low)/open",
    "gold_oc_pct": "Gold intraday return (close-open)/open",
    "gold_std7": "Gold price 7-day volatility (std dev)",
    "gold_std30": "Gold price 30-day volatility (std dev)",
    "gold_pct": "Gold daily return (close)",
    "gold_ma30": "Gold 30-day moving average (close)",
    "liquidity_index": "Liquidity index (standardized M2, bank loans, Fed assets, reserves, RRP, etc.)",
    "debt_index": "Debt index (standardized federal public debt)",
    "neutral_balance": "Liquidity-debt balance (liquidity index - debt index)",
    "yc_10y2y": "Yield curve spread (10y-2y)",
    "yc_10y3m": "Yield curve spread (10y-3m)",
    "yc_10y5y": "Yield curve spread (10y-5y)",
    "yc_30y10y": "Yield curve spread (30y-10y)",
    "yc_10y2y_chg": "Yield curve spread change (10y-2y)",
    "yc_10y2y_ma30": "Yield curve spread 30-day mean (10y-2y)",
    "yc_10y2y_std30": "Yield curve spread 30-day volatility (10y-2y)",
    "t5yie_chg": "5-year inflation expectations change",
    "t5yie_ma30": "5-year inflation expectations 30-day mean",
    "t5yie_std30": "5-year inflation expectations 30-day volatility",
    "t10yie_chg": "10-year inflation expectations change",
    "t10yie_ma30": "10-year inflation expectations 30-day mean",
    "t10yie_std30": "10-year inflation expectations 30-day volatility",
    "real_rate_proxy_5y": "Real rate proxy (5y nominal - 5y breakeven)",
    "real_rate_proxy": "Real rate proxy (10y nominal - 10y breakeven)",
    "real_rate_proxy_ma30": "Real rate proxy 30-day mean",
    "real_rate_proxy_chg": "Real rate proxy change",
    "equity_avg_pct": "Average equity daily return (SPX/DJI/NDX)",
    "equity_avg_vol30": "Average equity 30-day return volatility",
    "risk_appetite": "Risk appetite (equity avg return - VIX return)",
    "equity_vs_usd": "Equity strength vs USD (equity avg return - USD index return)",
}
