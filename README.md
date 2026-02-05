# Gold Valuation Framework

## Features

- Automatically discovers and organizes macro and market factors related to gold (defaults included, extensible)
- Fetches macro/market data and historical gold prices
- Trains multi-target models and predicts next-period OHLC/aggregate targets
- Retrieves realtime gold prices for comparison and feedback
- Generates reports and optionally sends them by email

## Quick Start

1) Install dependencies
   - Use uv or pip (dependencies are listed in pyproject.toml)
2) Run
   - python main.py --start 2015-01-01 --end 2025-12-31
   - python main.py --send-email
   - python main.py --daemon --interval-minutes 60 --send-email
   - python main.py --start 2026-02-01 --end 2026-02-02 --freq 1min
   - python main.py --start 2015-01-01 --end 2025-12-31 --horizons 1,5,20,60

## Environment Variables

- FRED_API_KEY: FRED macro data API key (optional; if unset, macro factors are skipped)
- FRED_CACHE_DIR: FRED cache directory (optional, default artifacts/fred_cache)
- FRED_CACHE_TTL_DAYS: FRED cache TTL in days (default 7)
- HTTP_TIMEOUT: Request timeout in seconds (optional, default 20)
- ARTIFACT_DIR: Local persistence directory (optional, default artifacts)
- TWELVE_DATA_API_KEY: Twelve Data API key (required for 1-minute data)
- TWELVE_DATA_SYMBOL: Twelve Data symbol (default XAU/USD)
- TWELVE_CACHE_DIR: Twelve Data cache directory (default artifacts/twelve_cache)
- TWELVE_CACHE_TTL_MINUTES: Twelve Data cache TTL in minutes (default 15)
- SMTP_HOST: SMTP host
- SMTP_PORT: SMTP port
- SMTP_USER: SMTP username
- SMTP_PASS: SMTP password
- SMTP_FROM: Sender email
- SMTP_TO: Recipient email

## Notes

- Macro factors are sourced from FRED (API key required)
- Historical gold prices and some market factors are from stooq
- Realtime gold price uses metals.live (falls back to Twelve Data on failure)
- Use --daemon for persistent runs with interval-minutes scheduling
- Factors, raw data, and models are cached in ARTIFACT_DIR and updated incrementally
- If email is not configured or sending is disabled, reports are saved locally to ARTIFACT_DIR
- 1-minute frequency uses Twelve Data (--freq 1min) and skips macro/market factors

## Practice Note: Handling Non-Daily Factors

- Some macro factors are published monthly/quarterly; naive daily alignment causes missing values at the latest date.
- Use augmentation (forward-fill and sensible missing-value handling) to keep the latest date usable; **do not** drop these factors outright.
