# Gold Valuation Framework

## Features

- Automatically discovers and organizes macro and market factors related to gold (defaults included, extensible)
- Fetches macro/market data and historical gold prices
- Trains multi-target models and predicts next-period OHLC/aggregate targets
- Retrieves realtime gold prices for comparison and feedback
- Generates reports and optionally sends them by email
- Optional LLM-optimized reports (Markdown + HTML email)
- Structured JSON output for downstream use

## Quick Start

1) Install dependencies
   - Use uv or pip (dependencies are listed in pyproject.toml)
2) Run
   - uv run main.py --start 2015-01-01 --end 2025-12-31
   - uv run main.py --send-email
   - uv run main.py --daemon --interval-minutes 60 --send-email
   - uv run main.py --start 2026-02-01 --end 2026-02-02 --freq 1min
   - uv run main.py --start 2015-01-01 --end 2025-12-31 --horizons 1,5,20,60
   - uv run main.py --start 2015-01-01 --end 2025-12-31 --format json
   - uv run main.py --start 2015-01-01 --end 2025-12-31 --optimize

## Output

- Default output is plain text; use --format json for structured output.
- LLM optimization is enabled via --optimize (text output only).
- Optimized reports are saved as .md and rendered to HTML when emailing.
- Reports are saved under ARTIFACT_DIR as report_YYYYMMDD_HHMMSS.{txt,json,md}.

## Environment Variables

- FRED_API_KEY: FRED macro data API key (optional; if unset, macro factors are skipped)
- FRED_CACHE_DIR: FRED cache directory (optional, default artifacts/fred_cache)
- FRED_CACHE_TTL_DAYS: FRED cache TTL in days (default 7)
- HTTP_TIMEOUT: Request timeout in seconds (optional, default 20)
- ARTIFACT_DIR: Local persistence directory (optional, default artifacts)
- GOLD_HISTORY_SOURCE: Gold history source (default twelve)
- MARKET_FACTORS: Comma-separated market factors override (optional)
- TARGET_COLUMN: Single target column (default gold_ohlc4)
- TARGET_COLUMNS: Comma-separated multi-target columns (optional)
- TRAIN_WINDOW: Train window size (optional)
- REFIT_FULL_AFTER_EVAL: Refit full dataset after evaluation (default true)
- USE_REALTIME_LAST_CLOSE: Use realtime last close to adjust latest prediction (default true)
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
- SMTP_USE_SSL: SMTP SSL mode (default true)
- SMTP_STARTTLS: SMTP STARTTLS (default false)
- HTTP_TIMEOUT: Request timeout in seconds (default 30)
- HTTP_RETRIES: Request retry count (default 3)
- HTTP_BACKOFF: Retry backoff in seconds (default 0.5)
- LLM_API_KEY: LLM API key (required for --optimize)
- LLM_BASE_URL: LLM base URL (optional, OpenAI-compatible)
- LLM_MODEL: LLM model name (default gpt-4o)
- LLM_REPORT_LANGUAGE: Report output language (default Chinese)
- LLM_MAX_OUTPUT_TOKENS: Max output tokens (default 128000)
- LLM_MAX_INPUT_CHARS: Max input chars (default 128000)
- LLM_MIN_INTERVAL_SECONDS: Min interval between LLM calls (default 8)
- LLM_LOCK_TIMEOUT_SECONDS: LLM lock timeout in seconds (default 300)
- LLM_EMPTY_RETRY_COUNT: Empty response retry count (default 2)
- LLM_EMPTY_RETRY_DELAY_SECONDS: Empty response retry delay in seconds (default 5)

## Notes

- Macro factors are sourced from FRED (API key required)
- Historical gold prices and some market factors are from stooq
- Realtime gold price uses metals.live (falls back to Twelve Data on failure)
- Use --daemon for persistent runs with interval-minutes scheduling
- Factors, raw data, and models are cached in ARTIFACT_DIR and updated incrementally
- If email is not configured or sending is disabled, reports are saved locally to ARTIFACT_DIR
- 1-minute frequency uses Twelve Data (--freq 1min) and skips macro/market factors
- LLM optimization requires LLM_API_KEY and uses an OpenAI-compatible API

## Practice Note: Handling Non-Daily Factors

- Some macro factors are published monthly/quarterly; naive daily alignment causes missing values at the latest date.
- Use augmentation (forward-fill and sensible missing-value handling) to keep the latest date usable; **do not** drop these factors outright.
