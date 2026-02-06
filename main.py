from __future__ import annotations

import argparse
from datetime import date, datetime
import logging
import json
from dataclasses import asdict

from pathlib import Path

import markdown

from config import Settings
from data_sources import NewsClient, NewsSource
from llm_client import LLMClient
from notification import EmailConfig, EmailNotifier
from pipeline import GoldValuationPipeline, format_report, format_report_multi
from scheduler import ScheduleConfig, Scheduler


class DateEncoder(json.JSONEncoder):
    """JSON encoder for date/datetime objects."""
    def default(self, o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return super().default(o)


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def persist_report(settings: Settings, report: str, extension: str = "txt") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(settings.artifact_dir).mkdir(parents=True, exist_ok=True)
    path = Path(settings.artifact_dir) / f"report_{timestamp}.{extension}"
    path.write_text(report, encoding="utf-8")
    return path


def run_once(
    settings: Settings,
    start: date | None,
    end: date | None,
    send_email: bool,
    subject: str,
    frequency: str,
    horizons: list[int],
    output_format: str = "text",
    optimize: bool = False,
) -> None:
    pipeline = GoldValuationPipeline(settings)
    if settings.target_columns and len(settings.target_columns) > 1:
        result = pipeline.run_multi(start, end, frequency=frequency, horizons=horizons)
        raw_report = format_report_multi(result)
    else:
        result = pipeline.run(start, end, frequency=frequency, horizons=horizons)
        raw_report = format_report(result)

    report_content = raw_report
    extension = "txt"
    structured_payload: str | None = None
    data_dict: dict | None = None
    news_payload: dict | None = None

    try:
        data_dict = asdict(result)
        structured_payload = json.dumps(data_dict, cls=DateEncoder, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning("Failed to build structured payload: %s", e)

    if settings.news_enabled and settings.news_sources:
        try:
            sources: list[NewsSource] = []
            for raw_source in settings.news_sources:
                if not isinstance(raw_source, dict):
                    logging.warning(
                        "Skipping invalid news source configuration (not a dict): %r",
                        raw_source,
                    )
                    continue
                try:
                    source = NewsSource(**raw_source)
                except Exception as e:
                    logging.warning(
                        "Skipping malformed news source configuration %r: %s",
                        raw_source,
                        e,
                    )
                    continue
                sources.append(source)
            if sources:
                news_client = NewsClient(
                    sources=sources,
                    timeout=settings.http_timeout,
                    cache_dir=settings.news_cache_dir,
                    cache_ttl_hours=settings.news_cache_ttl_hours,
                    retries=max(1, settings.http_retries),
                    backoff=settings.http_backoff,
                    min_interval_seconds=settings.news_min_interval_seconds,
                    max_total_items=settings.news_max_total_items,
                    max_items_per_category=settings.news_max_items_per_category,
                    category_limits=settings.news_category_limits,
                )
                news_payload = news_client.fetch_news(
                    limit_per_source=max(1, settings.news_max_items_per_source)
                )
        except Exception as e:
            logging.warning("Failed to fetch news payload: %s", e)

    if output_format == "json":
        if structured_payload is not None:
            if news_payload is not None and data_dict is not None:
                data_dict["news"] = news_payload
                structured_payload = json.dumps(data_dict, cls=DateEncoder, ensure_ascii=False, indent=2)
            report_content = structured_payload
            extension = "json"
        else:
            logging.error("Failed to serialize result to JSON. Falling back to text.")
            output_format = "text"

    if output_format == "text" and optimize:
        if news_payload is not None and structured_payload is not None and data_dict is not None:
            llm_news_payload = NewsClient.strip_for_llm_payload(news_payload)
            data_dict["news"] = llm_news_payload
            structured_payload = json.dumps(
                data_dict, cls=DateEncoder, ensure_ascii=False, indent=2
            )
        llm_client = LLMClient(settings)
        logging.info("Optimizing report with LLM...")
        report_content = llm_client.optimize_report(raw_report, structured_payload)
        extension = "md"
    elif output_format == "text" and news_payload is not None:
        news_text = json.dumps(news_payload, ensure_ascii=False, indent=2)
        report_content = f"{raw_report}\n\nNEWS_JSON:\n{news_text}"

    print(report_content)

    if send_email:
        missing = [
            key
            for key, value in {
                "SMTP_HOST": settings.smtp_host,
                "SMTP_PORT": settings.smtp_port,
                "SMTP_USER": settings.smtp_user,
                "SMTP_PASS": settings.smtp_pass,
                "SMTP_FROM": settings.smtp_from,
                "SMTP_TO": settings.smtp_to,
            }.items()
            if not value
        ]
        if missing:
            path = persist_report(settings, report_content, extension)
            logging.warning("Missing email settings: %s, report saved to %s", ", ".join(missing), path)
            return

        email_body = report_content
        is_html = False
        if extension == "md":
            try:
                html_content = markdown.markdown(report_content, extensions=['tables', 'fenced_code'])
                email_body = """
                <html>
                <head>
                    <style>
                        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
                        h1 {{ border-bottom: 2px solid #eaeaea; padding-bottom: 0.3em; }}
                        h2 {{ border-bottom: 1px solid #eaeaea; padding-bottom: 0.3em; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f7f7f7; font-weight: bold; }}
                        tr:nth-child(even) {{ background-color: #f8f8f8; }}
                        code {{ background-color: #f1f1f1; padding: 2px 4px; border-radius: 4px; font-family: monospace; }}
                        blockquote {{ border-left: 4px solid #ddd; padding-left: 15px; color: #666; margin: 0; }}
                    </style>
                </head>
                <body>
                {html_content}
                <br>
                <hr>
                <p style="font-size: 0.8em; color: #777;">Generated by Gold Valuation Framework AI</p>
                </body>
                </html>
                """.format(html_content=html_content)
                is_html = True
            except Exception as e:
                logging.warning("Markdown conversion failed: %s, sending as plain text", e)

        notifier = EmailNotifier(
            EmailConfig(
                host=settings.smtp_host or "",
                port=settings.smtp_port or 587,
                user=settings.smtp_user or "",
                password=settings.smtp_pass or "",
                sender=settings.smtp_from or "",
                recipient=settings.smtp_to or "",
                use_ssl=settings.smtp_use_ssl,
                starttls=settings.smtp_starttls,
            )
        )
        notifier.send(subject, email_body, is_html=is_html)
    else:
        persist_report(settings, report_content, extension)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Gold Valuation Framework")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--send-email", action="store_true", help="Send email report")
    parser.add_argument("--subject", default="Gold Valuation Report", help="Email subject")
    parser.add_argument("--daemon", action="store_true", help="Run continuously on a schedule")
    parser.add_argument("--interval-minutes", type=int, default=60, help="Schedule interval in minutes")
    parser.add_argument("--freq", default="day", help="Data frequency: day or 1min")
    parser.add_argument("--horizons", default="1,5,20", help="Forecast horizons in trading days (comma-separated)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--optimize", action="store_true", help="Optimize report with LLM (requires text format)")
    args = parser.parse_args()

    settings = Settings.from_env()
    start = parse_date(args.start)
    end = parse_date(args.end)

    horizons = [int(x) for x in args.horizons.split(",") if x.strip().isdigit() and int(x) > 0]

    if args.daemon:
        scheduler = Scheduler(
            ScheduleConfig(interval_minutes=max(1, args.interval_minutes)),
            job=lambda: run_once(
                settings, start, end, args.send_email, args.subject, args.freq, horizons,
                output_format=args.format, optimize=args.optimize
            ),
        )
        scheduler.run_forever()
    else:
        run_once(
            settings, start, end, args.send_email, args.subject, args.freq, horizons,
            output_format=args.format, optimize=args.optimize
        )


if __name__ == "__main__":
    main()
