from __future__ import annotations

import argparse
from datetime import date, datetime
import logging

from pathlib import Path

from config import Settings
from notification import EmailConfig, EmailNotifier
from pipeline import GoldValuationPipeline, format_report, format_report_multi
from scheduler import ScheduleConfig, Scheduler


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def persist_report(settings: Settings, report: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(settings.artifact_dir).mkdir(parents=True, exist_ok=True)
    path = Path(settings.artifact_dir) / f"report_{timestamp}.txt"
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
) -> None:
    pipeline = GoldValuationPipeline(settings)
    if settings.target_columns and len(settings.target_columns) > 1:
        result = pipeline.run_multi(start, end, frequency=frequency, horizons=horizons)
        report = format_report_multi(result)
    else:
        result = pipeline.run(start, end, frequency=frequency, horizons=horizons)
        report = format_report(result)

    print(report)

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
            path = persist_report(settings, report)
            logging.warning("Missing email settings: %s, report saved to %s", ", ".join(missing), path)
            return

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
        notifier.send(subject, report)
    else:
        persist_report(settings, report)


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
    args = parser.parse_args()

    settings = Settings.from_env()
    start = parse_date(args.start)
    end = parse_date(args.end)

    horizons = [int(x) for x in args.horizons.split(",") if x.strip().isdigit() and int(x) > 0]

    if args.daemon:
        scheduler = Scheduler(
            ScheduleConfig(interval_minutes=max(1, args.interval_minutes)),
            job=lambda: run_once(
                settings, start, end, args.send_email, args.subject, args.freq, horizons
            ),
        )
        scheduler.run_forever()
    else:
        run_once(settings, start, end, args.send_email, args.subject, args.freq, horizons)


if __name__ == "__main__":
    main()
