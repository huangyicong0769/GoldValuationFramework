from __future__ import annotations

import logging
import os
from pathlib import Path
import time
import re
from openai import OpenAI
from config import Settings

LOGGER = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = None
        self.lock_path = Path(settings.artifact_dir) / "llm_api.lock"
        self.last_call_path = Path(settings.artifact_dir) / "llm_api.last"
        self.min_interval_seconds = max(0, int(settings.llm_min_interval_seconds))
        self.lock_timeout_seconds = max(1, int(settings.llm_lock_timeout_seconds))
        self.max_output_tokens = max(2560, int(settings.llm_max_output_tokens))
        self.max_input_chars = max(10000, int(settings.llm_max_input_chars))
        self.empty_retry_count = max(0, int(settings.llm_empty_retry_count))
        self.empty_retry_delay_seconds = max(1, int(settings.llm_empty_retry_delay_seconds))
        if settings.llm_api_key:
            self.client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            )
        else:
            LOGGER.warning("LLM API key (LLM_API_KEY) not found. LLM features will be disabled.")

    def _acquire_lock(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(f"{os.getpid()}\n{time.time()}\n")
                return True
            except FileExistsError:
                try:
                    pid, lock_ts = self._read_lock_info()
                    if pid is not None and not self._is_process_alive(pid):
                        LOGGER.warning("Stale LLM lock (dead pid %s), removing: %s", pid, self.lock_path)
                        self.lock_path.unlink(missing_ok=True)
                        continue
                    if lock_ts is not None and time.time() - lock_ts > self.lock_timeout_seconds:
                        LOGGER.warning("Stale LLM lock (timeout), removing: %s", self.lock_path)
                        self.lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(0.5)

    def _release_lock(self) -> None:
        try:
            self.lock_path.unlink(missing_ok=True)
        except Exception as exc:
            LOGGER.warning("Failed to release LLM lock: %s", exc)

    def _apply_min_interval(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        try:
            if self.last_call_path.exists():
                last_value = self.last_call_path.read_text(encoding="utf-8").strip()
                last_ts = float(last_value) if last_value else 0.0
                wait_seconds = self.min_interval_seconds - (time.time() - last_ts)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
        except Exception as exc:
            LOGGER.warning("Failed to enforce LLM min interval: %s", exc)

    def _record_last_call(self) -> None:
        try:
            self.last_call_path.write_text(str(time.time()), encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("Failed to record LLM last call: %s", exc)

    def _read_lock_info(self) -> tuple[int | None, float | None]:
        try:
            raw = self.lock_path.read_text(encoding="utf-8").strip().splitlines()
        except FileNotFoundError:
            return None, None
        except Exception as exc:
            LOGGER.warning("Failed to read LLM lock file: %s", exc)
            return None, None

        pid = None
        ts = None
        if raw:
            try:
                pid = int(raw[0].strip())
            except (TypeError, ValueError):
                pid = None
        if len(raw) > 1:
            try:
                ts = float(raw[1].strip())
            except (TypeError, ValueError):
                ts = None
        return pid, ts

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


    def optimize_report(self, raw_report: str, structured_payload: str | None = None) -> str:
        if not self.client:
            LOGGER.warning("LLM client not initialized. Returning raw report.")
            return raw_report

        system_prompt = (
            "You are a professional gold market analyst and quantitative researcher. "
            "Rewrite the raw, structured and unstructured report into a polished market analysis. "
            "Write in the specified OUTPUT_LANGUAGE. Do not use code fences (no ```). \n\n"
            "使用带有清晰规范标题的 Markdown，列表/表格前后留空行，表格对齐规范。"
            "***绝不能输出单行报告***。空格和换行务必符合 Markdown 规范。\n\n"
            "Use Markdown with clear headings, blank lines around lists/tables, and properly aligned tables. "
            "***Never output a single-line report***; Space and line breaks must comply with Markdown specifications.\n\n"
            # "每个要点必须单独成行。"
            # "每个表格必须是标准 Markdown 表格，包含表头行、分隔行，且每条数据行独立成行。"
            # "不要把多个标题或句子写在同一行。"
            # "不要把表格单元格拆到多行；每一行应保持为单行。"
            "Use prompt-engineering best practices: be concise, structured, and evidence-based. "
            "Avoid verbatim copying; synthesize and reorganize.\n\n"
            "Provide deeper, richer analysis with these required sections and checks: \n\n"
            "1) Key Numbers Interpretation: explain latest price, forecasts, errors, and realtime adjustments. \n"
            "2) Structured Attribution: link factor importance to short/medium/long-term differences. \n"
            "3) Ranges & Scenarios: core range, upside/downside triggers, and risk points. \n"
            "4) Confidence & Limitations: error bands, sample coverage limits, abnormal volatility impacts. \n"
            "5) Strategy Guidance: separate short/medium/long horizons with risk controls. \n\n"
            "Use structured JSON as the primary source of truth; use raw text only to clarify context. "
            "You may use simple Unicode trend markers or text charts; avoid external images or links.\n\n"
            "No next steps or follow-up suggestions. Focus on delivering a comprehensive, standalone report."
        )

        output_language = self.settings.llm_report_language
        if structured_payload:
            user_prompt = (
                f"OUTPUT_LANGUAGE: {output_language}\n\n"
                "Structured JSON (primary source of truth):\n\n"
                f"{structured_payload}\n\n"
                "Raw text report (context only):\n\n"
                f"{raw_report}\n\n"
                "Generate the final Markdown report per instructions.\n\n"
            )
        else:
            user_prompt = (
                f"OUTPUT_LANGUAGE: {output_language}\n\n"
                "Raw text report:\n\n"
                f"{raw_report}\n\n"
                "Generate the final Markdown report per instructions.\n\n"
            )

        if len(user_prompt) > self.max_input_chars:
            user_prompt = user_prompt[: self.max_input_chars]

        self._acquire_lock()
        try:
            self._apply_min_interval()
            attempts = self.empty_retry_count + 1
            for attempt in range(attempts):
                response = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.max_output_tokens,
                )
                content = response.choices[0].message.content
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text") or item.get("content") or ""
                            if text:
                                parts.append(str(text))
                    content = "".join(parts)
                if content:
                    self._record_last_call()
                    return str(content)

                if attempt < attempts - 1:
                    LOGGER.warning("LLM response empty; retrying in %s seconds.", self.empty_retry_delay_seconds)
                    time.sleep(self.empty_retry_delay_seconds)
                    continue

                LOGGER.warning("LLM response empty after retries; falling back to raw report.")
                self._record_last_call()
                return raw_report
            return raw_report
        except Exception as e:
            LOGGER.error("Error calling LLM: %s", e)
            return raw_report
        finally:
            self._release_lock()
