from __future__ import annotations

from dataclasses import dataclass
import logging
import smtplib
from email.message import EmailMessage


LOGGER = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    host: str
    port: int
    user: str
    password: str
    sender: str
    recipient: str
    use_ssl: bool = False
    starttls: bool = True


class EmailNotifier:
    def __init__(self, config: EmailConfig) -> None:
        self.config = config

    def send(self, subject: str, body: str, is_html: bool = False) -> None:
        message = EmailMessage()
        message["From"] = self.config.sender
        message["To"] = self.config.recipient
        message["Subject"] = subject
        
        if is_html:
            message.set_content(body, subtype="html")
        else:
            message.set_content(body)

        use_ssl = self.config.use_ssl or self.config.port in {465, 994}
        if use_ssl:
            with smtplib.SMTP_SSL(self.config.host, self.config.port, timeout=30) as server:
                server.login(self.config.user, self.config.password)
                server.send_message(message)
        else:
            with smtplib.SMTP(self.config.host, self.config.port, timeout=30) as server:
                if self.config.starttls:
                    server.starttls()
                server.login(self.config.user, self.config.password)
                server.send_message(message)
        LOGGER.info("Email sent to %s", self.config.recipient)
