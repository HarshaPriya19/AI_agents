from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any, Dict, List, Optional

from meeting_models import MeetingReport, ActionItem

from telegram import Bot


@dataclass
class WorkflowResult:
    """
    Generic envelope for workflow executions so the API can return
    a consistent payload regardless of the downstream provider.
    """

    success: bool
    details: Dict[str, Any]


def _format_action_items_for_text(action_items: List[ActionItem]) -> str:
    lines: List[str] = []
    for idx, item in enumerate(action_items, start=1):
        lines.append(
            f"{idx}. Task: {item.task}\n"
            f"   Owner: {item.owner}\n"
            f"   Deadline: {item.deadline}"
        )
    return "\n".join(lines) if lines else "No explicit action items were identified."


def send_telegram_summary(report: MeetingReport) -> WorkflowResult:
    """
    Send the meeting summary and action items to a specific Telegram chat
    using the Telegram Bot API.

    Requires the following environment variables:
    - TELEGRAM_BOT_TOKEN
    - TELEGRAM_CHAT_ID
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return WorkflowResult(
            success=False,
            details={
                "error": "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in the environment."
            },
        )

    bot = Bot(token=token)

    summary_text = (
        "Meeting Summary\n"
        "================\n\n"
        f"{report.summary}\n\n"
        "Key Decisions:\n"
        "- " + "\n- ".join(report.decisions or ["No major decisions recorded."]) + "\n\n"
        "Action Items:\n"
        f"{_format_action_items_for_text(report.action_items)}"
    )

    bot.send_message(chat_id=chat_id, text=summary_text)

    return WorkflowResult(
        success=True,
        details={
            "provider": "telegram",
            "chat_id": chat_id,
            "message_preview": summary_text[:2000],
        },
    )


def send_email_draft(
    report: MeetingReport,
    to_address: Optional[str] = None,
) -> WorkflowResult:
    """
    Create and optionally send a professional summary email based on the
    MeetingReport. If SMTP configuration is missing, this function will
    only return the composed email as structured JSON without sending.

    Optional environment variables:
    - SMTP_HOST
    - SMTP_PORT (default 587)
    - SMTP_USERNAME
    - SMTP_PASSWORD
    - EMAIL_FROM
    - EMAIL_TO (used if to_address is not provided)
    """
    subject = "Meeting Summary and Action Items"

    body_lines = [
        "Dear team,",
        "",
        "Here is a concise summary of our recent meeting, including key decisions and concrete action items.",
        "",
        "Summary",
        "-------",
        report.summary,
        "",
        "Key Decisions",
        "-------------",
    ]

    body_lines.extend(["- " + d for d in (report.decisions or ["No major decisions recorded."])])
    body_lines.append("")
    body_lines.append("Action Items")
    body_lines.append("------------")

    if report.action_items:
        for idx, item in enumerate(report.action_items, start=1):
            body_lines.append(
                f"{idx}. {item.task} "
                f"(Owner: {item.owner}; Deadline: {item.deadline})"
            )
    else:
        body_lines.append("No explicit action items were identified.")

    body_lines.append("")
    body_lines.append("Best regards,")
    body_lines.append("Meeting Intelligence Agent")

    body = "\n".join(body_lines)

    email_payload = {
        "subject": subject,
        "body": body,
    }

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_address = os.getenv("EMAIL_FROM") or smtp_username
    # Default recipient if nothing else is provided.
    to_address = to_address or os.getenv("EMAIL_TO") or "harshapriyaputta@gmail.com"

    # If we do not have minimal SMTP config, just return the draft.
    # Note: to_address always has a fallback, so SMTP_HOST + FROM are the key requirements.
    if not (smtp_host and from_address):
        return WorkflowResult(
            success=False,
            details={
                "reason": "Email draft created but SMTP configuration is incomplete.",
                "email": {
                    "from": from_address,
                    "to": to_address,
                    "subject": subject,
                    "body": body,
                },
            },
        )

    # Compose and send via SMTP (plain TLS).
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_address
    msg["To"] = to_address
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        if smtp_username and smtp_password:
            server.login(smtp_username, smtp_password)
        server.send_message(msg)

    return WorkflowResult(
        success=True,
        details={
            "provider": "smtp",
            "host": smtp_host,
            "port": smtp_port,
            "from": from_address,
            "to": to_address,
            "subject": subject,
        },
    )


def create_calendar_reminder(report: MeetingReport) -> WorkflowResult:
    """
    Generate a structured JSON object representing a calendar reminder for
    the earliest deadline mentioned in the action items.

    We intentionally keep this as JSON so that a separate integration layer
    can map it onto Google Calendar, Outlook, or any other provider.
    """
    earliest_deadline: Optional[str] = None
    earliest_item: Optional[ActionItem] = None

    for item in report.action_items:
        # Heuristic: treat the first non-empty deadline as "earliest" if we
        # do not have a proper date parser available in this layer.
        if item.deadline and not earliest_deadline:
            earliest_deadline = item.deadline
            earliest_item = item

    if not earliest_item:
        return WorkflowResult(
            success=False,
            details={
                "error": "No action items with deadlines found to build a reminder.",
            },
        )

    reminder = {
        "title": f"Reminder: {earliest_item.task}",
        "description": (
            "Automatically generated reminder from Meeting Intelligence Agent.\n\n"
            f"Task: {earliest_item.task}\n"
            f"Owner: {earliest_item.owner}\n"
            f"Deadline (natural language): {earliest_item.deadline}"
        ),
        "deadline_text": earliest_item.deadline,
        "source": "meeting_intelligence_agent",
    }

    return WorkflowResult(
        success=True,
        details={"reminder": reminder},
    )

