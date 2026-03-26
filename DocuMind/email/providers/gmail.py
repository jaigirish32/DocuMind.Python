from __future__ import annotations

import base64
import os
from datetime import datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from DocuMind.core.settings import get_settings
from DocuMind.core.logging.logger import get_logger
from DocuMind.email.models import EmailMessage

logger = get_logger(__name__)

# Only read access — we never modify emails
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class GmailProvider:
    """
    Connects to Gmail using OAuth 2.0.
    Fetches emails and returns list of EmailMessage.

    First run: opens browser for Google consent.
    After that: uses saved token — no browser needed.
    """

    def __init__(self) -> None:
        settings                = get_settings()
        self._credentials_file  = settings.gmail_credentials_file
        self._token_file        = settings.gmail_token_file
        self._service           = None

    def authenticate(self) -> None:
        """
        Authenticate with Google OAuth 2.0.
        Opens browser on first run to get user consent.
        Saves token to file so subsequent runs are silent.
        """
        creds = None

        # Load saved token if it exists
        if Path(self._token_file).exists():
            creds = Credentials.from_authorized_user_file(
                self._token_file, SCOPES
            )

        # If no valid token — get one
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                # Silently refresh expired token
                creds.refresh(Request())
                logger.info("Gmail token refreshed")
            else:
                # First time — open browser for consent
                flow = InstalledAppFlow.from_client_secrets_file(
                    self._credentials_file, SCOPES
                )
                creds = flow.run_local_server(port=0)
                logger.info("Gmail OAuth consent completed")

            # Save token for next run
            with open(self._token_file, "w") as f:
                f.write(creds.to_json())

        self._service = build("gmail", "v1", credentials=creds)
        logger.info("Gmail connected")

    def fetch_emails(
        self,
        max_results: int         = 50,
        query:       str         = "",
        label:       str         = "INBOX",
    ) -> list[EmailMessage]:
        """
        Fetch emails from Gmail.

        Args:
            max_results: max number of emails to fetch
            query:       Gmail search query e.g. "from:john@company.com"
            label:       Gmail label e.g. "INBOX", "SENT", "IMPORTANT"

        Returns:
            list of EmailMessage
        """
        if self._service is None:
            self.authenticate()

        logger.info("Fetching emails", max_results=max_results, query=query)

        # Step 1 — get list of message IDs
        result = self._service.users().messages().list(
            userId     = "me",
            maxResults = max_results,
            q          = query,
            labelIds   = [label],
        ).execute()

        message_ids = result.get("messages", [])
        if not message_ids:
            logger.info("No emails found")
            return []

        # Step 2 — fetch full content for each message
        emails = []
        for item in message_ids:
            try:
                msg = self._service.users().messages().get(
                    userId = "me",
                    id     = item["id"],
                    format = "full",
                ).execute()
                email = self._parse(msg)
                if email:
                    emails.append(email)
            except Exception as e:
                logger.warning("Failed to fetch email", id=item["id"], error=str(e))

        logger.info("Emails fetched", count=len(emails))
        return emails

    # ── Private ───────────────────────────────────────────────────────────────

    def _parse(self, msg: dict) -> EmailMessage | None:
        """Convert raw Gmail API message into EmailMessage."""
        try:
            headers = {
                h["name"].lower(): h["value"]
                for h in msg["payload"]["headers"]
            }

            subject    = headers.get("subject", "(no subject)")
            sender     = headers.get("from", "")
            to         = headers.get("to", "")
            recipients = [r.strip() for r in to.split(",")] if to else []
            date       = self._parse_date(headers.get("date", ""))
            body       = self._extract_body(msg["payload"])
            labels     = msg.get("labelIds", [])

            return EmailMessage(
                message_id = msg["id"],
                thread_id  = msg["threadId"],
                subject    = subject,
                sender     = sender,
                recipients = recipients,
                body       = body,
                date       = date,
                labels     = labels,
            )
        except Exception as e:
            logger.warning("Failed to parse email", error=str(e))
            return None

    def _extract_body(self, payload: dict) -> str:
        """Extract plain text body from Gmail message payload."""
        # Direct body
        if "body" in payload and payload["body"].get("data"):
            return self._decode(payload["body"]["data"])

        # Multipart — find text/plain part
        for part in payload.get("parts", []):
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    return self._decode(data)

        # Fallback — try first part
        parts = payload.get("parts", [])
        if parts:
            data = parts[0].get("body", {}).get("data", "")
            if data:
                return self._decode(data)

        return ""

    @staticmethod
    def _decode(data: str) -> str:
        """Decode base64url encoded Gmail body."""
        try:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8")
        except Exception:
            return ""

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """Parse Gmail date string into datetime."""
        from email.utils import parsedate_to_datetime
        try:
            return parsedate_to_datetime(date_str)
        except Exception:
            return datetime.utcnow()