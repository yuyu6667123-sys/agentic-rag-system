"""SQLite-backed QQ email verification and opaque session management."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from email.message import EmailMessage
import hashlib
import hmac
import logging
import os
from pathlib import Path
import re
import secrets
import smtplib
import sqlite3
from threading import Lock
import time
from typing import Iterator, Protocol

from dotenv import load_dotenv


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QQ_EMAIL_PATTERN = re.compile(r"^[1-9][0-9]{4,11}@qq\.com$", re.IGNORECASE)


class AuthError(RuntimeError):
    """Base error for authentication operations."""


class EmailValidationError(AuthError):
    """Raised when an address is not a valid QQ mailbox."""


class VerificationCooldownError(AuthError):
    """Raised when another code is requested during the cooldown."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"请在 {retry_after_seconds} 秒后重新发送验证码")


class InvalidVerificationCodeError(AuthError):
    """Raised when a verification code is missing, malformed, or incorrect."""


class ExpiredVerificationCodeError(AuthError):
    """Raised when the latest verification code has expired."""


class InvalidSessionError(AuthError):
    """Raised when a session cookie is missing, invalid, or expired."""


class EmailConfigurationError(AuthError):
    """Raised when production SMTP settings are incomplete."""


class EmailDeliveryError(AuthError):
    """Raised when SMTP cannot deliver a verification message."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AuthSettings:
    """Runtime settings loaded from environment variables."""

    db_path: Path
    smtp_host: str = "smtp.qq.com"
    smtp_port: int = 465
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    smtp_use_ssl: bool = True
    smtp_starttls: bool = False
    email_debug_mode: bool = False
    code_ttl_seconds: int = 300
    code_cooldown_seconds: int = 60
    max_code_attempts: int = 5
    session_ttl_seconds: int = 604800
    session_cookie_name: str = "agent_session"
    session_cookie_secure: bool = False

    @classmethod
    def from_env(cls) -> "AuthSettings":
        load_dotenv(PROJECT_ROOT / ".env")
        configured_path = Path(os.getenv("AUTH_DB_PATH", "data/auth.db"))
        db_path = (
            configured_path
            if configured_path.is_absolute()
            else PROJECT_ROOT / configured_path
        )
        return cls(
            db_path=db_path.resolve(),
            smtp_host=os.getenv("SMTP_HOST", "smtp.qq.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "465")),
            smtp_username=os.getenv("SMTP_USERNAME") or None,
            smtp_password=os.getenv("SMTP_PASSWORD") or None,
            smtp_from=os.getenv("SMTP_FROM") or None,
            smtp_use_ssl=_env_bool("SMTP_USE_SSL", True),
            smtp_starttls=_env_bool("SMTP_STARTTLS", False),
            email_debug_mode=_env_bool("EMAIL_DEBUG_MODE", False),
            code_ttl_seconds=int(os.getenv("EMAIL_CODE_TTL_SECONDS", "300")),
            code_cooldown_seconds=int(
                os.getenv("EMAIL_CODE_COOLDOWN_SECONDS", "60")
            ),
            max_code_attempts=int(os.getenv("EMAIL_CODE_MAX_ATTEMPTS", "5")),
            session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "604800")),
            session_cookie_name=os.getenv("SESSION_COOKIE_NAME", "agent_session"),
            session_cookie_secure=_env_bool("SESSION_COOKIE_SECURE", False),
        )


class VerificationEmailSender(Protocol):
    def send_verification_code(self, email: str, code: str) -> None:
        """Send one verification code."""


class SMTPVerificationEmailSender:
    """Send verification messages through SMTP or local debug logging."""

    def __init__(self, settings: AuthSettings) -> None:
        self.settings = settings

    @staticmethod
    def _masked_email(email: str) -> str:
        local, domain = email.split("@", 1)
        visible = local[:2]
        return f"{visible}{'*' * max(1, len(local) - len(visible))}@{domain}"

    def send_verification_code(self, email: str, code: str) -> None:
        if self.settings.email_debug_mode:
            LOGGER.warning(
                "[Auth] EMAIL_DEBUG_MODE verification_code email=%s code=%s",
                self._masked_email(email),
                code,
            )
            return

        username = self.settings.smtp_username
        password = self.settings.smtp_password
        sender = self.settings.smtp_from or username
        if not username or not password or not sender:
            raise EmailConfigurationError(
                "SMTP 配置不完整，请设置 SMTP_USERNAME、SMTP_PASSWORD 和 SMTP_FROM"
            )

        message = EmailMessage()
        message["Subject"] = "Agent 登录验证码"
        message["From"] = sender
        message["To"] = email
        message.set_content(
            f"你的验证码是：{code}\n\n验证码 5 分钟内有效，请勿转发给他人。"
        )

        if self.settings.smtp_use_ssl:
            smtp: smtplib.SMTP = smtplib.SMTP_SSL(
                self.settings.smtp_host,
                self.settings.smtp_port,
                timeout=15,
            )
        else:
            smtp = smtplib.SMTP(
                self.settings.smtp_host,
                self.settings.smtp_port,
                timeout=15,
            )
        with smtp:
            smtp.ehlo()
            if self.settings.smtp_starttls and not self.settings.smtp_use_ssl:
                smtp.starttls()
                smtp.ehlo()
            smtp.login(username, password)
            smtp.send_message(message)


@dataclass(frozen=True)
class AuthenticationResult:
    user: dict[str, object]
    session_token: str


class AuthService:
    """Manage email codes, users, and opaque cookie sessions in SQLite."""

    def __init__(
        self,
        settings: AuthSettings | None = None,
        *,
        email_sender: VerificationEmailSender | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings or AuthSettings.from_env()
        self.email_sender = email_sender or SMTPVerificationEmailSender(self.settings)
        self._clock = clock
        self._write_lock = Lock()
        self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.settings.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Commit or roll back a short-lived connection, then always close it."""
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    last_login_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS verification_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    code_salt TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    consumed_at REAL,
                    attempts INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_codes_email_created
                ON verification_codes(email, created_at DESC);

                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user
                ON sessions(user_id);
                """
            )

    @staticmethod
    def normalize_email(email: str) -> str:
        if not isinstance(email, str):
            raise EmailValidationError("邮箱必须是字符串")
        normalized = email.strip().casefold()
        if not QQ_EMAIL_PATTERN.fullmatch(normalized):
            raise EmailValidationError("请输入有效的 QQ 邮箱，例如 123456@qq.com")
        return normalized

    @staticmethod
    def _hash_code(code: str, salt: str) -> str:
        return hashlib.sha256(f"{salt}:{code}".encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_session(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _public_user(row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": row["id"],
            "email": row["email"],
            "created_at": row["created_at"],
            "last_login_at": row["last_login_at"],
        }

    def request_verification_code(self, email: str) -> dict[str, object]:
        normalized = self.normalize_email(email)
        now = self._clock()
        with self._write_lock:
            with self._connection() as connection:
                latest = connection.execute(
                    "SELECT created_at FROM verification_codes "
                    "WHERE email = ? ORDER BY created_at DESC LIMIT 1",
                    (normalized,),
                ).fetchone()
                if latest is not None:
                    elapsed = now - float(latest["created_at"])
                    if elapsed < self.settings.code_cooldown_seconds:
                        retry_after = max(
                            1,
                            int(self.settings.code_cooldown_seconds - elapsed + 0.999),
                        )
                        raise VerificationCooldownError(retry_after)

                code = f"{secrets.randbelow(1_000_000):06d}"
                salt = secrets.token_hex(16)
                try:
                    self.email_sender.send_verification_code(normalized, code)
                except AuthError:
                    raise
                except Exception as exc:
                    raise EmailDeliveryError("验证码邮件发送失败，请稍后重试") from exc
                connection.execute(
                    "INSERT INTO verification_codes "
                    "(email, code_hash, code_salt, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        normalized,
                        self._hash_code(code, salt),
                        salt,
                        now,
                        now + self.settings.code_ttl_seconds,
                    ),
                )
        LOGGER.info("[Auth] verification_code_sent email=%s", normalized)
        return {
            "email": normalized,
            "expires_in_seconds": self.settings.code_ttl_seconds,
            "cooldown_seconds": self.settings.code_cooldown_seconds,
        }

    def verify_and_login(self, email: str, code: str) -> AuthenticationResult:
        normalized = self.normalize_email(email)
        if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
            raise InvalidVerificationCodeError("验证码必须是 6 位数字")

        now = self._clock()
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                record = connection.execute(
                    "SELECT * FROM verification_codes WHERE email = ? "
                    "ORDER BY created_at DESC, id DESC LIMIT 1",
                    (normalized,),
                ).fetchone()
                if record is None or record["consumed_at"] is not None:
                    connection.rollback()
                    raise InvalidVerificationCodeError("验证码无效或已使用")
                if now >= float(record["expires_at"]):
                    connection.execute(
                        "UPDATE verification_codes SET consumed_at = ? WHERE id = ?",
                        (now, record["id"]),
                    )
                    connection.commit()
                    raise ExpiredVerificationCodeError("验证码已过期")

                actual_hash = self._hash_code(code, record["code_salt"])
                if not hmac.compare_digest(actual_hash, record["code_hash"]):
                    attempts = int(record["attempts"]) + 1
                    consumed_at = (
                        now if attempts >= self.settings.max_code_attempts else None
                    )
                    connection.execute(
                        "UPDATE verification_codes SET attempts = ?, consumed_at = ? "
                        "WHERE id = ?",
                        (attempts, consumed_at, record["id"]),
                    )
                    connection.commit()
                    raise InvalidVerificationCodeError("验证码错误")

                connection.execute(
                    "UPDATE verification_codes SET consumed_at = ? WHERE id = ?",
                    (now, record["id"]),
                )
                user = connection.execute(
                    "SELECT * FROM users WHERE email = ?",
                    (normalized,),
                ).fetchone()
                if user is None:
                    cursor = connection.execute(
                        "INSERT INTO users (email, created_at, last_login_at) "
                        "VALUES (?, ?, ?)",
                        (normalized, now, now),
                    )
                    user_id = int(cursor.lastrowid)
                else:
                    user_id = int(user["id"])
                    connection.execute(
                        "UPDATE users SET last_login_at = ? WHERE id = ?",
                        (now, user_id),
                    )

                session_token = secrets.token_urlsafe(32)
                connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
                connection.execute(
                    "INSERT INTO sessions "
                    "(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                    (
                        self._hash_session(session_token),
                        user_id,
                        now,
                        now + self.settings.session_ttl_seconds,
                    ),
                )
                user = connection.execute(
                    "SELECT * FROM users WHERE id = ?",
                    (user_id,),
                ).fetchone()
                connection.commit()
            except AuthError:
                raise
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

        LOGGER.info("[Auth] login_success user_id=%s email=%s", user_id, normalized)
        return AuthenticationResult(
            user=self._public_user(user),
            session_token=session_token,
        )

    def get_user_by_session(self, session_token: str | None) -> dict[str, object]:
        if not session_token:
            raise InvalidSessionError("未登录")
        now = self._clock()
        token_hash = self._hash_session(session_token)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT users.*, sessions.expires_at AS session_expires_at "
                "FROM sessions JOIN users ON users.id = sessions.user_id "
                "WHERE sessions.token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                raise InvalidSessionError("Session 无效")
            if now >= float(row["session_expires_at"]):
                connection.execute(
                    "DELETE FROM sessions WHERE token_hash = ?",
                    (token_hash,),
                )
                raise InvalidSessionError("Session 已过期")
        return self._public_user(row)

    def logout(self, session_token: str | None) -> None:
        if not session_token:
            return
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?",
                (self._hash_session(session_token),),
            )
        LOGGER.info("[Auth] logout_success")


__all__ = [
    "AuthError",
    "AuthService",
    "AuthSettings",
    "AuthenticationResult",
    "EmailConfigurationError",
    "EmailDeliveryError",
    "EmailValidationError",
    "ExpiredVerificationCodeError",
    "InvalidSessionError",
    "InvalidVerificationCodeError",
    "SMTPVerificationEmailSender",
    "VerificationCooldownError",
]
