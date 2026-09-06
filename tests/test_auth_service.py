"""Unit tests for QQ verification codes and opaque sessions."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.auth.service import (
    AuthService,
    AuthSettings,
    EmailValidationError,
    ExpiredVerificationCodeError,
    InvalidSessionError,
    InvalidVerificationCodeError,
    VerificationCooldownError,
)


class _Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _EmailSender:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def send_verification_code(self, email: str, code: str) -> None:
        self.messages.append((email, code))

    @property
    def latest_code(self) -> str:
        return self.messages[-1][1]


class AuthServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        self.temp_dir = tempfile.TemporaryDirectory(prefix="tmp_auth_", dir=tests_dir)
        self.clock = _Clock()
        self.sender = _EmailSender()
        settings = AuthSettings(
            db_path=Path(self.temp_dir.name) / "auth.db",
            email_debug_mode=False,
            code_ttl_seconds=300,
            code_cooldown_seconds=60,
            max_code_attempts=5,
            session_ttl_seconds=600,
        )
        self.service = AuthService(
            settings,
            email_sender=self.sender,
            clock=self.clock,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _request_and_verify(self, email: str = "123456@qq.com"):
        self.service.request_verification_code(email)
        return self.service.verify_and_login(email, self.sender.latest_code)

    def test_new_user_is_created_and_session_is_valid(self) -> None:
        result = self._request_and_verify()

        self.assertEqual(len(self.sender.latest_code), 6)
        self.assertTrue(self.sender.latest_code.isdigit())
        self.assertEqual(result.user["email"], "123456@qq.com")
        self.assertEqual(
            self.service.get_user_by_session(result.session_token)["id"],
            result.user["id"],
        )

    def test_existing_user_logs_in_without_duplicate_account(self) -> None:
        first = self._request_and_verify()
        self.clock.advance(61)
        second = self._request_and_verify()

        self.assertEqual(first.user["id"], second.user["id"])
        self.assertNotEqual(first.session_token, second.session_token)

    def test_wrong_code_is_rejected_then_correct_code_still_works(self) -> None:
        self.service.request_verification_code("123456@qq.com")
        wrong_code = "000000" if self.sender.latest_code != "000000" else "000001"

        with self.assertRaises(InvalidVerificationCodeError):
            self.service.verify_and_login("123456@qq.com", wrong_code)
        result = self.service.verify_and_login(
            "123456@qq.com",
            self.sender.latest_code,
        )

        self.assertEqual(result.user["email"], "123456@qq.com")

    def test_expired_code_is_rejected(self) -> None:
        self.service.request_verification_code("123456@qq.com")
        self.clock.advance(301)

        with self.assertRaises(ExpiredVerificationCodeError):
            self.service.verify_and_login("123456@qq.com", self.sender.latest_code)

    def test_code_cannot_be_reused(self) -> None:
        self.service.request_verification_code("123456@qq.com")
        code = self.sender.latest_code
        self.service.verify_and_login("123456@qq.com", code)

        with self.assertRaises(InvalidVerificationCodeError):
            self.service.verify_and_login("123456@qq.com", code)

    def test_request_cooldown_is_enforced(self) -> None:
        self.service.request_verification_code("123456@qq.com")

        with self.assertRaises(VerificationCooldownError) as caught:
            self.service.request_verification_code("123456@qq.com")

        self.assertEqual(caught.exception.retry_after_seconds, 60)

    def test_invalid_qq_email_is_rejected(self) -> None:
        for email in ("not-an-email", "123456@gmail.com", "1234@qq.com"):
            with self.subTest(email=email):
                with self.assertRaises(EmailValidationError):
                    self.service.request_verification_code(email)

    def test_logout_invalidates_session(self) -> None:
        result = self._request_and_verify()
        self.service.logout(result.session_token)

        with self.assertRaises(InvalidSessionError):
            self.service.get_user_by_session(result.session_token)


if __name__ == "__main__":
    unittest.main()
