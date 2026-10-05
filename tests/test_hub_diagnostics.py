"""Hub failures must keep redacted, classifiable evidence instead of a generic message."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from orpheus_utils import HubOperationError, describe_hub_error, hub_token_report, redact
from pipeline_recovery import DeferredUploads
from test_pipeline import FakeHub, fake_checkpoint

SECRET = "hf_abcdefghijklmnopqrstuvwxyz123456"


class HttpError(Exception):
    def __init__(self, status, message="Client Error", server_message=None, headers=None):
        super().__init__(message)
        self.server_message = server_message
        self.response = SimpleNamespace(status_code=status, headers=headers or {"x-request-id": "Root=1-abc"},
                                        json=lambda: {})


class ConnectionError(Exception):  # name matches requests/httpx hierarchy
    pass


class RedactionTests(unittest.TestCase):
    def test_tokens_and_auth_headers_are_removed(self):
        text = redact(f"Authorization: Bearer {SECRET} url=https://u:pw@proxy:8080 token={SECRET}", SECRET)
        self.assertNotIn(SECRET, text)
        self.assertNotIn("pw@", text)
        self.assertNotIn("hf_abc", redact("leaked hf_abcdefghij in text"))


class ClassificationTests(unittest.TestCase):
    def store(self):
        store = FakeHub(private=True).store()
        store._token = SECRET
        return store

    def test_permission_error_keeps_server_message_and_request_id_without_retry(self):
        store = self.store()
        call = Mock(side_effect=HttpError(403, f"403 Forbidden for {SECRET}",
                                          server_message="You don't have the rights to create a commit"))
        with patch("orpheus_utils.time.sleep") as sleep, self.assertRaises(HubOperationError) as caught:
            store._call("upload", call)
        sleep.assert_not_called()
        message = str(caught.exception)
        self.assertIn("[permission] HTTP 403", message)
        self.assertIn("rights to create a commit", message)
        self.assertIn("Root=1-abc", message)
        self.assertNotIn(SECRET, message + json.dumps(caught.exception.diagnostics))
        self.assertIsInstance(caught.exception, RuntimeError)

    def test_categories(self):
        cases = {401: "authentication", 404: "not_found", 413: "payload_too_large", 429: "rate_limit",
                 412: "conflict", 502: "server", 422: "api_request"}
        for status, category in cases.items():
            self.assertEqual(describe_hub_error(HttpError(status), "x")["category"], category)
        quota = HttpError(403, server_message="You have exceeded your storage quota")
        self.assertEqual(describe_hub_error(quota, "x")["category"], "quota")
        self.assertEqual(describe_hub_error(ConnectionError("reset"), "x")["category"], "connectivity")
        wrapped = RuntimeError("outer")
        wrapped.__cause__ = ConnectionError("dns")
        self.assertEqual(describe_hub_error(wrapped, "x")["category"], "connectivity")

    def test_rate_limit_honours_retry_after(self):
        store = self.store()
        limited = HttpError(429, headers={"retry-after": "17"})
        call = Mock(side_effect=[limited, "ok"])
        with patch("orpheus_utils.time.sleep") as sleep:
            self.assertEqual(store._call("upload", call), "ok")
        sleep.assert_called_once_with(17.0)

    def test_network_errors_retry_then_report(self):
        store = self.store()
        call = Mock(side_effect=ConnectionError("Name resolution failed"))
        with patch("orpheus_utils.time.sleep") as sleep, self.assertRaisesRegex(HubOperationError, "connectivity"):
            store._call("upload", call)
        self.assertEqual(sleep.call_count, 2)


class TokenReportTests(unittest.TestCase):
    def test_roles(self):
        def who(role, scoped=None):
            access = {"role": role, "displayName": "kaggle"}
            if scoped is not None:
                access["fineGrained"] = {"scoped": scoped}
            return {"name": "student", "orgs": [], "auth": {"accessToken": access}}
        repo = "student/ckpt"
        self.assertIs(hub_token_report(who("write"), repo)["can_write"], True)
        self.assertIs(hub_token_report(who("read"), repo)["can_write"], False)
        granted = [{"entity": {"type": "user", "name": "student"}, "permissions": ["repo.content.read", "repo.write"]}]
        self.assertIs(hub_token_report(who("fineGrained", granted), repo)["can_write"], True)
        read_only = [{"entity": {"type": "model", "name": repo}, "permissions": ["repo.content.read"]}]
        self.assertIs(hub_token_report(who("fineGrained", read_only), repo)["can_write"], False)
        self.assertIsNone(hub_token_report({}, repo)["can_write"])

    def test_preflight_reports_scope_in_upload_failure(self):
        hub = FakeHub(private=True)
        hub.whoami = lambda: {"name": "student", "auth": {"accessToken": {"role": "read"}}}
        hub.upload_file = Mock(side_effect=HttpError(403, server_message="Forbidden"))
        with self.assertRaises(HubOperationError) as caught:
            hub.store().preflight()
        self.assertIn('"token_role": "read"', str(caught.exception))
        self.assertEqual(caught.exception.diagnostics["operation"], "upload")

    def test_whoami_failure_other_than_auth_is_not_fatal(self):
        hub = FakeHub(private=True)
        hub.whoami = Mock(side_effect=HttpError(429))
        with patch("orpheus_utils.time.sleep"):
            hub.store().preflight()
        self.assertIn("runs/experiment/connection_probe.json", hub.files)
        hub.whoami = Mock(side_effect=HttpError(401, server_message="Invalid user token."))
        with self.assertRaisesRegex(HubOperationError, "authentication"):
            hub.store().preflight()


class DeferredStatusTests(unittest.TestCase):
    def test_failed_backup_records_reason_and_stays_local(self):
        hub = FakeHub(private=True)
        hub.commit_failure = HttpError(507, server_message="Insufficient storage")
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            cp = fake_checkpoint(run / "checkpoint-1")
            queue = DeferredUploads(hub.store(), run)
            with patch("orpheus_utils.time.sleep"):
                self.assertFalse(queue.backup(run, cp))
            status = json.loads((run / "backup_status.json").read_text())
            self.assertFalse(status["remote_snapshot_current"])
            self.assertEqual(status["last_upload_error"]["category"], "quota")
            self.assertIn("Insufficient storage", status["last_upload_error"]["message"])


@unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "huggingface_hub not installed")
class RealClientExceptionTests(unittest.TestCase):
    def test_real_hf_http_error_is_classified(self):
        import requests
        from huggingface_hub.utils import HfHubHTTPError
        response = requests.Response()
        response.status_code = 403
        response.headers["x-request-id"] = "Root=1-real"
        response._content = b'{"error": "You don\'t have the rights to create a commit"}'
        error = HfHubHTTPError(f"403 Forbidden token={SECRET}", response=response)
        info = describe_hub_error(error, "upload", SECRET)
        self.assertEqual(info["category"], "permission")
        self.assertEqual(info["request_id"], "Root=1-real")
        self.assertIn("rights", info["server_message"])
        self.assertNotIn(SECRET, json.dumps(info))


if __name__ == "__main__":
    unittest.main()
