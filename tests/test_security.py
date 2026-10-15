"""Hermetic regression tests: no real credentials, HTTP calls, or user config."""
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import provider  # noqa: E402
from agent import secret_scope  # noqa: E402
from tests.eval import replay  # noqa: E402


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.multiplex_before = secret_scope.is_multiplex_active()
        self.scope = secret_scope.set_secret_scope(None)
        secret_scope.set_multiplex_active(True)

    def tearDown(self):
        secret_scope.reset_secret_scope(self.scope)
        secret_scope.set_multiplex_active(self.multiplex_before)

    def scoped(self, values):
        return secret_scope.set_secret_scope(values)

    def test_missing_scope_does_not_read_process_credentials(self):
        with patch.dict(os.environ, {
            "AZURE_FOUNDRY_API_KEY": "WRONG_PROFILE_KEY",
            "AZURE_FOUNDRY_BASE_URL": "https://other.openai.azure.com",
        }), patch.object(provider._opener, "open") as send:
            self.assertFalse(provider.check_requirements())
            self.assertEqual(provider._call("PRIVATE", {}), {"error": "credential scope unavailable"})
            send.assert_not_called()

    def test_scoped_miss_never_falls_back_to_environment(self):
        with patch.dict(os.environ, {"AZURE_FOUNDRY_API_KEY": "WRONG_PROFILE_KEY"}), patch.object(provider._opener, "open") as send:
            token = self.scoped({"AZURE_FOUNDRY_BASE_URL": "https://right.openai.azure.com"})
            try:
                self.assertFalse(provider.check_requirements())
                self.assertEqual(provider._call("PRIVATE", {}), {"error": "missing AZURE_FOUNDRY_API_KEY"})
                send.assert_not_called()
            finally:
                secret_scope.reset_secret_scope(token)

    def test_absent_endpoint_returns_requested_error_without_transport(self):
        token = self.scoped({"AZURE_FOUNDRY_API_KEY": "SENTINEL"})
        try:
            with patch.object(provider._opener, "open") as send:
                self.assertEqual(provider._call("PRIVATE", {}), {"error": "set AZURE_FOUNDRY_BASE_URL"})
                send.assert_not_called()
        finally:
            secret_scope.reset_secret_scope(token)

    def test_endpoint_allowlist_and_suffix(self):
        good = ["https://right.openai.azure.com", "https://right.openai.azure.com/openai/v1/", "https://right.openai.azure.com/openai"]
        bad = ["http://right.openai.azure.com", "https://right.openai.azure.com.evil.com",
               "https://evil.com", "https://evil.com@right.openai.azure.com",
               "https://right.openai.azure.com:444", "https://right.openai.azure.com/x",
               "https://right.openai.azure.com/?key=x", "https://right.openai.azure.com/#x"]
        for url in good + bad:
            with self.subTest(url=url):
                token = self.scoped({"AZURE_FOUNDRY_BASE_URL": url, "AZURE_FOUNDRY_API_KEY": "SENTINEL"})
                try:
                    self.assertEqual(provider._base_url(), "https://right.openai.azure.com" if url in good else "")
                    if url in bad:
                        with patch.object(provider._opener, "open") as send:
                            self.assertEqual(provider._call("PRIVATE", {}), {"error": "set AZURE_FOUNDRY_BASE_URL"})
                            send.assert_not_called()
                finally:
                    secret_scope.reset_secret_scope(token)

    def test_redirect_not_followed_and_body_not_echoed(self):
        token = self.scoped({"AZURE_FOUNDRY_BASE_URL": "https://right.openai.azure.com", "AZURE_FOUNDRY_API_KEY": "SENTINEL"})
        try:
            req = __import__("urllib.request", fromlist=["Request"]).Request("https://right.openai.azure.com/path", data=b"PRIVATE", headers={"api-key": "SENTINEL"})
            handler = provider._NoRedirect()
            self.assertIsNone(handler.redirect_request(req, None, 302, "redirect", {}, "https://evil.com/"))
            http_error = urllib.error.HTTPError(req.full_url, 302, "PRIVATE SENTINEL", {"Location": "https://evil.com/"}, None)
            with patch.object(provider._opener, "open", side_effect=http_error):
                self.assertEqual(provider._call("PRIVATE", {}), {"error": "HTTP 302"})
        finally:
            secret_scope.reset_secret_scope(token)

    def test_redirect_credential_stays_on_original_host(self):
        seen = []
        class Origin(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(("origin", self.headers.get("Api-key")))
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/leak")
                self.end_headers()
            def log_message(self, format, *args): pass
        class Target(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(("target", self.headers.get("Api-key")))
                self.send_response(200)
                self.end_headers()
            def log_message(self, format, *args): pass
        origin = HTTPServer(("127.0.0.1", 0), Origin)
        target = HTTPServer(("127.0.0.1", 0), Target)
        workers = [threading.Thread(target=s.serve_forever, daemon=True) for s in (origin, target)]
        for worker in workers:
            worker.start()
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{origin.server_port}/", data=b"PRIVATE", headers={"api-key": "SENTINEL"})
            with self.assertRaises(urllib.error.HTTPError) as context:
                provider._opener.open(request, timeout=2)
            context.exception.close()
            self.assertEqual(context.exception.code, 302)
            self.assertEqual(seen, [("origin", "SENTINEL")])
        finally:
            for server in (origin, target):
                server.shutdown()
                server.server_close()
            for worker in workers:
                worker.join(timeout=2)

    def test_gate_projects_no_args_and_requires_approval(self):
        private = "someone@example.invalid /Users/private/password.txt sk-EXAMPLE"
        with patch.object(provider, "_GATE_ENABLED", True), patch.object(provider, "_GATE_MODE", "approve"), patch.object(provider, "_call", return_value={"answers": {"safe": {"noul": 1.0}}}) as call:
            first = provider._gate("terminal", {"command": f"rm -rf '{private}'"})
            second = provider._gate("terminal", {"command": "rm -rf another-dir"})
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertEqual(first["action"], "approve")
            self.assertNotEqual(first["rule_key"], second["rule_key"])
            self.assertNotIn(private, str(call.call_args_list))
            self.assertIn("another-dir", second["message"])
            self.assertIn("rm -rf", first["message"])
            self.assertIsNone(provider._gate("decision_noul", {"state": private}))
            self.assertIsNone(provider._gate("read_file", {"path": private}))
            mcp = provider._gate("mcp__other_send_message", {"to": private})
            self.assertIsNotNone(mcp)
            self.assertEqual(mcp["action"], "approve")

    def test_gate_provider_error_malformed_and_low_probability(self):
        with patch.object(provider, "_GATE_ENABLED", True), patch.object(provider, "_GATE_MODE", "approve"):
            for response in [{"error": "offline"}, {"answers": {"safe": {"noul": "nan"}}}, {"answers": {"safe": {"noul": 0.01}}}]:
                with self.subTest(response=response), patch.object(provider, "_call", return_value=response):
                    result = provider._gate("write_file", {"path": "scratch/a"})
                    self.assertIsNotNone(result)
                    self.assertEqual(result["action"], "approve")
        with patch.object(provider, "_GATE_ENABLED", True), patch.object(provider, "_GATE_MODE", "block"), patch.object(provider, "_call", return_value={"error": "offline"}):
            result = provider._gate("write_file", {})
            self.assertIsNotNone(result)
            self.assertEqual(result["action"], "block")

        for raw in ("not a map", None, {}, {"answers": {"safe": {"noul": float("nan")}}}):
            with self.subTest(raw=raw), patch.object(provider, "_GATE_ENABLED", True), patch.object(provider, "_GATE_MODE", "approve"), patch.object(provider, "_call", return_value=raw):
                result = provider._gate("write_file", {})
                self.assertIsNotNone(result)
                self.assertEqual(result["action"], "approve")

    def test_projection_is_coarse_and_does_not_prove_safety(self):
        states = [
            provider._project_operation("terminal", {"command": "rm -rf ~/photos"}),
            provider._project_operation("terminal", {"command": "rm -rf scratch"}),
        ]
        self.assertEqual(states[0], states[1])
        self.assertEqual(provider._project_operation("mcp__send_private@example.invalid", {}),
                         "Tool call: mcp_tool; operation: unspecified")

    def test_replay_rejects_empty_family_and_invalid_threshold_before_egress(self):
        family = {"question_type": "noul", "cases": []}
        with patch.object(replay, "_call") as send:
            with self.assertRaisesRegex(ValueError, "at least one case"):
                replay.replay_family(family, 0.7, 0)
            family["cases"] = [{"id": "x", "state": "fixture", "label": True}]
            for coverage in (-1, 0.2, 1.5, float("nan"), float("inf")):
                with self.assertRaisesRegex(ValueError, "coverage"):
                    replay.replay_family(family, coverage, 0)
            send.assert_not_called()

    def test_gate_hook_receives_host_payload_without_duplicate_kwargs(self):
        hooks = {}
        class Context:
            def register_hook(self, event, callback): hooks[event] = callback
        provider.register_gate(Context())
        with patch.object(provider, "_GATE_ENABLED", True), patch.object(provider, "_call", return_value={"error": "offline"}):
            result = hooks["pre_tool_call"](tool_name="write_file", args={"path": "scratch/a"}, task_id="synthetic")
            self.assertEqual(result["action"], "approve")

    def test_request_uses_scoped_key_once_and_host_root(self):
        token = self.scoped({"AZURE_FOUNDRY_BASE_URL": "https://right.openai.azure.com/openai/v1", "AZURE_FOUNDRY_API_KEY": "SENTINEL"})
        try:
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def read(self): return b'{"answers": {}}'
            with patch.object(provider._opener, "open", return_value=Response()) as send:
                self.assertEqual(provider._call("PRIVATE", {}), {"answers": {}})
                req = send.call_args.args[0]
                self.assertEqual(req.full_url, "https://right.openai.azure.com/providers/microsoft/v1/systemone")
                self.assertEqual(req.get_header("Api-key"), "SENTINEL")
                self.assertIsNone(req.get_header("Authorization"))
                self.assertEqual(json.loads(req.data)["state"], "PRIVATE")
        finally:
            secret_scope.reset_secret_scope(token)


if __name__ == "__main__":
    unittest.main()
