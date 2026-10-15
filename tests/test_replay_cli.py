"""Replay CLI regression tests without credentials or provider traffic."""
import os
# Exercise the CLI using a trusted Python executable.
import subprocess  # nosec B404
import sys
import tempfile
import unittest
from pathlib import Path

from agent import secret_scope

ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = Path(secret_scope.__file__).resolve().parents[1]


class ReplayCliTests(unittest.TestCase):
    def run_replay(self, *args):
        with tempfile.TemporaryDirectory() as directory:
            env = {
                "PATH": os.defpath,
                "HOME": directory,
                "HERMES_HOME": directory,
                "PYTHONPATH": str(HERMES_ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            # Fixed local script and test-only arguments; no shell.
            return subprocess.run(  # nosec B603
                [sys.executable, str(ROOT / "tests" / "eval" / "replay.py"), *args],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

    def test_help_runs_outside_plugin_directory(self):
        result = self.run_replay("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cases", result.stdout)
        self.assertIn("--family", result.stdout)

    def test_default_corpus_loads_outside_plugin_directory_without_credentials(self):
        result = self.run_replay("--family", "missing-regression-family")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stderr.strip(), "no family named 'missing-regression-family'")
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
