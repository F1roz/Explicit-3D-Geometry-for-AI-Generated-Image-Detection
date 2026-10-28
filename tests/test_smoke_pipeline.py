"""Slow end-to-end test (about two minutes). Enable with PRISM3D_SLOW=1."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("PRISM3D_SLOW") == "1", "set PRISM3D_SLOW=1 to run the end-to-end smoke test")
class TestSmoke(unittest.TestCase):
    def test_run_smoke(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_smoke.py")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("SMOKE TEST PASSED", r.stdout)


if __name__ == "__main__":
    unittest.main()
