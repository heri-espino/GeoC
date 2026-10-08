"""Offline tests for the Checkpoint 07 acquisition runner."""
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "run_checkpoint_07_acquisition.py"
spec = importlib.util.spec_from_file_location("checkpoint07_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_simulate_without_network_or_keys(self):
        self.assertEqual(runner.main(["simulate", "--root", "/tmp/GeoC"]), 0)

    def test_init_template_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env.checkpoint07"
            self.assertEqual(runner.main(["init", "--root", folder]), 0)
            self.assertIn("REPLACE_WITH", path.read_text(encoding="utf-8"))
            path.write_text("CUSTOM=preserve\n", encoding="utf-8")
            self.assertEqual(runner.main(["init", "--root", folder]), 0)
            self.assertEqual(path.read_text(encoding="utf-8"), "CUSTOM=preserve\n")

    def test_placeholder_rejected(self):
        template = Path(__file__).resolve().parents[1] / ".env.checkpoint07.example"
        conf = runner.parse_env(template)
        for source in ("sentinel1", "hls", "smap", "agera5"):
            self.assertFalse(runner.requirements(conf, source))
        self.assertTrue(runner.requirements(conf, "prithvi"))

    def test_local_env_parser_never_interprets_shell(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "credentials.env"
            path.write_text(
                "CDSAPI_KEY='123-abc'\nFOO=$(touch /tmp/should_not_exist_123abc)\n",
                encoding="utf-8",
            )
            conf = runner.parse_env(path)
            self.assertEqual(conf["CDSAPI_KEY"], "123-abc")
            self.assertTrue(conf["FOO"].startswith("$(touch "))

    def test_requires_opt_in_for_live_network(self):
        self.assertEqual(runner.main(["check-auth", "--root", "/tmp/GeoC"]), 2)
        self.assertEqual(runner.main(["download", "--root", "/tmp/GeoC"]), 2)

    def test_redacts_secrets(self):
        secret = "TOKEN_ABC123456789"
        with io.StringIO() as output, redirect_stdout(output):
            runner.safe_print("Error: " + secret, {"CDSAPI_KEY": secret})
            text = output.getvalue()
        self.assertNotIn(secret, text)
        self.assertIn("[REDACTED]", text)


class IntegrationTests(unittest.TestCase):
    def test_download_with_stub_never_needs_real_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            downloader = root / "tools" / "download_checkpoint_07_data.py"
            downloader.parent.mkdir(parents=True)
            downloader.write_text(
                """
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--sources', nargs='+')
p.add_argument('--start')
p.add_argument('--end')
p.add_argument('--margin-deg')
p.add_argument('--s1-resolution-m')
p.add_argument('--parcels')
p.add_argument('--catalog-only', action='store_true')
args = p.parse_args()

out = Path('data/raw/checkpoint_07')
out.mkdir(parents=True, exist_ok=True)
(out / 'download_manifest.json').write_text(
    json.dumps({'sources': args.sources}), encoding='utf-8'
)
print('STUB: no remote downloads')
""",
                encoding="utf-8",
            )
            rc = runner.main(
                ["download", "--root", folder, "--sources", "prithvi", "--execute"]
            )
            self.assertEqual(rc, 0)
            manifest = (
                root / "data/raw/checkpoint_07"
                / "acquisition_manifests/prithvi_download.json"
            )
            self.assertTrue(manifest.exists())
            self.assertEqual(json.loads(manifest.read_text())["sources"], ["prithvi"])


if __name__ == "__main__":
    unittest.main()
