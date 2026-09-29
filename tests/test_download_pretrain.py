import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from tools.download_pretrain import is_verified, local_path, transfer


PAYLOAD = bytes(range(256)) * 257


class Handler(BaseHTTPRequestHandler):
    requests = []

    def do_GET(self):
        self.requests.append(self.headers.get("Range"))
        start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        use_range = "Range" in self.headers and not self.path.startswith("/ignore")
        if use_range:
            self.send_response(206)
            reported = start + 1 if self.path.startswith("/bad-range") else start
            self.send_header("Content-Range", f"bytes {reported}-{len(PAYLOAD)-1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
            start = 0
        self.send_header("Content-Length", str(len(PAYLOAD) - start))
        self.end_headers()
        self.wfile.write(PAYLOAD[start:])

    def log_message(self, *args):
        pass


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entry = {
            "source": "test", "repo_id": "test/data", "revision": "a" * 40,
            "remote_path": "test.parquet", "local_path": "raw/test.parquet",
            "url": f"http://127.0.0.1:{self.server.server_port}/file",
            "bytes": len(PAYLOAD), "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        }
        self.path = local_path(self.root, self.entry)
        self.path.parent.mkdir(parents=True)
        self.part = self.path.with_name(self.path.name + ".part")
        Handler.requests.clear()

    def download(self):
        transfer(self.entry, self.root, lambda *args: None, attempts=1)

    def test_resume_checks_hash_and_second_run_skips_network(self):
        self.part.write_bytes(PAYLOAD[:2048])
        self.download()
        self.assertEqual(Handler.requests, ["bytes=2048-"])
        self.assertEqual(self.path.read_bytes(), PAYLOAD)
        self.assertTrue(is_verified(self.path, self.entry))
        self.download()
        self.assertEqual(len(Handler.requests), 1)

    def test_server_ignoring_range_restarts_without_duplicate_bytes(self):
        self.entry["url"] = self.entry["url"].replace("/file", "/ignore")
        self.part.write_bytes(PAYLOAD[:2048])
        self.download()
        self.assertEqual(self.path.read_bytes(), PAYLOAD)

    def test_wrong_range_is_not_appended(self):
        self.entry["url"] = self.entry["url"].replace("/file", "/bad-range")
        self.part.write_bytes(PAYLOAD[:2048])
        with self.assertRaises(RuntimeError):
            self.download()
        self.assertEqual(self.part.read_bytes(), PAYLOAD[:2048])
        self.assertFalse(self.path.exists())

    def test_corrupt_completed_partial_is_quarantined(self):
        self.part.write_bytes(b"x" * len(PAYLOAD))
        with self.assertRaises(RuntimeError):
            self.download()
        self.assertFalse(self.path.exists())
        self.assertEqual(len(list(self.path.parent.glob("*.corrupt-*"))), 1)

    def test_changed_final_file_is_not_trusted(self):
        self.download()
        self.path.write_bytes(b"wrong")
        self.assertFalse(is_verified(self.path, self.entry))
        with self.assertRaises(RuntimeError):
            self.download()

    def test_destination_cannot_escape_root(self):
        self.entry["local_path"] = "../escape.parquet"
        with self.assertRaises(ValueError):
            local_path(self.root, self.entry)


if __name__ == "__main__":
    unittest.main()
