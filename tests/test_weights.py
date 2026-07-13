import http.server
import json
import os
import re
import threading
from pathlib import Path

import pytest

import weights


class RangeHandler(http.server.BaseHTTPRequestHandler):
    root = "."
    ranges = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = Path(self.root) / self.path.lstrip("/")
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        requested = self.headers.get("Range")
        if requested:
            self.ranges.append(requested)
            start = int(re.match(r"bytes=(\d+)-", requested).group(1))
            data = data[start:]
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def mirror(tmp_path):
    repo = weights.BASE_REPO
    source = tmp_path / "server" / repo
    files = {
        "config.json": b"{}",
        "diffusion_pytorch_model-00001-of-00004.safetensors": os.urandom(16_384),
        "assets/demo.mp4": os.urandom(1024),
    }
    for relative, data in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = [{"path": relative, "size": len(data)} for relative, data in files.items()]
    (source / "manifest.json").write_text(json.dumps(manifest))
    handler = type("Handler", (RangeHandler,), {"root": str(tmp_path / "server"), "ranges": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", repo, files, handler, tmp_path
    server.shutdown()


def test_mirror_filters_and_downloads(mirror, monkeypatch):
    base, repo, files, _, tmp_path = mirror
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    destination = weights.ensure_weights(repo, allow_patterns=weights.BASE_ALLOW_PATTERNS, base=base)
    assert (destination / "config.json").read_bytes() == files["config.json"]
    assert not (destination / "assets/demo.mp4").exists()
    assert not (destination / ".incomplete").exists()


def test_mirror_resumes_partial_file(mirror, monkeypatch):
    base, repo, files, handler, tmp_path = mirror
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    relative = "diffusion_pytorch_model-00001-of-00004.safetensors"
    partial = tmp_path / "weights" / repo.split("/")[-1] / relative
    partial.parent.mkdir(parents=True)
    partial.write_bytes(files[relative][:1000])
    (partial.parent / ".incomplete").touch()
    destination = weights.ensure_weights(repo, allow_patterns=weights.BASE_ALLOW_PATTERNS, base=base)
    assert (destination / relative).read_bytes() == files[relative]
    assert "bytes=1000-" in handler.ranges
