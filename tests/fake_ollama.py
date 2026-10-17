"""A tiny fake of the Ollama / OpenAI-compatible HTTP APIs, used only by the tests.

It lets us verify the REAL client code (request format, options, streaming, error handling)
without needing Ollama or a model. It records every request so tests can inspect them.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ragqa.embeddings import HashEmbedder

_PREFIX = re.compile(r"^search_(document|query): ")


class FakeOllama:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.installed_models = ["qwen2.5:3b", "nomic-embed-text:latest"]
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _reply(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                outer.requests.append((self.path, {}))
                if self.path == "/api/tags":
                    return self._reply(200, {"models": [{"name": n} for n in outer.installed_models]})
                if self.path == "/v1/models":
                    return self._reply(200, {"data": [{"id": "some-model"}]})
                self._reply(404, {"error": "unknown path"})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                model = body.get("model", "")
                if model == "missing":
                    return self._reply(404, {"error": f"model '{model}' not found"})
                if model == "boom":
                    return self._reply(500, {"error": "internal error"})

                if self.path == "/api/embed":
                    emb = HashEmbedder()
                    vecs = [emb.embed_documents([_PREFIX.sub("", t)])[0] for t in body["input"]]
                    return self._reply(200, {"model": model, "embeddings": vecs})

                if self.path == "/api/chat":
                    reply = f"Fake answer from {model}. [1]"
                    if body.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "application/x-ndjson")
                        self.end_headers()
                        for word in reply.split(" "):
                            line = {"message": {"role": "assistant", "content": word + " "}, "done": False}
                            self.wfile.write((json.dumps(line) + "\n").encode())
                        self.wfile.write((json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode())
                        return
                    return self._reply(200, {"message": {"role": "assistant", "content": reply}, "done": True})

                if self.path == "/v1/chat/completions":
                    return self._reply(200, {"choices": [{"message": {"content": f"OpenAI-style answer from {model}"}}]})

                self._reply(404, {"error": "unknown path"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
