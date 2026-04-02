"""
nova.net — Networking constellation (prototype)

This is a lightweight Python stdlib backing for NOVA's early demos.
It is intentionally small: enough to build simple HTTP services.
"""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Response:
    status: int = 200
    body: str = ""
    content_type: str = "text/plain; charset=utf-8"

    @staticmethod
    def ok(body: str = "ok") -> "Response":
        return Response(status=200, body=body)

    @staticmethod
    def not_found(body: str = "not found") -> "Response":
        return Response(status=404, body=body)


HandlerFn = Callable[[BaseHTTPRequestHandler], Response]


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    handler: HandlerFn


def get(path: str) -> Callable[[HandlerFn], Route]:
    def deco(fn: HandlerFn) -> Route:
        return Route(method="GET", path=path, handler=fn)

    return deco


def post(path: str) -> Callable[[HandlerFn], Route]:
    def deco(fn: HandlerFn) -> Route:
        return Route(method="POST", path=path, handler=fn)

    return deco


def route(routes: List[Route]) -> Dict[Tuple[str, str], HandlerFn]:
    return {(r.method.upper(), r.path): r.handler for r in routes}


def serve(port: int, routes: Union[List[Route], Dict[Tuple[str, str], HandlerFn]]) -> None:
    table = route(routes) if isinstance(routes, list) else routes

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self._handle("GET")

        def do_POST(self):  # noqa: N802
            self._handle("POST")

        def _handle(self, method: str) -> None:
            handler = table.get((method, self.path))
            resp = handler(self) if handler else Response.not_found()
            body_bytes = resp.body.encode("utf-8")
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.content_type)
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def log_message(self, fmt: str, *args) -> None:  # quiet default logs
            return

    server = HTTPServer(("127.0.0.1", int(port)), Handler)
    server.serve_forever()


def http_get(url: str, timeout_s: float = 10.0) -> str:
    with urlopen(url, timeout=float(timeout_s)) as resp:  # nosec - caller controlled
        return resp.read().decode("utf-8", errors="replace")


def http_post(url: str, body: str, content_type: str = "text/plain; charset=utf-8", timeout_s: float = 10.0) -> str:
    data = body.encode("utf-8")
    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(data)))
    with urlopen(req, timeout=float(timeout_s)) as resp:  # nosec - caller controlled
        return resp.read().decode("utf-8", errors="replace")

