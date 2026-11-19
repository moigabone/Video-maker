from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Workspace  # noqa: E402
from tests.make_fixtures import FIXTURES, make_fixtures  # noqa: E402

TEST_WS = ROOT / "tests" / "workspace"
SMOKE_OUTPUT = ROOT / "tests" / "smoke_output.mp4"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return make_fixtures(FIXTURES)


@pytest.fixture(scope="session")
def input_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "input"


@pytest.fixture(scope="session")
def ws() -> Workspace:
    if TEST_WS.exists():
        shutil.rmtree(TEST_WS)
    return Workspace.at(TEST_WS)


# ----------------------------------------------------------------------------- faux serveur Claude

def sse_message(text: str, model: str = "claude-opus-5-5", stop_reason: str = "end_turn") -> bytes:
    """Flux SSE Messages API minimal renvoyant `text` dans un bloc texte."""
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                           "usage": {"output_tokens": 200}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


class FakeClaude:
    """Client Anthropic réel branché sur un transport httpx simulé : capture les requêtes, renvoie `responder(body)`."""

    def __init__(self, responder: Callable[[dict], str]):
        import anthropic
        import httpx2 as httpx  # anthropic>=1.x embarque httpx2

        self.requests: list[dict] = []
        self.headers: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.requests.append(body)
            self.headers.append(dict(request.headers))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_message(responder(body)))

        self.client = anthropic.Anthropic(api_key="sk-ant-test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.fixture
def fake_claude() -> Callable[[Callable[[dict], str]], FakeClaude]:
    return FakeClaude
