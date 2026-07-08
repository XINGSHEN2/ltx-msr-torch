from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib import error, request


def queue_prompt(
    api_prompt: dict[str, Any],
    server: str = "127.0.0.1:8188",
) -> dict[str, Any]:
    payload = json.dumps({"prompt": api_prompt}).encode("utf-8")
    req = request.Request(
        f"http://{server}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    return _open_json(req, timeout=30)


def get_history(prompt_id: str, server: str = "127.0.0.1:8188") -> dict[str, Any]:
    req = request.Request(f"http://{server}/history/{prompt_id}")
    return _open_json(req, timeout=30)


def wait_for_history(
    prompt_id: str,
    server: str = "127.0.0.1:8188",
    poll_seconds: float = 5.0,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    start = time.time()
    while True:
        history = get_history(prompt_id, server=server)
        if prompt_id in history:
            return history[prompt_id]
        if timeout_seconds is not None and time.time() - start > timeout_seconds:
            raise TimeoutError(f"timed out waiting for ComfyUI prompt {prompt_id}")
        time.sleep(poll_seconds)


def load_api_prompt(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _open_json(req: request.Request, timeout: int) -> dict[str, Any]:
    opener = request.build_opener(request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI HTTP {exc.code}: {body}") from exc
