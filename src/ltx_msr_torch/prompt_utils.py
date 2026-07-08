from __future__ import annotations

from pathlib import Path


def parse_reference_prompt(text: str) -> tuple[str, str]:
    """Split an MSR prompt into global reference text and local motion text."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    reference_lines: list[str] = []
    local_lines: list[str] = []
    in_local = False

    for line in lines:
        if not line:
            continue
        if not in_local and line.startswith("参考"):
            reference_lines.append(line)
        else:
            in_local = True
            local_lines.append(line)

    return "\n".join(reference_lines), " | ".join(local_lines)


def parse_reference_prompt_file(path: str | Path) -> tuple[str, str]:
    return parse_reference_prompt(Path(path).read_text())
