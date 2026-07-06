import json
from pathlib import Path
from typing import Any


def _strip_json_comments(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    index = 0

    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == "#":
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue

        if char == "/" and next_char == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue

        result.append(char)
        index += 1

    return "".join(result)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON-style .cfg file with line comments allowed."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    return json.loads(_strip_json_comments(text))
