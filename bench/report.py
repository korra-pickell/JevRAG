from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("report_template.html")


def render_report(summary: dict, *, standalone: bool) -> str:
    # Escaping every "<" keeps "</script>" and "<!--" in the data from changing how the
    # HTML parser reads the inline script; JSON.parse / JS read "<" back as "<".
    data = json.dumps(summary).replace("<", "\\u003c")
    body = TEMPLATE.read_text("utf-8").replace("__SUMMARY_JSON__", data)
    if not standalone:
        return body
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, '
            'viewport-fit=cover">\n</head>\n<body>\n' + body + "\n</body>\n</html>\n")


def write_report(summary: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(summary, standalone=True), "utf-8", newline="\n")
    path.with_name(path.stem + ".fragment.html").write_text(
        render_report(summary, standalone=False), "utf-8", newline="\n")
    return path
