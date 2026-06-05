#!/usr/bin/env python3
"""Convert MENTOR_REPORT.md -> styled HTML -> PDF via headless Chrome."""
import subprocess
import sys
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "MENTOR_REPORT.md"
HTML = REPO / "MENTOR_REPORT.html"
PDF = REPO / "MENTOR_REPORT.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: Letter; margin: 0.6in 0.7in; }
body {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #222;
  max-width: 7.2in;
  margin: 0 auto;
}
h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 13pt; margin-top: 22px; border-bottom: 1px solid #ccc; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 11pt; margin-top: 16px; page-break-after: avoid; }
p, li { font-size: 10pt; }
code {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 9pt;
  background: #f4f4f4;
  padding: 1px 4px;
  border-radius: 3px;
}
pre {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 8pt;
  background: #f6f8fa;
  padding: 8px 10px;
  border-radius: 5px;
  border: 1px solid #e1e4e8;
  overflow-x: auto;
  page-break-inside: avoid;
}
pre code { background: transparent; padding: 0; font-size: 8pt; }
table {
  border-collapse: collapse;
  margin: 8px 0;
  font-size: 9pt;
  width: 100%;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #d0d7de;
  padding: 4px 8px;
  text-align: left;
  vertical-align: top;
}
th { background: #f0f3f6; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote {
  margin: 10px 0;
  padding: 6px 14px;
  border-left: 4px solid #6c8ebf;
  background: #f3f7fb;
  color: #333;
  font-style: italic;
  page-break-inside: avoid;
}
blockquote p { margin: 4px 0; }
hr { border: none; border-top: 1px solid #ccc; margin: 16px 0; }
strong { color: #111; }
a { color: #0366d6; text-decoration: none; }
ul, ol { margin: 4px 0 8px 0; padding-left: 22px; }
li { margin: 2px 0; }
h2, h3, h4 { page-break-after: avoid; }
"""


def main() -> int:
    if not SRC.exists():
        print(f"Source not found: {SRC}", file=sys.stderr)
        return 1

    md_text = SRC.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"guess_lang": False, "noclasses": True}},
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AFDPS Gamma-Scaling Analysis - Mentor Report</title>
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    HTML.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {HTML}")

    cmd = [
        CHROME,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        f"file://{HTML}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Chrome stderr:", result.stderr, file=sys.stderr)
        return 1

    print(f"Wrote {PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
