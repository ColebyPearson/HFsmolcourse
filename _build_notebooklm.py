"""Build notebooklm/unit_N.txt and unit_N/README.md from the cloned source.

Reads _source/units/en/_toctree.yml for the canonical chapter order, then
finds each section's source file (try .md, .mdx). Concatenates the chapters
of each unit into one NotebookLM-friendly .txt and emits a per-unit README
mirroring the chapter list.

Re-run any time the upstream course updates (after `git -C _source pull`).
"""
from __future__ import annotations
import re
from pathlib import Path
import yaml

ROOT = Path(__file__).parent
SRC = ROOT / "_source" / "units" / "en"
NLM = ROOT / "notebooklm"
NLM.mkdir(exist_ok=True)

toc = yaml.safe_load((SRC / "_toctree.yml").read_text(encoding="utf-8"))

FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
IMPORTS = re.compile(r"^import\s+.*$", re.MULTILINE)


def clean(text: str) -> str:
    text = FRONTMATTER.sub("", text, count=1)
    text = IMPORTS.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def find_source(local: str) -> Path | None:
    for ext in (".md", ".mdx"):
        p = SRC / f"{local}{ext}"
        if p.exists():
            return p
    return None


for unit_idx, unit in enumerate(toc):
    title = unit["title"]
    sections = unit["sections"]
    folder = ROOT / f"unit_{unit_idx}"
    folder.mkdir(exist_ok=True)

    parts = [f"# {title}\n"]
    for s in sections:
        local = s["local"]
        sec_title = s["title"]
        path = find_source(local)
        if not path:
            parts.append(f"\n\n## {sec_title}\n\n[missing source: {local}.md(x)]\n")
            continue
        parts.append(f"\n\n## {sec_title}\n\n{clean(path.read_text(encoding='utf-8'))}")
    (NLM / f"unit_{unit_idx}.txt").write_text("".join(parts), encoding="utf-8")

    # --- unit_N/README.md
    lines = [
        f"# Unit {unit_idx} — {title.split('. ', 1)[-1]}",
        "",
        f"Official chapter URL base: https://huggingface.co/learn/smol-course/unit{unit_idx}/",
        f"NotebookLM source: [`notebooklm/unit_{unit_idx}.txt`](../notebooklm/unit_{unit_idx}.txt)",
        "",
        "## Chapters (in order)",
        "",
    ]
    for s in sections:
        anchor = s["local"].split("/", 1)[1]
        lines.append(
            f"- **{s['title']}** — https://huggingface.co/learn/smol-course/unit{unit_idx}/{anchor}"
        )
    lines += [
        "",
        "## Status",
        "",
        "- [ ] Read all chapters (or listen via NotebookLM)",
    ]
    if unit_idx >= 1:
        lines.append("- [ ] Hands-on fine-tune (artifact / model link in this folder)")
    lines.append("")
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"unit_{unit_idx}: {len(sections)} chapter(s) -> notebooklm/unit_{unit_idx}.txt + unit_{unit_idx}/README.md")
