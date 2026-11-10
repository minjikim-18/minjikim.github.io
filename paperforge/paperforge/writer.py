"""Steps 4 and 7: plan the paper, then write the front matter and each section in the template's own markup."""
from __future__ import annotations

import re
from pathlib import Path

from .llm import LLM, ask_json, extract_block
from .template import Template

PLAN_PROMPT = """Plan a manuscript for {journal} from the repository material above.

The journal's sample file (shows the expected structure and markup):
```latex
{sample}
```
{guide}{emphasis}
Reply with JSON:
{{"title": "...",
  "sections": [{{"name": "Introduction", "brief": "what this section must cover, in 1-3 sentences"}}, ...],
  "figures": [{{"id": "fig_short_name", "caption": "...", "description": "what is plotted against what", "data_files": ["path/in/repo.csv"]}}, ...]}}

Follow the section order this journal's sample uses. Propose a figure only when it supports a claim; name the repository files it would be drawn from. Figure ids: lowercase letters, digits and underscores."""

HEAD_PROMPT = """Write the opening of main.tex for our manuscript, using the journal sample below as the model for markup.

Journal sample:
```latex
{sample}
```

Manuscript title: {title}
Authors: {authors}
Planned sections: {sections}

Produce everything from \\documentclass up to, but not including, the first \\section: the preamble, \\begin{{document}}, title, authors and affiliations, abstract, keywords - each in the exact commands this sample uses.
- Keep the \\documentclass line and the sample's \\usepackage lines unchanged. Remove sample-only filler (lipsum, example macros that are unused).
- Write a real abstract from the repository material. Unknown author details (affiliation, email, ORCID) become [TODO: ...].
- Do not include \\section, bibliography commands or \\end{{document}}.
Reply with the LaTeX in a single ```latex block."""

SECTION_PROMPT = """Write the "{name}" section of the manuscript "{title}".

What it must cover: {brief}
All sections, for context: {sections}

Figures that exist (include one with a figure environment where it is first discussed; path and label exactly as given):
{figures}

Verified citation keys you may use with \\cite{{...}} (use no others):
{keys}

Markup conventions of this journal, from its sample:
```latex
{sample_body}
```

Start with \\section{{{name}}}. Write complete prose, not an outline. Every number must come from the repository material; otherwise write [TODO: ...].
For a figure use:
\\begin{{figure}}[t]\\centering\\includegraphics[width=\\linewidth]{{figures/<id>.pdf}}\\caption{{...}}\\label{{fig:<id>}}\\end{{figure}}
Reply with the LaTeX in a single ```latex block."""

DISCLOSURE = r"""
\section*{Declaration of generative AI use}
[TODO: This draft was prepared with paperforge, which uses a large language model (Claude) to draft text and figure code. Before submission, replace this note with the disclosure statement your journal requires, and confirm that every number, claim and citation has been checked by the authors. AI tools cannot be listed as authors.]
"""

CITE_RE = re.compile(r"[~ ]?\\(no)?cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
FIGURE_RE = re.compile(r"\\begin\{figure(\*?)\}.*?\\end\{figure\1\}\n?", re.S)


def plan(llm: LLM, tpl: Template, context: str, journal: str, guide: str, emphasis: str) -> dict:
    data = ask_json(llm, "plan", PLAN_PROMPT.format(
        journal=journal or "the target journal", sample=tpl.source,
        guide=f"\nAuthor guidelines (excerpt):\n{guide}\n" if guide else "",
        emphasis=f"\nThe authors want to emphasise: {emphasis}\n" if emphasis else ""), context)
    data.setdefault("title", "[TODO: title]")
    data["sections"] = [s for s in data.get("sections", []) if isinstance(s, dict) and s.get("name")]
    figures = []
    for fig in data.get("figures", []):
        fid = re.sub(r"[^a-z0-9_]", "_", str(fig.get("id", "")).lower()).strip("_")
        if fid and fid not in {f["id"] for f in figures}:
            figures.append({**fig, "id": fid})
    data["figures"] = figures
    return data


def write_head(llm: LLM, tpl: Template, context: str, plan_: dict, authors: str) -> str:
    answer = llm.complete("frontmatter", HEAD_PROMPT.format(
        sample=tpl.source, title=plan_["title"], authors=authors or "[TODO: authors]",
        sections=", ".join(s["name"] for s in plan_["sections"])), context)
    head = extract_block(answer, "latex")
    # The template class must survive verbatim whatever the model returned.
    head, n = re.subn(r"\\documentclass\s*(\[[^\]]*\])?\s*\{[^}]+\}", lambda _: tpl.class_line, head, count=1)
    if n == 0:
        head = tpl.class_line + "\n" + head
    head = re.sub(r"\\(bibliography|printbibliography|end\{document\})\b.*", "", head)
    head = re.sub(r"\\addbibresource\{[^}]*\}\n?", "", head)
    if r"\begin{document}" not in head:
        head += "\n\\begin{document}\n"
    extra = ""
    if "graphicx" not in head:
        extra += "\\usepackage{graphicx}\n"
    if tpl.biblatex:
        extra += "\\addbibresource{refs.bib}\n"
    return head.replace(r"\begin{document}", extra + r"\begin{document}", 1)


def write_section(llm: LLM, tpl: Template, context: str, plan_: dict, section: dict,
                  figures: list[dict], keys: dict[str, str]) -> str:
    body = tpl.source.split(r"\begin{document}", 1)[-1]
    answer = llm.complete("section", SECTION_PROMPT.format(
        name=section["name"], title=plan_["title"], brief=section.get("brief", ""),
        sections=", ".join(s["name"] for s in plan_["sections"]),
        figures="\n".join(f"- figures/{f['id']}.pdf  label fig:{f['id']}  caption: {f['caption']}" for f in figures) or "(none)",
        keys="\n".join(f"- {k}: {v}" for k, v in keys.items()) or "(none - write without citations)",
        sample_body=body[:6000]), context)
    return extract_block(answer, "latex")


def tail(tpl: Template, has_refs: bool) -> str:
    out = DISCLOSURE
    if has_refs:
        if tpl.biblatex:
            out += "\n\\printbibliography\n"
        else:
            out += f"\n\\bibliographystyle{{{tpl.bib_style or 'plain'}}}\n\\bibliography{{refs}}\n"
    return out + "\n\\end{document}\n"


def strip_unverified_citations(tex: str, valid: set[str]) -> tuple[str, list[str]]:
    """Drop every citation key that is not in `valid`; remove a \\cite left with no keys."""
    removed: list[str] = []

    def repl(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(2).split(",") if k.strip()]
        keep = [k for k in keys if k in valid or k == "*"]
        removed.extend(k for k in keys if k not in keep)
        if not keep:
            return ""
        if len(keep) == len(keys):
            return m.group(0)
        return m.group(0)[: m.group(0).rfind("{")] + "{" + ",".join(keep) + "}"

    return CITE_RE.sub(repl, tex), removed


def drop_missing_figures(tex: str, build: Path) -> tuple[str, list[str]]:
    """Remove figure environments whose image file does not exist in the build directory."""
    dropped: list[str] = []

    def repl(m: re.Match) -> str:
        paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", m.group(0))
        missing = [p for p in paths if not any((build / (p + ext)).exists() for ext in ("", ".pdf", ".png"))]
        if missing or not paths:
            dropped.extend(missing or ["(figure without an image)"])
            return ""
        return m.group(0)

    return FIGURE_RE.sub(repl, tex), dropped
