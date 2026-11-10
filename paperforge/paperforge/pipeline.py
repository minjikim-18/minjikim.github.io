"""The eight steps end to end, the REPORT.md, and output.zip."""
from __future__ import annotations

import html
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests

from . import compile as tex, figures, references, repo, slack_style, template, writer
from .llm import LLM
from .net import assert_public_https

STEPS = ["Template analysis", "Repository summary", "Figure style", "Paper plan",
         "Figures", "References", "Writing", "Compilation"]
Progress = Callable[[int, str, str], None]   # (step index, "running"|"done"|"failed", detail)


@dataclass
class Config:
    template_zip: Path
    repo_source: str                 # https URL, .zip path or directory
    out_zip: Path
    workdir: Path
    journal: str = ""
    guide_url: str = ""
    authors: str = ""
    emphasis: str = ""
    slack_channel: str = ""
    offline: bool = False            # skip Crossref/arXiv (demo mode)


@dataclass
class Outcome:
    out_zip: Path
    compiled: bool
    report: str
    figures: list[figures.FigureResult] = field(default_factory=list)


def _guide_text(url: str) -> tuple[str, str]:
    if not url:
        return "", ""
    try:
        resp = requests.get(assert_public_https(url), timeout=30, headers={"User-Agent": "paperforge/0.1"})
        resp.raise_for_status()
    except (requests.RequestException, ValueError) as exc:
        return "", f"author guide could not be fetched ({exc})"
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", resp.text)
    text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())
    note = "author guide fetched"
    if len(text) > 20000:
        text, note = text[:20000], "author guide fetched (only the first 20,000 characters were used)"
    return text, note


def run(cfg: Config, llm: LLM, progress: Progress = lambda i, s, d: None) -> Outcome:
    figures.check_sandbox()
    work = Path(cfg.workdir)
    build = work / "build"
    notes: list[str] = []

    def step(i: int, detail: str = ""):
        if i > 0:
            progress(i - 1, "done", "")
        progress(i, "running", detail)

    # 1. template
    step(0)
    template.safe_extract(cfg.template_zip, work / "template")
    tpl = template.analyze(work / "template")
    shutil.copytree(tpl.root, build)
    (build / tpl.main_tex.name).unlink()            # the sample is replaced by our main.tex
    if tpl.removed_rc:
        notes.append(f"ignored and removed from the template: {', '.join(tpl.removed_rc)}")

    # 2. repository
    step(1)
    repo_root = repo.fetch(cfg.repo_source, work / "repo")
    summary = repo.summarize(repo_root)
    context = "# Repository material (the only source of facts)\n\n" + summary.text

    # 3. figure style
    step(2)
    style_text, style_note = slack_style.build_style(llm, cfg.slack_channel, work)
    (build / "figures").mkdir(exist_ok=True)
    style_path = build / "figures" / "lab.mplstyle"
    style_path.write_text(style_text)

    # 4. plan
    step(3)
    guide, guide_note = _guide_text(cfg.guide_url)
    notes += [guide_note] if guide_note else []
    plan = writer.plan(llm, tpl, context, cfg.journal, guide, cfg.emphasis)

    # 5. figures
    results = []
    for n, spec in enumerate(plan["figures"], 1):
        step(4, f"{n}/{len(plan['figures'])}: {spec['id']}") if n == 1 else progress(4, "running", f"{n}/{len(plan['figures'])}: {spec['id']}")
        results.append(figures.make_figure(llm, spec, context, repo_root, build / "figures", style_path, work / "sandbox"))
    if not plan["figures"]:
        step(4, "no figures planned")
    made = [f for f in plan["figures"] if any(r.id == f["id"] and r.status == "ok" for r in results)]

    # 6. references
    step(5)
    bib = references.build(llm, context, summary.bib_files,
                           references.offline_fetch if cfg.offline else references.http_fetch)
    if bib.keys:
        (build / "refs.bib").write_text(bib.text)

    # 7. writing
    step(6, "front matter")
    head = writer.write_head(llm, tpl, context, plan, cfg.authors)
    sections = []
    for n, section in enumerate(plan["sections"], 1):
        progress(6, "running", f"{n}/{len(plan['sections'])}: {section['name']}")
        sections.append(writer.write_section(llm, tpl, context, plan, section, made, bib.keys))
    removed_cites: list[str] = []
    dropped_figs: list[str] = []

    def sanitize(text: str) -> str:
        text, removed = writer.strip_unverified_citations(text, set(bib.keys))
        text, dropped = writer.drop_missing_figures(text, build)
        removed_cites.extend(k for k in removed if k not in removed_cites)
        dropped_figs.extend(d for d in dropped if d not in dropped_figs)
        return text

    main = sanitize(head + "\n\n" + "\n\n".join(sections) + "\n" + writer.tail(tpl, bool(bib.keys)))
    (build / "main.tex").write_text(main)

    # 8. compile
    step(7)
    compiled = tex.compile_with_repair(llm, build, tpl.engine, sanitize)
    progress(7, "done" if compiled.ok else "failed", "" if compiled.ok else "main.pdf was not produced; see REPORT.md")

    final_tex = (build / "main.tex").read_text()
    report = _report(cfg, tpl, summary, plan, results, bib, removed_cites, dropped_figs, compiled, style_note,
                     notes, final_tex.count("[TODO"), llm)
    (build / "REPORT.md").write_text(report)
    for png in (build / "figures").glob("*.png"):   # previews were only needed for the visual review
        png.unlink()
    cfg.out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cfg.out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(build.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(build))
    return Outcome(cfg.out_zip, compiled.ok, report, results)


def _report(cfg, tpl, summary, plan, results, bib, removed_cites, dropped_figs, compiled, style_note, notes, todos, llm) -> str:
    lines = [
        "# paperforge report", "",
        "**This is a draft.** Every number, claim and citation must be checked by the authors before submission.", "",
        "## Summary", "",
        f"- Title: {plan['title']}",
        f"- Template class: `{tpl.doc_class}` (engine: {tpl.engine}, bibliography: {'biblatex' if tpl.biblatex else 'BibTeX ' + (tpl.bib_style or 'plain')})",
        f"- Compiled: {'yes, main.pdf included' if compiled.ok else 'NO - main.pdf missing, see errors below'}"
        + (f" (after {compiled.fixes} automatic repair(s))" if compiled.fixes else ""),
        f"- `[TODO]` markers left in main.tex: {todos}",
        f"- Figure style: {style_note}", "",
        "## Figures", "",
    ]
    for r in results:
        if r.status == "ok":
            lines.append(f"- `{r.id}`: drawn from repository data in {r.attempts} attempt(s); script `figures/scripts/{r.id}.py`"
                         + (f" - {r.note}" if r.note else ""))
        else:
            lines.append(f"- `{r.id}`: **held, not drawn** - {r.note}")
    lines += ["- (none planned)"] if not results else []
    lines += ["", "## References", "", f"- {len(bib.keys)} verified entries in refs.bib"]
    lines += [f"- {n}" for n in bib.notes]
    if bib.rejected:
        lines += ["- Proposed by the model but dropped because they could not be verified:"] + [f"  - {r}" for r in bib.rejected]
    if removed_cites:
        lines.append(f"- Citation keys removed from the text because they were not verified: {', '.join(sorted(set(removed_cites)))}")
    if dropped_figs:
        lines += ["", "## Figure environments removed from the text", ""] + [f"- {d} (file did not exist; a `\\ref` to it may now show ??)" for d in dropped_figs]
    if compiled.errors:
        lines += ["", "## Compilation errors encountered", ""]
        for i, err in enumerate(compiled.errors, 1):
            fixed = i <= compiled.fixes
            lines += [f"Attempt {i} ({'repaired automatically' if fixed else 'not repaired'}):", "```", err.strip(), "```"]
    if summary.truncated:
        lines += ["", "## Repository material the model did not see in full", ""] + [f"- {t}" for t in summary.truncated]
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    usage = getattr(llm, "usage", None)
    if usage:
        lines += ["", "## Model usage", "", f"- Model: {getattr(llm, 'model', '?')}; calls: {usage['calls']}; "
                  f"input tokens: {usage['input_tokens']} (+{usage['cache_read_input_tokens']} cached); output tokens: {usage['output_tokens']}"]
    lines += ["", "## Before you submit", "",
              "- Replace every `[TODO]`, including the generative-AI disclosure at the end of main.tex.",
              "- Re-run `figures/scripts/*.py` yourself (set PF_REPO, PF_OUT, PF_STYLE) and compare against your own plots.",
              "- Upload this zip to Overleaf: New Project -> Upload Project.", ""]
    return "\n".join(lines)
