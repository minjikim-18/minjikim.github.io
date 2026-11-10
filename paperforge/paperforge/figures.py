"""Step 5: the LLM writes matplotlib code, we run it on the repo's real data, the LLM looks at the result."""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .llm import LLM, ask_json, extract_block

MAX_ATTEMPTS = 3
TIMEOUT_S = 180


@dataclass
class FigureResult:
    id: str
    caption: str
    status: str            # "ok" | "held"
    note: str = ""
    attempts: int = 0


class SandboxRequired(RuntimeError):
    pass


def _in_container() -> bool:
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def check_sandbox() -> None:
    """LLM-written code only runs inside a container (or with an explicit opt-in)."""
    if not (_in_container() or os.environ.get("PAPERFORGE_ALLOW_UNSANDBOXED") == "1"):
        raise SandboxRequired(
            "Figure scripts are written by an LLM and run only inside the Docker image. "
            "Use `docker compose up`, or set PAPERFORGE_ALLOW_UNSANDBOXED=1 on a machine you accept the risk on.")


def _limits() -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_FSIZE, (200 * 1024 * 1024,) * 2)


def run_script(script: Path, repo: Path, out_stem: Path, style: Path, scratch: Path) -> tuple[bool, str]:
    scratch.mkdir(parents=True, exist_ok=True)
    env = {  # a clean environment: no API keys or tokens reach generated code
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(scratch), "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(scratch / "mpl"), "PF_REPO": str(repo), "PF_OUT": str(out_stem), "PF_STYLE": str(style),
    }
    try:
        proc = subprocess.run([sys.executable, "-I", str(script)], cwd=scratch, env=env, capture_output=True,
                              text=True, timeout=TIMEOUT_S, preexec_fn=_limits if os.name == "posix" else None)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {TIMEOUT_S}s"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout)[-3000:]
    missing = [ext for ext in (".pdf", ".png") if not out_stem.with_suffix(ext).exists()]
    if missing:
        return False, f"script finished but did not write {', '.join(str(out_stem) + m for m in missing)}"
    return True, proc.stdout[-1000:]


CODE_PROMPT = """Write a standalone Python script that draws this figure with matplotlib.

Figure id: {id}
Caption: {caption}
What it should show: {description}
Data files suggested by the plan (relative to the repo root): {data_files}

Contract:
- Read data only from the repository at os.environ["PF_REPO"]. Every plotted value must come from those files. Do not type data values into the script and do not generate synthetic or random data.
- Call plt.style.use(os.environ["PF_STYLE"]) first, and do not override fonts or colours afterwards.
- Save to os.environ["PF_OUT"] + ".pdf" and os.environ["PF_OUT"] + ".png" (dpi 200 for the png).
- Size it for a journal column (about 3.4 in wide), label axes with units when the data states them, no title inside the figure.
- Available: numpy, pandas, matplotlib, the standard library.

If the repository does not contain the data this figure needs, do not draw it. Reply with one line: NO_DATA: <what is missing>
Otherwise reply with the script in a single ```python block.{feedback}"""

REVIEW_PROMPT = """This is the rendered figure "{id}" (intended caption: {caption}).
Check it as a reviewer would: overlapping or clipped text, unreadable labels, missing axis labels or legend, empty axes, obviously wrong plot type.
Reply with JSON: {{"ok": true|false, "issues": ["..."]}}. Say ok when it is publishable as a draft figure; minor taste issues are fine."""


def make_figure(llm: LLM, spec: dict, context: str, repo: Path, figdir: Path, style: Path, scratch: Path) -> FigureResult:
    fid, caption = spec["id"], spec.get("caption", "")
    scripts = figdir / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    out_stem, script = figdir / fid, scripts / f"{fid}.py"
    feedback, last_error = "", ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        answer = llm.complete("figure_code", CODE_PROMPT.format(
            id=fid, caption=caption, description=spec.get("description", ""),
            data_files=spec.get("data_files") or "(none given)", feedback=feedback), context)
        if answer.strip().startswith("NO_DATA"):
            return FigureResult(fid, caption, "held", answer.strip().partition(":")[2].strip() or "no data", attempt)
        script.write_text(extract_block(answer, "python") + "\n")
        for ext in (".pdf", ".png"):
            out_stem.with_suffix(ext).unlink(missing_ok=True)
        ok, log = run_script(script, repo, out_stem, style, scratch)
        if not ok:
            last_error = log
            feedback = f"\n\nYour previous script failed. Fix it.\nPrevious script:\n```python\n{script.read_text()}```\nError:\n{log}"
            continue
        review = ask_json(llm, "figure_review", REVIEW_PROMPT.format(id=fid, caption=caption),
                          images=[out_stem.with_suffix(".png")])
        if review.get("ok", True) or attempt == MAX_ATTEMPTS:
            note = "" if review.get("ok", True) else "kept despite review issues: " + "; ".join(review.get("issues", []))
            return FigureResult(fid, caption, "ok", note, attempt)
        feedback = (f"\n\nYour previous script ran, but the rendered figure has problems: {'; '.join(review.get('issues', []))}\n"
                    f"Previous script:\n```python\n{script.read_text()}```")
    for ext in (".pdf", ".png"):
        out_stem.with_suffix(ext).unlink(missing_ok=True)
    return FigureResult(fid, caption, "held", f"script kept failing: {last_error[-300:]}", MAX_ATTEMPTS)
