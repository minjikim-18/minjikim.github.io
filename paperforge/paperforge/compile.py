"""Step 8: compile with latexmk; on failure let the LLM repair main.tex (bounded retries)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .llm import LLM, extract_block

MAX_FIXES = 3
ENGINE_FLAG = {"pdflatex": "-pdf", "xelatex": "-xelatex", "lualatex": "-lualatex"}


@dataclass
class CompileResult:
    ok: bool
    fixes: int = 0
    errors: list[str] = field(default_factory=list)   # one excerpt per failed attempt


def run_latexmk(build: Path, engine: str, timeout: int = 300) -> tuple[bool, str]:
    if not shutil.which("latexmk"):
        return False, "latexmk is not installed (use the Docker image, or install TeX Live)"
    cmd = ["latexmk", "-norc",                       # never read a latexmkrc
           ENGINE_FLAG.get(engine, "-pdf"), "-interaction=nonstopmode", "-halt-on-error",
           "-file-line-error", "-no-shell-escape", "main.tex"]
    env = dict(os.environ, openout_any="p", openin_any="p")   # TeX may not write/read outside the build tree
    try:
        proc = subprocess.run(cmd, cwd=build, env=env, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"latexmk timed out after {timeout}s"
    ok = proc.returncode == 0 and (build / "main.pdf").exists()
    log = build / "main.log"
    return ok, (log.read_text(errors="replace") if log.exists() else proc.stdout + proc.stderr)


def error_excerpt(log: str, limit: int = 4000) -> str:
    lines = log.splitlines()
    hits = [i for i, line in enumerate(lines) if line.startswith("!") or re.match(r".+:\d+: ", line)]
    if not hits:
        return "\n".join(lines[-40:])[-limit:]
    out = []
    for i in hits[:5]:
        out += lines[i: i + 8] + ["..."]
    return "\n".join(out)[:limit]


FIX_PROMPT = """main.tex fails to compile with {engine}. Fix the cause with the smallest change; do not rewrite content, and keep the \\documentclass line and the template's packages.

Error excerpt from the log:
```
{error}
```

Current main.tex:
```latex
{tex}
```

Reply with the complete corrected main.tex in a single ```latex block."""


def clean_aux(build: Path) -> None:
    for pattern in ("main.aux", "main.bbl", "main.blg", "main.log", "main.out", "main.fls", "main.fdb_latexmk",
                    "main.bcf", "main.run.xml", "main.toc", "main.xdv", "main.synctex.gz"):
        (build / pattern).unlink(missing_ok=True)


def compile_with_repair(llm: LLM, build: Path, engine: str,
                        sanitize: Callable[[str], str] = lambda tex: tex) -> CompileResult:
    result = CompileResult(ok=False)
    main = build / "main.tex"
    while True:
        ok, log = run_latexmk(build, engine)
        if ok:
            result.ok = True
            break
        result.errors.append(error_excerpt(log))
        if result.fixes >= MAX_FIXES or "is not installed" in log:
            break
        answer = llm.complete("fix_tex", FIX_PROMPT.format(engine=engine, error=result.errors[-1], tex=main.read_text()))
        fixed = extract_block(answer, "latex")
        if r"\begin{document}" not in fixed or r"\end{document}" not in fixed:
            break
        main.write_text(sanitize(fixed))   # re-apply the citation/figure guards to the repaired text
        result.fixes += 1
        clean_aux(build)
        (build / "main.pdf").unlink(missing_ok=True)
    clean_aux(build)
    return result
