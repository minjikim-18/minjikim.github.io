"""Step 2: fetch a research repository and condense it into text an LLM can read."""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .net import assert_public_https
from .template import safe_extract

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "wandb", ".idea"}
DATA_EXT = {".csv", ".tsv", ".json", ".jsonl", ".log", ".txt", ".yaml", ".yml", ".npy", ".npz"}
CODE_EXT = {".py", ".ipynb", ".m", ".jl", ".r", ".cpp", ".c", ".h", ".rs", ".go", ".sh"}
PREVIEW_CHARS = 2500
CODE_CHARS = 6000
CODE_BUDGET = 60000
TREE_LIMIT = 400


@dataclass
class RepoSummary:
    root: Path
    text: str
    data_files: list[str]
    bib_files: list[Path]
    truncated: list[str] = field(default_factory=list)


def fetch(source: str, dest: Path) -> Path:
    """`source` is an https git URL, a .zip path or a local directory."""
    dest = Path(dest)
    if source.startswith(("http://", "https://")):
        url = assert_public_https(source)
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        token = os.environ.get("GITHUB_TOKEN")
        if token and "github.com" in url:
            # Passed through the environment so the token never shows up in argv.
            basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
            env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                       GIT_CONFIG_VALUE_0=f"Authorization: Basic {basic}")
        proc = subprocess.run(["git", "clone", "--depth", "1", "--", url, str(dest)],
                              env=env, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"git clone failed: {proc.stderr.strip()[-400:]}")
        return dest
    path = Path(source)
    if path.is_dir():
        shutil.copytree(path, dest, ignore=shutil.ignore_patterns(*SKIP_DIRS), symlinks=False)
        return dest
    if path.suffix.lower() == ".zip":
        safe_extract(path, dest)
        entries = [p for p in dest.iterdir() if p.name != "__MACOSX"]
        return entries[0] if len(entries) == 1 and entries[0].is_dir() else dest
    raise ValueError(f"repo must be an https URL, a .zip or a directory: {source!r}")


def _files(root: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts) or path.is_symlink() or not path.is_file():
            continue
        yield path, rel


def _preview(path: Path, limit: int) -> tuple[str, bool]:
    if path.suffix in {".npy", ".npz"}:
        return f"(binary numpy file, {path.stat().st_size} bytes)", False
    text = path.read_text(errors="replace")
    if len(text) <= limit:
        return text, False
    if path.suffix == ".log":  # the end of a log holds the final metrics
        return text[: limit // 3] + "\n...[middle omitted]...\n" + text[-2 * limit // 3:], True
    return text[:limit] + "\n...[rest omitted]...", True


def summarize(root: Path) -> RepoSummary:
    root = Path(root)
    files = list(_files(root))
    truncated: list[str] = []
    parts: list[str] = []

    tree = [f"{rel}  ({path.stat().st_size} B)" for path, rel in files]
    if len(tree) > TREE_LIMIT:
        truncated.append(f"file tree: showing {TREE_LIMIT} of {len(tree)} files")
        tree = tree[:TREE_LIMIT]
    parts.append("## File tree\n" + "\n".join(tree))

    for path, rel in files:
        if path.name.lower().startswith("readme") and len(rel.parts) <= 2:
            text, cut = _preview(path, 12000)
            truncated += [f"{rel} (README cut)"] if cut else []
            parts.append(f"## {rel}\n{text}")

    data_files, bib_files = [], []
    for path, rel in files:
        if path.suffix.lower() == ".bib":
            bib_files.append(path)
        elif path.suffix.lower() in DATA_EXT and not path.name.lower().startswith(("readme", "requirements", "license")):
            data_files.append(str(rel))
            text, cut = _preview(path, PREVIEW_CHARS)
            truncated += [f"{rel} (preview only)"] if cut else []
            parts.append(f"## Result/data file: {rel}\n```\n{text}\n```")

    budget = CODE_BUDGET
    code = [(p, r) for p, r in files if p.suffix.lower() in CODE_EXT]
    # Entry points and shallow files tell the most about what the project does.
    code.sort(key=lambda pr: (len(pr[1].parts), not any(k in pr[1].name for k in ("main", "train", "run", "eval", "model"))))
    for path, rel in code:
        if budget <= 0:
            truncated.append(f"{rel} (code not shown: summary budget used)")
            continue
        text, cut = _preview(path, min(CODE_CHARS, budget))
        truncated += [f"{rel} (code cut)"] if cut else []
        budget -= len(text)
        parts.append(f"## Code: {rel}\n```\n{text}\n```")

    return RepoSummary(root, "\n\n".join(parts), data_files, bib_files, truncated)
