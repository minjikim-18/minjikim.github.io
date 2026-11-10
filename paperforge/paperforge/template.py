"""Step 1: unpack a journal template safely and work out how it is built."""
from __future__ import annotations

import re
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

MAX_FILES = 5000
MAX_TOTAL_BYTES = 500 * 1024 * 1024
RC_NAMES = {"latexmkrc", ".latexmkrc"}


class UnsafeArchive(ValueError):
    pass


def safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract a zip, refusing path escapes, symlinks and zip bombs."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_FILES:
            raise UnsafeArchive(f"archive has more than {MAX_FILES} entries")
        if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
            raise UnsafeArchive("archive is too large when unpacked")
        for info in infos:
            name = info.filename
            if stat.S_ISLNK(info.external_attr >> 16):
                raise UnsafeArchive(f"symlink in archive: {name}")
            target = (dest / name).resolve()
            if name.startswith(("/", "\\")) or not target.is_relative_to(dest):
                raise UnsafeArchive(f"path escapes the archive root: {name}")
        for info in infos:
            target = dest / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)


@dataclass
class Template:
    root: Path              # directory holding the main .tex and its support files
    main_tex: Path
    source: str             # full text of the sample main .tex
    doc_class: str
    class_line: str         # the \documentclass line, verbatim
    engine: str             # pdflatex | xelatex | lualatex
    biblatex: bool
    bib_style: str | None   # \bibliographystyle argument, if the template uses BibTeX
    removed_rc: list[str] = field(default_factory=list)


def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def find_main_tex(root: Path) -> Path:
    candidates = []
    for path in root.rglob("*.tex"):
        if "__MACOSX" in path.parts:
            continue
        text = _strip_comments(path.read_text(errors="replace"))
        if r"\documentclass" not in text:
            continue
        score = 0
        score += 10 if r"\begin{document}" in text else 0
        score += 3 if re.search(r"main|template|sample|manuscript|article", path.stem, re.I) else 0
        score -= len(path.relative_to(root).parts)
        candidates.append((score, len(text), path))
    if not candidates:
        raise ValueError("no .tex file with \\documentclass found in the template archive")
    return max(candidates)[2]


def analyze(root: Path) -> Template:
    main = find_main_tex(Path(root))
    source = main.read_text(errors="replace")
    bare = _strip_comments(source)
    m = re.search(r"\\documentclass\s*(\[[^\]]*\])?\s*\{([^}]+)\}", bare)
    class_line, doc_class = m.group(0), m.group(2).strip()

    preamble = bare.split(r"\begin{document}")[0]
    engine = "pdflatex"
    first_lines = "\n".join(source.splitlines()[:5]).lower()
    if "xelatex" in first_lines or re.search(r"\\usepackage(\[[^\]]*\])?\{[^}]*(fontspec|xeCJK|kotex)[^}]*\}", preamble):
        engine = "xelatex"
    if "lualatex" in first_lines or "luatexja" in preamble:
        engine = "lualatex"

    biblatex = bool(re.search(r"\{biblatex\}|\\addbibresource", bare))
    style = re.search(r"\\bibliographystyle\s*\{([^}]+)\}", bare)
    bib_style = style.group(1).strip() if style else None
    if not biblatex and bib_style is None:
        bsts = sorted(main.parent.glob("*.bst"))
        bib_style = bsts[0].stem if bsts else None

    # A template's latexmkrc is arbitrary Perl; never let latexmk read it.
    removed = []
    for rc in list(Path(root).rglob("*")):
        if rc.is_file() and rc.name.lower() in RC_NAMES:
            removed.append(str(rc.relative_to(root)))
            rc.unlink()

    return Template(main.parent, main, source, doc_class, class_line, engine, biblatex, bib_style, removed)
