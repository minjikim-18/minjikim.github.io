"""Demo mode: run the whole pipeline without an API key by replaying prepared answers.

`ReplayLLM` reads {"task": [answer, ...]} and hands the answers out in order per
task. An answer may be {"edit_main_tex": [old, new]}: it returns the main.tex
found in the prompt with that replacement applied (used for the compile-repair step).
`RecordingLLM` wraps a real LLM and saves its answers in the same format.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .llm import LLM, LLMError


class ReplayLLM:
    def __init__(self, script: Path | dict):
        data = json.loads(Path(script).read_text()) if not isinstance(script, dict) else script
        self.answers = {k: list(v) for k, v in data.items() if not k.startswith("_")}
        self.calls: list[str] = []

    def complete(self, task: str, prompt: str, context: str = "", images: Sequence[Path] = ()) -> str:
        self.calls.append(task)
        queue = self.answers.get(task)
        if not queue:
            raise LLMError(f"replay script has no answer left for task '{task}'")
        answer = queue.pop(0)
        if isinstance(answer, dict) and "edit_main_tex" in answer:
            old, new = answer["edit_main_tex"]
            tex = re.findall(r"```latex\n(.*?)```", prompt, re.S)[-1]
            return "```latex\n" + tex.replace(old, new) + "```"
        return answer if isinstance(answer, str) else json.dumps(answer)


class RecordingLLM:
    def __init__(self, inner: LLM, path: Path):
        self.inner, self.path, self.log = inner, Path(path), defaultdict(list)

    def complete(self, task: str, prompt: str, context: str = "", images: Sequence[Path] = ()) -> str:
        answer = self.inner.complete(task, prompt, context, images)
        self.log[task].append(answer)
        self.path.write_text(json.dumps(self.log, indent=2, ensure_ascii=False))
        return answer

    def __getattr__(self, name):
        return getattr(self.inner, name)
