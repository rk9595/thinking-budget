"""Dependency-free helpers shared by experiment entry points."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_prompt(problem, budget, wording="exact"):
    phrases = {"exact": f"Think for exactly {budget} tokens.",
               "max": f"Think for maximum {budget} tokens.",
               "l1": f"Think for {budget} tokens."}
    return f"{problem}\n\n{INSTRUCTION} {phrases[wording]}"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def environment():
    versions = {}
    for name in ("torch", "transformers", "trl", "vllm", "peft", "datasets", "math-verify"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cwd = Path(__file__).resolve().parent
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True)
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=cwd, capture_output=True, text=True)
    return {"created_at": datetime.now(timezone.utc).isoformat(), "versions": versions,
            "git_commit": sha.stdout.strip(), "tracked_diff_sha256": digest(diff.stdout),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(cwd.iterdir())
                              if p.is_file() and p.suffix in (".py", ".sh", ".in", ".txt")}}


def artifact_path(summary_path, value):
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    return Path(summary_path).parent / path.name


def resolve_model(model, revision=None):
    if Path(model).is_dir():
        return str(Path(model).resolve()), None
    from huggingface_hub import model_info
    return model, model_info(model, revision=revision).sha


def read_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
