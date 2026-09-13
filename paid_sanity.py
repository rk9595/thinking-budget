"""Lifecycle safeguards for the explicitly budgeted September 13 diagnostic.

The remote timeout stops compute; only verified destruction ends storage billing.
The account-wide credential stays on the operator's machine.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from cleanup_gpu import destroy_and_verify
from experiment import write_json

RUN = "sanity-2026-09-13"
REMOTE = "/workspace/thinking-budget"


def provider_instances():
    # The CLI follows pagination on the current v1 listing endpoint. The v0
    # list URL in the provider's introductory documentation now returns 410.
    result = subprocess.run(["uv", "tool", "run", "--from", "vastai==1.6.0",
                             "vastai", "show", "instances", "--raw"],
                            capture_output=True, text=True, check=True, timeout=60)
    rows = json.loads(result.stdout)
    if not isinstance(rows, list):
        raise ValueError("unexpected instance listing")
    return rows


def api(key, path, method="GET", payload=None):
    request = urllib.request.Request("https://console.vast.ai/api/v0/" + path,
        data=None if payload is None else json.dumps(payload).encode(), method=method,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def artifact_hashes(root):
    result = {}
    for folder in [root / "results" / RUN, root / "checkpoints" / RUN]:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name not in {"artifact_hashes.json", "workflow.log"}:
                result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def backup_verified(root):
    path = root / "results" / RUN / "artifact_hashes.json"
    if not path.exists():
        return False
    expected = json.loads(path.read_text())
    return bool(expected) and all(
        (root / name).is_file() and hashlib.sha256((root / name).read_bytes()).hexdigest() == value
        for name, value in expected.items())


def remote_job():
    root = Path.cwd()
    code = 1
    try:
        code = subprocess.call([sys.executable, "sanity_sft.py", "run"])
    except KeyboardInterrupt:
        code = 130
    finally:
        write_json(root / "results" / RUN / "job_status.json",
                   {"exit_code": code, "completed_at": time.time()})
        write_json(root / "results" / RUN / "artifact_hashes.json", artifact_hashes(root))
    return code


def remote_guard(instance, deadline):
    key = os.environ["CONTAINER_API_KEY"]
    # Validate that the restricted credential is present without exposing it.
    if not key or instance <= 0:
        raise ValueError("missing restricted instance identity")
    print("Remote compute timeout armed", flush=True)
    while True:
        status = Path(REMOTE) / "results" / RUN / "job_status.json"
        finished = json.loads(status.read_text())["completed_at"] if status.exists() else None
        if time.time() >= deadline or (finished and time.time() >= finished + 600):
            try:
                result = api(key, f"instances/{instance}/", "PUT", {"state": "stopped"})
                if not result.get("success"):
                    raise RuntimeError("provider did not acknowledge compute stop")
                print("Compute stop requested; local supervisor must verify destruction", flush=True)
            except Exception as exc:
                print("Stop retry:", type(exc).__name__, flush=True)
        time.sleep(20)


def watch(state_path):
    state = json.loads(state_path.read_text())
    root = Path(state["local_root"])
    key = Path(state["key_path"]).read_text().strip()
    instance = state["instance"]
    started = state["created_at"]
    while True:
        try:
            rows = provider_instances()
            row = next((r for r in rows if int(r["id"]) == instance), None)
            if row is None:
                write_json(state_path.with_suffix(".done.json"),
                           {"instance": instance, "verified_absent_at": time.time(),
                            "backup_verified": backup_verified(root)})
                print("Instance verified absent:", instance, flush=True)
                return
            elapsed = time.time() - started
            cutoff = elapsed >= 80 * 60
            # Bound provisioning failures; no useful job artifacts exist yet.
            if elapsed >= 15 * 60 and not state_path.with_suffix(".ready").exists():
                print("Provisioning deadline reached", flush=True)
                destroy_and_verify(instance)
                continue
            port = (row.get("ports") or {}).get("22/tcp", [])
            host = row.get("public_ipaddr") if port else row.get("ssh_host")
            ssh_port = port[0]["HostPort"] if port else row.get("ssh_port")
            if host and ssh_port and state_path.with_suffix(".ready").exists():
                ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                       "-o", "StrictHostKeyChecking=accept-new", "-i", state["ssh_key"],
                       "-p", str(ssh_port)]
                target = "root@" + host
                if cutoff:
                    subprocess.run(ssh + [target, "tmux send-keys -t tb-sanity C-c"],
                                   capture_output=True, timeout=20)
                for folder in ["results", "checkpoints"]:
                    destination = root / folder / RUN
                    destination.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["rsync", "-az", "--timeout=40", "-e", " ".join(ssh),
                                    target + f":{REMOTE}/{folder}/{RUN}/", str(destination) + "/"],
                                   capture_output=True, timeout=180)
                if backup_verified(root):
                    print("All final artifacts verified locally; destroying", instance, flush=True)
                    destroy_and_verify(instance)
                    continue
            if cutoff:
                # Fail safe when retrieval cannot finish: stop compute, preserve data,
                # and keep attempting cleanup. Stopped storage still has a cost.
                result = api(key, f"instances/{instance}/", "PUT", {"state": "stopped"})
                print("Deadline compute stop:", bool(result.get("success")), flush=True)
        except Exception as exc:
            print("Supervisor retry:", type(exc).__name__, str(exc)[:180], flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["remote-job", "remote-guard", "watch"])
    parser.add_argument("--instance", type=int)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    if args.action == "remote-job":
        raise SystemExit(remote_job())
    if args.action == "remote-guard":
        if not args.instance or not args.deadline:
            parser.error("remote guard requires instance and absolute deadline")
        remote_guard(args.instance, args.deadline)
    else:
        if args.state is None:
            parser.error("watch requires a local state file")
        watch(args.state)


if __name__ == "__main__":
    main()
