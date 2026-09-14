"""Destroy one explicitly selected Vast instance and verify provider-side removal."""
import argparse
import json
from pathlib import Path
import subprocess


def destroy_and_verify(instance_id, run=subprocess.run):
    if not isinstance(instance_id, int) or instance_id <= 0:
        raise ValueError("instance ID must be a positive integer")
    cli = ["uv", "tool", "run", "--from", "vastai==1.6.0", "vastai"]
    run(cli + ["destroy", "instance", str(instance_id), "--yes", "--raw"],
        capture_output=True, text=True, check=True, timeout=60)
    response = run(cli + ["show", "instances", "--raw"],
                   capture_output=True, text=True, check=True, timeout=60)
    instances = json.loads(response.stdout)
    if not isinstance(instances, list):
        raise RuntimeError("unexpected provider response; deletion is not verified")
    if any(str(row["id"]) == str(instance_id) for row in instances):
        raise RuntimeError("instance still exists; keep the spending watchdog active and retry cleanup")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance", type=int, required=True)
    ap.add_argument("--backup-dir", required=True,
                    help="previously retrieved artifacts; verify completeness/checksums before invoking")
    args = ap.parse_args()
    backup = Path(args.backup_dir)
    if not backup.is_dir() or not any(backup.iterdir()):
        ap.error("backup directory must exist and contain recovered artifacts")
    # Existence is only a guard against an obviously missing backup, not proof of completeness.
    destroy_and_verify(args.instance)
    print(f"Instance {args.instance} is verified absent. Backups remain in {backup.resolve()}.")


if __name__ == "__main__":
    main()
