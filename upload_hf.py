"""Push the adapter, eval results and plots to a HF model repo.

Token comes from HF_TOKEN in the environment - never hardcode it.

  python upload_hf.py --adapter checkpoints/lcpo-max/checkpoint-1000 \
      --repo rk9595/thinking-budget-qwen1.5b-lcpo
"""
import argparse
import glob
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--repo", default="rk9595/thinking-budget-qwen1.5b-lcpo")
    ap.add_argument("--results", default="results")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)

    if os.path.isdir(args.adapter):
        api.upload_folder(folder_path=args.adapter, path_in_repo="adapter",
                          repo_id=args.repo, repo_type="model")
        print("uploaded adapter from", args.adapter)

    for pattern in ("*.json", "*.png", "*.csv"):
        for f in glob.glob(os.path.join(args.results, pattern)):
            api.upload_file(path_or_fileobj=f, path_in_repo=f"results/{os.path.basename(f)}",
                            repo_id=args.repo, repo_type="model")
            print("uploaded", f)

    if os.path.exists("MODEL_CARD.md"):
        api.upload_file(path_or_fileobj="MODEL_CARD.md", path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model")
        print("uploaded model card")

    print(f"\nhttps://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
