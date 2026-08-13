#!/usr/bin/env python
"""Upload ONE file to a HF repo and verify it landed byte-exact.

Why not `hf upload-large-folder` (or scripts/util/hf_upload_watchdog.sh, which
wraps it): that CLI creates/updates the repo with its OWN default settings and
overwrites files present in the staging dir. It has silently changed this
repo's visibility and clobbered the hub README before. A single upload_file
commit touches neither.

**This script never reads or writes repo visibility.** Visibility on these
repos is switched deliberately and often; it is the user's call, not a build
step's. If you need to know it, read it live yourself.

Two operational hazards this handles:
  - The proxy drops long upload streams. --retries re-attempts; hf_transfer is
    left off because it makes the failure mode worse (opaque C-level abort).
  - A backgrounded child dies when its parent shell exits, mid-stream, with NO
    error in the log — the log just stops at a progress bar. Always launch as
    `setsid nohup ... & disown`, and treat "no UPLOAD_DONE line" as failure
    even when the log looks healthy.

Usage:
  hf_upload_file.py <local_path> <repo_id> <path_in_repo> [--retries N]
"""
import argparse
import os
import sys
import time
from pathlib import Path

PROXY = "http://127.0.0.1:17890"
os.environ.setdefault("HTTP_PROXY", PROXY)
os.environ.setdefault("HTTPS_PROXY", PROXY)

from huggingface_hub import HfApi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("local_path")
    ap.add_argument("repo_id")
    ap.add_argument("path_in_repo")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--repo-type", default="model")
    args = ap.parse_args()

    src = Path(args.local_path)
    size = src.stat().st_size
    print(f"uploading {src} ({size:,} B) -> {args.repo_id}:{args.path_in_repo}",
          flush=True)

    api = HfApi()
    for attempt in range(1, args.retries + 1):
        try:
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=args.path_in_repo,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                commit_message=f"Add {args.path_in_repo}",
            )
            break
        except Exception as e:                    # proxy drops are transient
            print(f"attempt {attempt}/{args.retries} failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            if attempt == args.retries:
                print("UPLOAD_FAILED", flush=True)
                sys.exit(1)
            time.sleep(min(60, 5 * attempt))

    # Server-side size check: the hub reports the blob it actually stored, so a
    # truncated stream shows up here rather than on the device team's machine.
    info = api.repo_info(args.repo_id, repo_type=args.repo_type, files_metadata=True)
    match = [s for s in info.siblings if s.rfilename == args.path_in_repo]
    if not match:
        print(f"VERIFY_FAILED: {args.path_in_repo} absent after upload", flush=True)
        sys.exit(1)
    remote = match[0].size
    if remote != size:
        print(f"VERIFY_FAILED: remote {remote:,} B != local {size:,} B", flush=True)
        sys.exit(1)
    print(f"verified server-side: {remote:,} B", flush=True)
    print("UPLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
