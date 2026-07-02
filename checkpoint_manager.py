"""
Checkpoint persistence for training runs on Kaggle, where sessions reset
and can wipe anything not saved outside the session.

Local save/restore and local git mechanics (init, add, commit) are fully
testable and tested below. The actual `git push` to a real GitHub remote
is NOT tested here -- it needs real, authenticated credentials that only
exist in your own Kaggle/GitHub setup, not in this sandbox. Everything up
to that point is verified; that one step is code-complete but unverified
until it runs in your environment.
"""

import json
import os
import subprocess
import time


def save_checkpoint(weights, episode_count, path, extra_meta=None):
    """Writes weights + training metadata as one JSON file."""
    from td_leaf import FEATURE_NAMES
    payload = {
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "episode_count": episode_count,
        "saved_at": time.time(),
        "meta": extra_meta or {},
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)  # atomic on POSIX -- avoids a half-written
    # checkpoint if the process dies mid-save, which matters a lot more
    # on a platform that can kill your session without warning.


def load_checkpoint(path):
    from td_leaf import FEATURE_NAMES
    with open(path) as f:
        payload = json.load(f)
    assert payload["feature_names"] == FEATURE_NAMES, "feature set changed -- checkpoint is stale"
    return payload["weights"], payload["episode_count"], payload["meta"]


def _run_git(args, cwd, check=True):
    result = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def commit_checkpoint(repo_dir, checkpoint_path, message):
    """Stages and commits the checkpoint file within an EXISTING git repo
    at `repo_dir`. Does not push -- see push_checkpoint() and the
    module-level note about why that part is untested here."""
    rel_path = os.path.relpath(checkpoint_path, repo_dir)
    _run_git(["add", rel_path], cwd=repo_dir)
    result = _run_git(["commit", "-m", message], cwd=repo_dir, check=False)
    if result.returncode != 0 and "nothing to commit" not in result.stdout:
        raise RuntimeError(f"git commit failed: {result.stdout}\n{result.stderr}")
    return result.returncode == 0


def push_checkpoint(repo_dir, remote="origin", branch=None):
    """Pushes committed checkpoints to the remote. NOT exercised by the
    tests below -- needs a real, authenticated remote, which only exists
    in your own environment. Wrapped so a failed push (e.g. no
    credentials configured yet) raises a clear error rather than hanging
    or failing silently mid-training."""
    args = ["push", remote] + ([branch] if branch else [])
    result = _run_git(args, cwd=repo_dir, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"git push failed -- check that {remote} is configured with "
            f"push credentials in this environment: {result.stderr.strip()}"
        )


def checkpoint_and_commit(weights, episode_count, repo_dir, checkpoint_name="checkpoint.json"):
    """Convenience wrapper: save, then stage+commit within repo_dir. Call
    push_checkpoint() separately (e.g. every N calls, not every one) to
    avoid hammering the remote."""
    path = os.path.join(repo_dir, checkpoint_name)
    save_checkpoint(weights, episode_count, path)
    return commit_checkpoint(repo_dir, path, f"checkpoint: episode {episode_count}")


if __name__ == "__main__":
    import tempfile
    from td_leaf import init_weights

    # 1. Local save/load round-trips, including the metadata.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.json")
        w = init_weights()
        save_checkpoint(w, episode_count=123, path=path, extra_meta={"alpha": 0.001})
        w2, ep, meta = load_checkpoint(path)
        assert w2 == w and ep == 123 and meta == {"alpha": 0.001}
        print("OK: local save/load round-trips weights, episode count, and metadata.")

        # 2. Atomic write: a checkpoint file is never observed half-written
        # (simulated by checking the .tmp file is cleaned up after save).
        assert not os.path.exists(path + ".tmp")
        print("OK: no leftover .tmp file after a successful save (atomic replace worked).")

    # 3. Local git mechanics: init a throwaway repo, save a checkpoint
    # into it, commit it, verify it's actually in the commit. This is as
    # far as this sandbox can verify -- no real remote to push to here.
    with tempfile.TemporaryDirectory() as repo:
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)

        w = init_weights()
        committed = checkpoint_and_commit(w, episode_count=50, repo_dir=repo)
        assert committed
        log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True)
        print("git log after checkpoint commit:", log.stdout.strip())
        assert "episode 50" in log.stdout

        # A later checkpoint (different episode count -- the realistic
        # case, since save_checkpoint always stamps a fresh timestamp) 
        # produces its own commit on top.
        committed_again = checkpoint_and_commit(w, episode_count=100, repo_dir=repo)
        assert committed_again is True
        log2 = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True)
        print("git log after second checkpoint:", log2.stdout.strip().replace(chr(10), " | "))
        assert log2.stdout.count("checkpoint:") == 2
        print("OK: local git init/add/commit works, and successive checkpoints "
              "each produce their own commit.")

    print(
        "\nNOT tested here: push_checkpoint() against a real remote. "
        "To use this on Kaggle: git-init (or clone) your repo into the "
        "notebook's working directory once, with push credentials "
        "configured (a token-based remote URL is the usual approach on "
        "Kaggle, since there's no interactive auth prompt), then call "
        "checkpoint_and_commit() periodically during training and "
        "push_checkpoint() every few checkpoints. Flag it back to me if "
        "the push step errors -- I can't reproduce that failure mode here "
        "without real credentials, so I'll need the actual error message."
    )
