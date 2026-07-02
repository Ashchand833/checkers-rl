"""
Kaggle-ready training entry point. Pulls together checkers_engine,
baseline_agents, td_leaf, and td_train (all independently tested) plus
checkpoint_manager (tested except for the real git push, see its own
module docstring) into one script meant to be pasted into a Kaggle
notebook cell.

Checkpoint-aware from the first line, not added after the fact: every
run first tries to resume from the latest checkpoint, and saves+commits
periodically rather than only at the end -- a session dying mid-run
should cost at most CHECKPOINT_EVERY episodes of progress, not the
whole run.

Nothing in this file is testable end-to-end outside a real Kaggle
session with a real git remote (see checkpoint_manager.py's own
docstring for exactly which piece that is). Everything it calls into
has already been tested in isolation.
"""

import os
import time

from td_leaf import init_weights, FEATURE_NAMES
from td_train import train_episode, play_match, win_rate_vs
from baseline_agents import greedy_agent
from checkpoint_manager import (
    save_checkpoint, load_checkpoint, checkpoint_and_commit, push_checkpoint,
)

# ---------------------------------------------------------------------------
# Config -- the values worth tuning are here, not buried in the loop below.
# ---------------------------------------------------------------------------

REPO_DIR = os.environ.get("CHECKPOINT_REPO_DIR", ".")   # set this to your
                                                          # cloned repo's path
CHECKPOINT_NAME = "td_leaf_checkpoint.json"
SEARCH_DEPTH = 2          # shallow on purpose -- see td_train.py's timing
                           # notes; raise once real training time budget
                           # is known, not before
ALPHA = 0.001              # see td_train.py's docstring: 0.01 diverged
EXPLORATION_EPS = 0.15
EPISODES_PER_BATCH = 50
CHECKPOINT_EVERY_N_BATCHES = 1   # commit after every batch; push less
                                  # often (see PUSH_EVERY_N_CHECKPOINTS)
PUSH_EVERY_N_CHECKPOINTS = 4
EVAL_GAMES = 16


def resume_or_init():
    path = os.path.join(REPO_DIR, CHECKPOINT_NAME)
    if os.path.exists(path):
        weights, episode_count, meta = load_checkpoint(path)
        print(f"Resumed from checkpoint: {episode_count} episodes already trained.")
        return weights, episode_count
    print("No checkpoint found -- starting fresh.")
    return init_weights(), 0


def run(total_episodes, rng):
    weights, episode_count = resume_or_init()
    prev_checkpoint_weights = weights[:]
    checkpoints_since_push = 0
    t_start = time.time()

    while episode_count < total_episodes:
        for _ in range(EPISODES_PER_BATCH):
            train_episode(weights, depth=SEARCH_DEPTH, alpha=ALPHA, rng=rng,
                           exploration_eps=EXPLORATION_EPS)
            episode_count += 1

        elapsed = time.time() - t_start
        vs_greedy = win_rate_vs(weights, SEARCH_DEPTH, greedy_agent, EVAL_GAMES, rng)
        vs_prev = play_match(weights, prev_checkpoint_weights, SEARCH_DEPTH, EVAL_GAMES, rng)
        print(f"[{episode_count} episodes, {elapsed:.0f}s] "
              f"vs_greedy={vs_greedy:.2f}  vs_previous_checkpoint={vs_prev:.2f}  "
              f"max|weight|={max(abs(x) for x in weights):.1f}")
        prev_checkpoint_weights = weights[:]

        save_checkpoint(weights, episode_count, os.path.join(REPO_DIR, CHECKPOINT_NAME),
                         extra_meta={"alpha": ALPHA, "depth": SEARCH_DEPTH, "vs_greedy": vs_greedy})
        try:
            checkpoint_and_commit(weights, episode_count, REPO_DIR, CHECKPOINT_NAME)
            checkpoints_since_push += 1
        except RuntimeError as e:
            print(f"WARNING: local commit failed ({e}). Weights are still saved to disk "
                  f"at {CHECKPOINT_NAME} even though the commit didn't happen -- training "
                  f"continues, but this needs attention before the session ends.")
            continue

        if checkpoints_since_push >= PUSH_EVERY_N_CHECKPOINTS:
            try:
                push_checkpoint(REPO_DIR)
                checkpoints_since_push = 0
                print(f"  pushed to remote at {episode_count} episodes.")
            except RuntimeError as e:
                print(f"WARNING: push failed ({e}). Commits are still local and safe; "
                      f"will retry on the next push interval. If this keeps failing, "
                      f"the remote/credentials likely need attention -- send me the "
                      f"exact error and I'll help from there.")

    return weights, episode_count


if __name__ == "__main__":
    import random as _random
    print(f"Config: depth={SEARCH_DEPTH} alpha={ALPHA} "
          f"episodes_per_batch={EPISODES_PER_BATCH} repo_dir={REPO_DIR}")
    print(f"Features: {FEATURE_NAMES}\n")
    rng = _random.Random(0)
    # A short run as a smoke test that the whole wiring holds together --
    # not a real training session. REPO_DIR defaults to "." so this
    # writes/commits into whatever directory the script runs from; point
    # CHECKPOINT_REPO_DIR at a real cloned repo for actual use.
    import tempfile, subprocess
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=tmp, check=True)
        REPO_DIR = tmp
        weights, episodes = run(total_episodes=100, rng=rng)
        log = subprocess.run(["git", "log", "--oneline"], cwd=tmp, capture_output=True, text=True)
        print(f"\ncommits made during smoke test:\n{log.stdout}")
        assert episodes >= 100
        assert os.path.exists(os.path.join(tmp, CHECKPOINT_NAME))

        # The actual point of all this: simulate a session reset by
        # calling run() again as if from a fresh process, pointed at the
        # same repo dir, and confirm it resumes rather than starting over.
        weights2, episodes2 = run(total_episodes=150, rng=rng)
        assert episodes2 == 150, f"expected to resume to 150, got {episodes2}"
        print(f"\nOK: a fresh run() call against the same repo dir resumed from "
              f"{episodes} and trained to {episodes2}, not restarted from 0 -- "
              f"this is the actual guarantee a Kaggle session reset needs.")
    print("OK: end-to-end wiring (train -> save -> commit) holds together "
          "over a short smoke-test run. Push was never exercised (no real "
          "remote here) -- that's the one step still to confirm on your end.")
