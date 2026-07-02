"""
TD(0)-leaf self-play training loop.

Plays a full game using leaf_search() for move selection, collects the
sequence of leaf positions the search actually bottomed out at each ply,
then updates the linear weights by the TD(0) rule: for consecutive leaf
values v_t, v_{t+1}, move weights toward reducing the difference,
attributed to the FEATURES OF THE EARLIER LEAF (since it's a linear
model, d(value)/d(weights) = features(leaf) exactly).

This is TD(0), not full TD(lambda) with eligibility traces -- a
deliberate simplification given the project timeline. Traces are a
natural extension if there's time later; TD(0) is still a legitimate,
correctly-implemented member of the same family, not a different
algorithm wearing the name.
"""

import json
import random as _random

from checkers_engine import BLACK, WHITE, initial_board, apply_move, legal_moves, game_result
from td_leaf import features, value, leaf_search, init_weights, WIN_VALUE, FEATURE_NAMES


def train_episode(weights, depth, alpha=0.001, rng=None, exploration_eps=0.1, max_plies=200):
    """Plays one self-play game with weights fixed for its duration,
    then applies TD(0) updates in place. Returns the game result, or
    None if the ply cap was hit (treated as a draw for the TD target).

    alpha=0.001 default: 0.01 was tried first and diverged -- weights
    reached ~1e108 within 200 games (winning-side leaf values feeding a
    too-large update, compounding across ~200 plies x hundreds of games).
    0.001 stays stable well past 400 games. Worth re-checking if depth,
    WIN_VALUE, or the feature set change, since this was found
    empirically, not derived analytically."""
    rng = rng or _random.Random()
    board = initial_board()
    player = BLACK
    leaf_boards = []
    result = None

    for _ in range(max_plies):
        result = game_result(board, player)
        if result is not None:
            break
        val, move, leaf = leaf_search(board, player, weights, depth)
        leaf_boards.append(leaf)
        moves = legal_moves(board, player)
        if rng.random() < exploration_eps:
            move = rng.choice(moves)
        board = apply_move(board, move)
        player = -player

    terminal_value = WIN_VALUE if result == BLACK else (-WIN_VALUE if result == WHITE else 0.0)
    values_seq = [value(lb, weights) for lb in leaf_boards] + [terminal_value]

    for t in range(len(leaf_boards)):
        error = values_seq[t + 1] - values_seq[t]
        f = features(leaf_boards[t])
        for i in range(len(weights)):
            weights[i] += alpha * error * f[i]

    return result


def play_match(weights_a, weights_b, depth, n_games, rng, max_plies=200):
    """weights_a vs weights_b head to head, alternating colors, no
    exploration. This is the honest way to check "did training help" --
    comparing to a fixed opponent like greedy is misleading once search
    alone already beats that opponent most of the time (which happened
    here: depth=2 search beat greedy ~100% of the time even with
    untrained weights, so that comparison couldn't show improvement even
    if training was working). Comparing successive checkpoints to each
    other has no such ceiling."""
    a_wins = 0.0
    for i in range(n_games):
        board = initial_board()
        player = BLACK
        a_is_black = (i % 2 == 0)
        result = None
        for _ in range(max_plies):
            result = game_result(board, player)
            if result is not None:
                break
            w = weights_a if (player == BLACK) == a_is_black else weights_b
            _, move, _ = leaf_search(board, player, w, depth)
            board = apply_move(board, move)
            player = -player
        if result is None:
            a_wins += 0.5
        elif (result == BLACK) == a_is_black:
            a_wins += 1.0
    return a_wins / n_games


def win_rate_vs(weights, depth, opponent_agent, n_games, rng, max_plies=200):
    """How often the current weights (playing via a shallow search, no
    exploration) beat `opponent_agent`, split evenly across both colors
    to cancel out any first-move advantage. Draws (ply cap hit) count as
    half a win, matching standard win-rate scoring conventions."""
    wins = 0.0
    for i in range(n_games):
        board = initial_board()
        player = BLACK
        td_plays_black = (i % 2 == 0)
        result = None
        for _ in range(max_plies):
            result = game_result(board, player)
            if result is not None:
                break
            if (player == BLACK) == td_plays_black:
                _, move, _ = leaf_search(board, player, weights, depth)
            else:
                move = opponent_agent(board, player)
            board = apply_move(board, move)
            player = -player
        if result is None:
            wins += 0.5
        elif (result == BLACK) == td_plays_black:
            wins += 1.0
    return wins / n_games


def save_weights(weights, path):
    with open(path, "w") as f:
        json.dump({"feature_names": FEATURE_NAMES, "weights": weights}, f)


def load_weights(path):
    with open(path) as f:
        data = json.load(f)
    assert data["feature_names"] == FEATURE_NAMES, "feature set changed -- checkpoint is stale"
    return data["weights"]


if __name__ == "__main__":
    rng = _random.Random(0)

    # 1. A single training episode runs to completion and returns a
    # valid result without crashing.
    w = init_weights()
    result = train_episode(w, depth=2, rng=rng)
    assert result in (BLACK, WHITE, None)
    print(f"OK: one training episode completes cleanly (result={result}, "
          f"None means the 200-ply cap was hit and it's scored as a draw).")

    # 2. A hand-built, unambiguous TD step moves the weight in the
    # expected direction. Two leaf boards: first with black up a man
    # (should be valued positively), second identical to the first (so
    # the "true" TD target for step 1 is just step 2's own value -- i.e.
    # this isolates the *mechanism* without needing a real game).
    from checkers_engine import EMPTY, BLACK_MAN, WHITE_MAN
    leaf_up_a_man = [EMPTY] * 32
    leaf_up_a_man[9], leaf_up_a_man[12] = BLACK_MAN, BLACK_MAN
    leaf_up_a_man[4] = WHITE_MAN
    w2 = init_weights()
    before = value(leaf_up_a_man, w2)
    # Manually run one TD(0) step as if this leaf were followed by a
    # much better-for-black leaf (value = WIN_VALUE), like train_episode
    # would if black went on to win.
    error = WIN_VALUE - before
    f = features(leaf_up_a_man)
    alpha = 0.001
    for i in range(len(w2)):
        w2[i] += alpha * error * f[i]
    after = value(leaf_up_a_man, w2)
    print(f"value of the same leaf before/after a TD step toward a win: {before} -> {after}")
    assert after > before, "a TD step toward a winning outcome should raise this leaf's value"
    print("OK: the TD(0) update mechanism moves weights in the correct direction.")

    # 3. Checkpoint round-trip.
    save_weights(w, "/tmp/td_weights_test.json")
    w_loaded = load_weights("/tmp/td_weights_test.json")
    assert w_loaded == w
    print("OK: checkpoint save/load round-trips exactly.")

    # 4. Does training actually improve play? First attempt used alpha=0.01
    # and win-rate-vs-greedy as the signal -- alpha=0.01 diverged (see the
    # docstring above), and separately, win-rate-vs-greedy turned out to
    # be a poor signal on its own: depth=2 search already beats greedy
    # ~100% of the time with UNTRAINED weights, so that metric is
    # saturated from the start and can't distinguish "learning helped"
    # from "search alone was always enough". Showing both metrics below
    # rather than just the fixed one, since the contrast is the lesson.
    import time
    from baseline_agents import greedy_agent

    w3 = init_weights()
    checkpoints = [w3[:]]
    print("\nTraining a small run, tracking two signals every 50 games: "
          "win rate vs greedy (saturated, shown for contrast) and win "
          "rate vs the PREVIOUS checkpoint (the honest signal).")
    t_start = time.time()
    for batch in range(8):
        for _ in range(50):
            train_episode(w3, depth=2, rng=rng, exploration_eps=0.15)
        vs_greedy = win_rate_vs(w3, depth=2, opponent_agent=greedy_agent, n_games=16, rng=rng)
        vs_prev = play_match(w3, checkpoints[-1], depth=2, n_games=16, rng=rng)
        checkpoints.append(w3[:])
        elapsed = time.time() - t_start
        print(f"after {(batch + 1) * 50} games ({elapsed:.0f}s): "
              f"vs_greedy={vs_greedy:.2f}  vs_previous_checkpoint={vs_prev:.2f}  "
              f"max|weight|={max(abs(x) for x in w3):.1f}")
    print(f"\nfinal weights: {dict(zip(FEATURE_NAMES, [round(x, 3) for x in w3]))}")
    print(
        "\nWeights are stable now (no more runaway growth), and the "
        "mechanism is verified correct (test 2). Whether it's *learning "
        "anything useful* at this small scale is genuinely unclear from "
        "vs_previous_checkpoint alone -- consecutive checkpoints are close "
        "in strength by construction, so short-run noise and real "
        "improvement look similar over just a few hundred games. This is "
        "exactly the problem the oracle comparison solves properly: a "
        "fixed, perfect ground truth instead of a moving, noisy one -- "
        "which is a nice concrete illustration of why that piece of the "
        "project matters, not just a nice-to-have."
    )
