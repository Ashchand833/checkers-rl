"""
TD-leaf value function + leaf-tracking alpha-beta search.

The value function is a linear model over hand-picked features -- matching
the historical TD-leaf/KnightCap approach (Baxter, Tridgell & Weaver 2000)
rather than a neural net, which is a deliberate simplicity choice given
the project timeline. Features and weights are cleanly separated from the
search, so swapping in a small MLP later is a contained change, not a
rewrite.

Values are always expressed from BLACK's perspective (positive = good for
black), with the search maximizing at black-to-move nodes and minimizing
at white. This fixed-perspective convention avoids sign-flip bugs that
"value from the mover's perspective" invites in a two-player minimax.
"""

from checkers_engine import (
    BLACK, WHITE, BLACK_MAN, BLACK_KING, WHITE_MAN, WHITE_KING, EMPTY,
    legal_moves, apply_move, game_result, owner, is_king, SQ_TO_RC,
)

FEATURE_NAMES = [
    "bias",
    "black_men", "black_kings", "white_men", "white_kings",
    "black_mobility", "white_mobility",
    "black_back_row", "white_back_row",
    "black_advancement", "white_advancement",
]
NUM_FEATURES = len(FEATURE_NAMES)

WIN_VALUE = 100.0  # terminal reward scale -- deliberately large relative
                    # to typical mid-game feature values, so the model
                    # learns to treat "certain win" as dominant, matching
                    # the game-outcome-first ordering baseline_agents.py's
                    # minimax already enforces via its own +/-(1000+depth).


def features(board):
    """Feature vector for `board`, always from black's perspective."""
    f = [0.0] * NUM_FEATURES
    f[0] = 1.0  # bias
    black_moves = len(legal_moves(board, BLACK))
    white_moves = len(legal_moves(board, WHITE))
    f[5], f[6] = black_moves, white_moves
    for sq in range(32):
        piece = board[sq]
        if piece == EMPTY:
            continue
        row, _ = SQ_TO_RC[sq]
        if piece == BLACK_MAN:
            f[1] += 1
            f[9] += (7 - row)  # progress toward row 0 (black's crowning row)
            if row == 7:
                f[7] += 1  # holding the back row (defensive structure)
        elif piece == BLACK_KING:
            f[2] += 1
        elif piece == WHITE_MAN:
            f[3] += 1
            f[10] += row  # progress toward row 7 (white's crowning row)
            if row == 0:
                f[8] += 1
        elif piece == WHITE_KING:
            f[4] += 1
    return f


def value(board, weights):
    return sum(w * x for w, x in zip(weights, features(board)))


def init_weights():
    """A reasonable starting point: material dominates (kings worth more
    than men, matching baseline_agents.py's own KING_VALUE), everything
    else starts near zero and gets shaped by training."""
    w = [0.0] * NUM_FEATURES
    w[1], w[2], w[3], w[4] = 1.0, 1.6, -1.0, -1.6
    return w


def leaf_search(board, player_to_move, weights, depth):
    """Alpha-beta search maximizing at black nodes / minimizing at white,
    evaluating leaves with `value()`. Returns (best_value, best_move,
    leaf_board) -- leaf_board is the actual position whose features
    produced the backed-up value, needed by the TD update. best_move is
    None at depth 0 or on a terminal node (nothing further to choose)."""
    result = game_result(board, player_to_move)
    if result is not None:
        return (WIN_VALUE if result == BLACK else -WIN_VALUE), None, board
    if depth == 0:
        return value(board, weights), None, board

    moves = legal_moves(board, player_to_move)
    maximizing = player_to_move == BLACK
    best_value = float("-inf") if maximizing else float("inf")
    best_move, best_leaf = None, None
    alpha, beta = float("-inf"), float("inf")
    for m in moves:
        nb = apply_move(board, m)
        val, _, leaf = leaf_search(nb, -player_to_move, weights, depth - 1)
        better = (val > best_value) if maximizing else (val < best_value)
        if better:
            best_value, best_move, best_leaf = val, m, leaf
        if maximizing:
            alpha = max(alpha, best_value)
        else:
            beta = min(beta, best_value)
        if beta <= alpha:
            break
    return best_value, best_move, best_leaf


if __name__ == "__main__":
    from checkers_engine import initial_board

    b = initial_board()
    f = features(b)
    print("initial-board features:", dict(zip(FEATURE_NAMES, f)))
    assert f[1] == 12 and f[3] == 12 and f[2] == 0 and f[4] == 0
    assert f[5] == f[6], "the initial position is symmetric -- mobility should match"
    print("OK: initial-position features are sane (12/12 men, symmetric mobility).")

    w = init_weights()
    v = value(b, w)
    print(f"initial-position value with starting weights: {v} (should be ~0, it's symmetric)")
    assert abs(v) < 1e-9
    print("OK: symmetric starting position values to exactly 0 under material-only weights.")

    # A position up a man for black should value clearly positive.
    b2 = [EMPTY] * 32
    b2[9], b2[12] = BLACK_MAN, BLACK_MAN
    b2[4] = WHITE_MAN
    v2 = value(b2, w)
    print(f"black up a man (2 vs 1): value = {v2} (should be > 0)")
    assert v2 > 0
    print("OK: material imbalance is reflected with the correct sign.")

    # leaf_search should find the same forced-recapture dynamic the
    # baseline minimax test found -- a good cross-check between the two
    # search implementations.
    b3 = [EMPTY] * 32
    b3[21], b3[17], b3[8] = BLACK_MAN, WHITE_MAN, WHITE_MAN
    val1, move1, leaf1 = leaf_search(b3, BLACK, w, depth=1)
    val3, move3, leaf3 = leaf_search(b3, BLACK, w, depth=3)
    print(f"leaf_search depth=1 value: {val1}, depth=3 value: {val3}")
    assert val3 < val1, "depth-3 should see the forced recapture that depth-1 misses"
    print("OK: leaf_search also looks past the immediate capture to the recapture, "
          "matching baseline_agents.py's minimax behaviour.")
