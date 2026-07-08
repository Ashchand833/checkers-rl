"""
Five-feature evaluation function for the rule-based minimax agent, matching
the specification in the submitted proposal (Section 4.3):

    1. Piece count      -- own regular pieces minus opponent's
    2. King count       -- own kings minus opponent's
    3. King advancement -- how far own men have advanced toward crowning
    4. Piece safety     -- own pieces NOT under threat minus opponent's not
                           under threat (higher = you have more safe pieces)
    5. Centre control   -- own pieces on the four central rows minus opponent's

Each feature is a scalar computed from BLACK's or WHITE's perspective, and
returned as (own - opponent) so a positive value is good for `player`. This
means the feature vector fed to SHAP and LIME is 5-dimensional -- clean,
proposal-aligned, and small enough that both Kernel-SHAP and LIME can run
efficiently over 1500 positions.
"""

from checkers_engine import (
    BLACK, WHITE, BLACK_MAN, BLACK_KING, WHITE_MAN, WHITE_KING, EMPTY,
    legal_moves, owner, is_king, SQ_TO_RC,
)


CENTRE_SQUARES = frozenset(
    sq for sq, (row, _) in SQ_TO_RC.items() if row in (3, 4)
)

FEATURE_WEIGHTS = {
    "piece_count":      1.0,
    "king_count":       1.6,
    "king_advancement": 0.1,
    "piece_safety":     0.3,
    "centre_control":   0.2,
}
FEATURE_ORDER = ["piece_count", "king_count", "king_advancement",
                 "piece_safety", "centre_control"]


def _own_men(board, player):
    target = BLACK_MAN if player == BLACK else WHITE_MAN
    return sum(1 for p in board if p == target)


def piece_count(board, player):
    """Own regular (non-king) pieces minus opponent's regular pieces."""
    return _own_men(board, player) - _own_men(board, -player)


def _own_kings(board, player):
    target = BLACK_KING if player == BLACK else WHITE_KING
    return sum(1 for p in board if p == target)


def king_count(board, player):
    """Own kings minus opponent's kings."""
    return _own_kings(board, player) - _own_kings(board, -player)


def _advancement_sum(board, player):
    total = 0
    man = BLACK_MAN if player == BLACK else WHITE_MAN
    for sq in range(32):
        if board[sq] == man:
            row, _ = SQ_TO_RC[sq]
            total += (7 - row) if player == BLACK else row
    return total


def king_advancement(board, player):
    """Sum of advancement of own men, minus same for opponent."""
    return _advancement_sum(board, player) - _advancement_sum(board, -player)


def _threatened_pieces(board, defender):
    """Set of squares of `defender`'s pieces that the ATTACKER (their
    opponent) could capture on the attacker's next move."""
    attacker = -defender
    threatened = set()
    for m in legal_moves(board, attacker):
        for captured_sq in m["captures"]:
            if owner(board[captured_sq]) == defender:
                threatened.add(captured_sq)
    return threatened


def _own_piece_count(board, player):
    return sum(1 for p in board if owner(p) == player)


def piece_safety(board, player):
    """(Own pieces not under threat) minus (opponent's pieces not under
    threat). Positive = your position is safer than the opponent's."""
    own_total = _own_piece_count(board, player)
    opp_total = _own_piece_count(board, -player)
    own_threat = len(_threatened_pieces(board, player))
    opp_threat = len(_threatened_pieces(board, -player))
    own_safe = own_total - own_threat
    opp_safe = opp_total - opp_threat
    return own_safe - opp_safe


def _own_pieces_in_centre(board, player):
    return sum(1 for sq in CENTRE_SQUARES if owner(board[sq]) == player)


def centre_control(board, player):
    """Own pieces on central rows minus opponent's."""
    return _own_pieces_in_centre(board, player) - _own_pieces_in_centre(board, -player)


def feature_vector(board, player):
    """Returns a dict {feature_name: value} of the 5 features for `player`."""
    return {
        "piece_count":      piece_count(board, player),
        "king_count":       king_count(board, player),
        "king_advancement": king_advancement(board, player),
        "piece_safety":     piece_safety(board, player),
        "centre_control":   centre_control(board, player),
    }


def evaluate(board, player, weights=None):
    """Weighted linear combination of the 5 features."""
    w = weights or FEATURE_WEIGHTS
    fv = feature_vector(board, player)
    return sum(w[name] * fv[name] for name in FEATURE_ORDER)


if __name__ == "__main__":
    from checkers_engine import initial_board, RC_TO_SQ

    b = initial_board()
    assert piece_count(b, BLACK) == 0
    assert piece_count(b, WHITE) == 0
    print("OK: piece_count on initial board is 0 (symmetric).")

    b = [EMPTY] * 32
    b[9] = BLACK_MAN
    b[10] = BLACK_MAN
    b[4] = WHITE_MAN
    assert piece_count(b, BLACK) == 1
    assert piece_count(b, WHITE) == -1
    print("OK: piece_count is +1 for black when black is up a man.")

    b = [EMPTY] * 32
    b[10] = BLACK_KING
    b[20] = WHITE_KING
    b[21] = WHITE_KING
    assert king_count(b, BLACK) == -1
    assert king_count(b, WHITE) == 1
    print("OK: king_count is -1 for black when white has one more king.")

    b = [EMPTY] * 32
    b[RC_TO_SQ[(7, 0)]] = BLACK_MAN
    b[RC_TO_SQ[(3, 0)]] = BLACK_MAN
    b[RC_TO_SQ[(4, 0)]] = BLACK_KING
    assert king_advancement(b, BLACK) == 4
    print("OK: king_advancement sums correctly and excludes kings.")

    b = [EMPTY] * 32
    b[4] = BLACK_MAN
    b[0] = WHITE_MAN
    white_moves = legal_moves(b, WHITE)
    black_moves = legal_moves(b, BLACK)
    white_captures_black = any(4 in m["captures"] for m in white_moves)
    black_captures_white = any(0 in m["captures"] for m in black_moves)
    assert white_captures_black
    assert not black_captures_white
    assert piece_safety(b, BLACK) == -1
    assert piece_safety(b, WHITE) == 1
    print("OK: piece_safety reflects one-sided threats correctly.")

    b = [EMPTY] * 32
    b[RC_TO_SQ[(3, 0)]] = BLACK_MAN
    b[RC_TO_SQ[(4, 1)]] = BLACK_MAN
    b[RC_TO_SQ[(0, 0)]] = BLACK_MAN
    b[RC_TO_SQ[(3, 2)]] = WHITE_MAN
    assert centre_control(b, BLACK) == 1
    assert centre_control(b, WHITE) == -1
    print("OK: centre_control counts only pieces on rows 3-4.")

    b = initial_board()
    fv = feature_vector(b, BLACK)
    assert set(fv.keys()) == set(FEATURE_ORDER)
    print(f"OK: feature_vector returns all 5 features. Initial board (BLACK): {fv}")

    v = evaluate(b, BLACK)
    assert abs(v) < 1e-9
    print(f"OK: evaluate() on initial board is exactly 0 (symmetric): {v}")

    b = [EMPTY] * 32
    b[9] = BLACK_MAN
    b[10] = BLACK_MAN
    b[4] = WHITE_MAN
    v_black = evaluate(b, BLACK)
    v_white = evaluate(b, WHITE)
    assert v_black > 0
    assert v_white < 0
    assert abs(v_black + v_white) < 1e-9
    print(f"OK: evaluate() up-a-man position: black={v_black:.2f}, white={v_white:.2f}")

    print("\nALL TESTS PASSED")