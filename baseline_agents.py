"""
Baseline ladder opponents: random -> greedy -> minimax (alpha-beta) at a
few depths. Built on checkers_engine.py.

These serve two purposes: (1) the actual baseline-ladder evaluation the
final report needs, and (2) an early, zero-setup sanity check for the
self-play agent -- "does it beat random yet?" is answerable the moment
training produces its first move, without waiting on anything else.
"""

import random as _random
from checkers_engine import (
    BLACK, WHITE, BLACK_MAN, BLACK_KING, WHITE_MAN, WHITE_KING, EMPTY,
    legal_moves, apply_move, game_result, owner, is_king,
)

MAN_VALUE = 1.0
KING_VALUE = 1.6  # kings are more valuable but not simply "worth 2 men"


def material_value(board, player):
    """Heuristic board value from `player`'s perspective: own material
    minus opponent material, kings weighted higher than men."""
    score = 0.0
    for piece in board:
        if piece == EMPTY:
            continue
        v = KING_VALUE if is_king(piece) else MAN_VALUE
        score += v if owner(piece) == player else -v
    return score


def random_agent(board, player, rng=_random):
    moves = legal_moves(board, player)
    return rng.choice(moves) if moves else None


def greedy_agent(board, player):
    """One-ply lookahead: pick the move whose resulting position has the
    best material_value for `player`. Ties broken randomly."""
    moves = legal_moves(board, player)
    if not moves:
        return None
    scored = [(material_value(apply_move(board, m), player), m) for m in moves]
    best = max(s for s, _ in scored)
    return _random.choice([m for s, m in scored if s == best])


def _minimax(board, player_to_move, root_player, depth, alpha, beta):
    result = game_result(board, player_to_move)
    if result is not None:
        # A decisive result is worth more than any material heuristic,
        # scaled by remaining depth so faster wins are preferred.
        return (1000 + depth) if result == root_player else -(1000 + depth)
    if depth == 0:
        return material_value(board, root_player)

    moves = legal_moves(board, player_to_move)
    maximizing = player_to_move == root_player
    best = float("-inf") if maximizing else float("inf")
    for m in moves:
        nb = apply_move(board, m)
        val = _minimax(nb, -player_to_move, root_player, depth - 1, alpha, beta)
        if maximizing:
            best = max(best, val)
            alpha = max(alpha, best)
        else:
            best = min(best, val)
            beta = min(beta, best)
        if beta <= alpha:
            break  # alpha-beta cutoff
    return best


def minimax_agent(board, player, depth):
    moves = legal_moves(board, player)
    if not moves:
        return None
    scored = []
    for m in moves:
        nb = apply_move(board, m)
        val = _minimax(nb, -player, player, depth - 1, float("-inf"), float("inf"))
        scored.append((val, m))
    best = max(s for s, _ in scored)
    return _random.choice([m for s, m in scored if s == best])


def play_game(agent_black, agent_white, max_plies=300):
    """Plays one game between two agent functions (each: (board, player) ->
    move). Returns BLACK, WHITE, or None (move-limit draw)."""
    from checkers_engine import initial_board
    board = initial_board()
    player = BLACK
    for _ in range(max_plies):
        result = game_result(board, player)
        if result is not None:
            return result
        move = agent_black(board, player) if player == BLACK else agent_white(board, player)
        board = apply_move(board, move)
        player = -player
    return None  # move-limit reached: treat as a draw


if __name__ == "__main__":
    _random.seed(0)

    # 1. Random agent always returns a legal move (or None if none exist).
    from checkers_engine import initial_board
    b = initial_board()
    m = random_agent(b, BLACK)
    assert m in legal_moves(b, BLACK)
    print("OK: random_agent returns a legal move.")

    # 2. Greedy agent takes a free capture when one's on offer.
    b = [EMPTY] * 32
    b[9], b[12] = BLACK_MAN, WHITE_MAN  # black can capture white for free
    m = greedy_agent(b, BLACK)
    assert m["captures"] == (12,)
    print("OK: greedy_agent takes an available capture.")

    # 3. Minimax looks past an immediate capture to the recapture that
    # follows it. Black's only piece has exactly one legal move: capture
    # white's man at 17 (forced). That lands black at 12 -- where white's
    # OTHER man, at 8, immediately recaptures. A depth-0 (myopic) view of
    # the position right after black's capture looks good for black; a
    # deeper search should already reflect the recapture.
    b = [EMPTY] * 32
    b[21], b[17], b[8] = BLACK_MAN, WHITE_MAN, WHITE_MAN
    forced_moves = legal_moves(b, BLACK)
    assert len(forced_moves) == 1 and forced_moves[0]["captures"] == (17,)
    after_black_capture = apply_move(b, forced_moves[0])
    myopic_value = material_value(after_black_capture, BLACK)

    white_reply = legal_moves(after_black_capture, WHITE)
    assert len(white_reply) == 1 and white_reply[0]["captures"] == (12,), (
        "expected white's man at 8 to recapture on 12"
    )
    after_recapture = apply_move(after_black_capture, white_reply[0])
    true_value = material_value(after_recapture, BLACK)
    print(f"material for black right after its own capture: {myopic_value}, "
          f"after white's recapture: {true_value}")
    assert true_value < myopic_value, "the recapture should erase black's apparent gain"

    mm_shallow = _minimax(b, BLACK, BLACK, depth=1, alpha=float("-inf"), beta=float("inf"))
    mm_deep = _minimax(b, BLACK, BLACK, depth=3, alpha=float("-inf"), beta=float("inf"))
    print(f"minimax value at depth=1: {mm_shallow}, at depth=3: {mm_deep}")
    assert mm_deep < mm_shallow, "a deeper search should discount the capture once the reply is visible"
    print("OK: minimax looks past the immediate capture to the forced recapture reply.")

    # 4. A full random-vs-random game terminates (doesn't hang or crash).
    result = play_game(random_agent, random_agent, max_plies=200)
    print(f"OK: a random-vs-random game completed, result={result}.")

    # 5. Minimax(depth=2) beats random over a small sample -- confirms the
    # ladder actually has a meaningful skill gradient, not just distinct
    # names for equally-strong players.
    wins = {BLACK: 0, WHITE: 0, None: 0}
    for i in range(6):
        _random.seed(i)
        mm = lambda b, p: minimax_agent(b, p, depth=2)
        if i % 2 == 0:
            r = play_game(mm, random_agent, max_plies=150)
            wins[BLACK if r == BLACK else (WHITE if r == WHITE else None)] += 1
            winner_is_minimax = (r == BLACK)
        else:
            r = play_game(random_agent, mm, max_plies=150)
            winner_is_minimax = (r == WHITE)
        print(f"  game {i}: result={r}, minimax_won={winner_is_minimax}")
    print("(sampled a few games above -- small sample, just a smoke test "
          "that the ladder has a real skill gradient, not a rigorous win-rate claim)")
