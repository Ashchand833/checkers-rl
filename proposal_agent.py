"""
Rule-based minimax agent for the SHAP/LIME validation study.

This is THE agent the proposal calls for (Section 4.3): a deterministic
rule-based checkers agent using alpha-beta pruning, evaluating leaf
positions via the hand-designed five-feature evaluation function in
evaluation_features.py. SHAP and LIME will be applied to this agent's
decisions to produce feature-attribution rankings, which will then be
compared against Chinook-derived ground truth.
"""

from checkers_engine import (
    BLACK, WHITE, BLACK_MAN, BLACK_KING, WHITE_MAN, WHITE_KING, EMPTY,
    legal_moves, apply_move, game_result, initial_board,
)
from evaluation_features import evaluate, FEATURE_WEIGHTS

DEFAULT_SEARCH_DEPTH = 4
WIN_VALUE = 1000.0


def _minimax(board, player_to_move, root_player, depth, alpha, beta, weights):
    result = game_result(board, player_to_move)
    if result is not None:
        return (WIN_VALUE + depth) if result == root_player else -(WIN_VALUE + depth)
    if depth == 0:
        return evaluate(board, root_player, weights)

    moves = legal_moves(board, player_to_move)
    maximizing = (player_to_move == root_player)
    if maximizing:
        best = float("-inf")
        for m in moves:
            nb = apply_move(board, m)
            v = _minimax(nb, -player_to_move, root_player, depth - 1, alpha, beta, weights)
            if v > best: best = v
            if best > alpha: alpha = best
            if beta <= alpha: break
        return best
    else:
        best = float("inf")
        for m in moves:
            nb = apply_move(board, m)
            v = _minimax(nb, -player_to_move, root_player, depth - 1, alpha, beta, weights)
            if v < best: best = v
            if best < beta: beta = best
            if beta <= alpha: break
        return best


def proposal_agent(board, player, depth=DEFAULT_SEARCH_DEPTH, weights=None):
    moves = legal_moves(board, player)
    if not moves:
        return None
    w = weights or FEATURE_WEIGHTS
    scored = []
    for m in moves:
        nb = apply_move(board, m)
        v = _minimax(nb, -player, player, depth - 1, float("-inf"), float("inf"), w)
        scored.append((v, m))
    best_value = max(v for v, _ in scored)
    for v, m in scored:
        if v == best_value:
            return m


def searched_value(board, player, depth=DEFAULT_SEARCH_DEPTH, weights=None):
    """The scalar value SHAP and LIME will explain."""
    w = weights or FEATURE_WEIGHTS
    return _minimax(board, player, player, depth, float("-inf"), float("inf"), w)


def play_game(agent_black, agent_white, max_plies=200, verbose=False):
    board = initial_board()
    player = BLACK
    for ply in range(max_plies):
        result = game_result(board, player)
        if result is not None:
            return result
        agent = agent_black if player == BLACK else agent_white
        move = agent(board, player)
        if verbose:
            print(f"ply {ply}: {'BLACK' if player == BLACK else 'WHITE'} plays {move['path']}")
        board = apply_move(board, move)
        player = -player
    return None


if __name__ == "__main__":
    import random

    b = initial_board()
    m = proposal_agent(b, BLACK, depth=2)
    assert m in legal_moves(b, BLACK)
    print("OK: proposal_agent returns a legal move from the initial board.")

    m1 = proposal_agent(b, BLACK, depth=3)
    m2 = proposal_agent(b, BLACK, depth=3)
    m3 = proposal_agent(b, BLACK, depth=3)
    assert m1 == m2 == m3
    print("OK: proposal_agent is deterministic (repeated calls give the same move).")

    b = [EMPTY] * 32
    b[9] = BLACK_MAN
    b[12] = WHITE_MAN
    m = proposal_agent(b, BLACK, depth=2)
    assert m is not None and m["captures"] == (12,)
    print("OK: proposal_agent takes an available capture at depth 2.")

    b = [EMPTY] * 32
    b[21] = BLACK_MAN
    b[17] = WHITE_MAN
    b[8] = WHITE_MAN
    forced = legal_moves(b, BLACK)
    assert len(forced) == 1 and forced[0]["captures"] == (17,)
    m = proposal_agent(b, BLACK, depth=3)
    assert m == forced[0]
    v_shallow = searched_value(b, BLACK, depth=1)
    v_deep = searched_value(b, BLACK, depth=3)
    assert v_deep < v_shallow
    print(f"OK: searched_value depth-1={v_shallow:.2f}, depth-3={v_deep:.2f} -- deeper search sees the recapture.")

    def make_random(seed):
        rng = random.Random(seed)
        def _agent(board, player):
            moves = legal_moves(board, player)
            return rng.choice(moves) if moves else None
        return _agent

    def _agent_d3(board, player):
        return proposal_agent(board, player, depth=3)

    wins = {"agent": 0, "random": 0, "draw": 0}
    for i in range(4):
        rand = make_random(i)
        if i % 2 == 0:
            r = play_game(_agent_d3, rand, max_plies=150)
            if r == BLACK: wins["agent"] += 1
            elif r == WHITE: wins["random"] += 1
            else: wins["draw"] += 1
        else:
            r = play_game(rand, _agent_d3, max_plies=150)
            if r == WHITE: wins["agent"] += 1
            elif r == BLACK: wins["random"] += 1
            else: wins["draw"] += 1
    print(f"OK: 4-game sample vs random: proposal_agent won {wins['agent']}, "
          f"lost {wins['random']}, drew {wins['draw']}.")
    assert wins["agent"] >= 3

    print("\nALL TESTS PASSED")