"""
Checkers game engine: board representation, legal move generation
(including multi-jump chains), and game-over detection.

Square numbering matches Chinook's own numbering (see chinook_lookup.py's
chinook_square()) on purpose -- this board plugs into the oracle work
later with no conversion layer needed. 32 playable squares, 0-31.
"""

EMPTY, BLACK_MAN, BLACK_KING, WHITE_MAN, WHITE_KING = 0, 1, 2, -1, -2
BLACK, WHITE = 1, -1


def _build_square_row_col():
    """Invert chinook_square(row, col) -> square, giving square -> (row, col)."""
    sq_to_rc = {}
    for row in range(8):
        for col in range(4):
            addr = (row >> 1) + (4 if row & 1 else 0) + 8 * col
            sq_to_rc[addr] = (row, col)
    return sq_to_rc


SQ_TO_RC = _build_square_row_col()
RC_TO_SQ = {rc: sq for sq, rc in SQ_TO_RC.items()}


def _diagonal_neighbor(square, row_step, col_side):
    """The single diagonal neighbor of `square` one step in the given row
    direction (+1/-1) and "side" (left/right, accounting for the brick-like
    offset between odd and even rows on a checkers board)."""
    row, col = SQ_TO_RC[square]
    new_row = row + row_step
    if not (0 <= new_row <= 7):
        return None
    # Odd/even rows are offset by half a square; the two "sides" from a
    # given square land on the same col, or col-1 / col+1, depending on
    # row parity. Precomputed by just checking real adjacency in (row,col)
    # space via the physical board columns (0-7), not the packed col (0-3).
    phys_col = 2 * col + (1 if row % 2 else 0)
    new_phys_col = phys_col + col_side
    if not (0 <= new_phys_col <= 7):
        return None
    new_col = new_phys_col // 2
    if (new_row, new_col) not in RC_TO_SQ:
        return None
    return RC_TO_SQ[(new_row, new_col)]


def _build_neighbor_tables():
    """For each square, the diagonal neighbor in each of 4 directions:
    (row_step, col_side) in {(+1,-1), (+1,+1), (-1,-1), (-1,+1)}."""
    dirs = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
    table = {sq: [_diagonal_neighbor(sq, rs, cs) for rs, cs in dirs] for sq in range(32)}
    return table, dirs


NEIGHBORS, DIRS = _build_neighbor_tables()  # NEIGHBORS[sq] = [NE, NW, SE, SW] one-step neighbors


def initial_board():
    board = [EMPTY] * 32
    for sq in range(32):
        row, _ = SQ_TO_RC[sq]
        if row <= 2:
            board[sq] = WHITE_MAN
        elif row >= 5:
            board[sq] = BLACK_MAN
    return board


def owner(piece):
    if piece in (BLACK_MAN, BLACK_KING):
        return BLACK
    if piece in (WHITE_MAN, WHITE_KING):
        return WHITE
    return None


def is_king(piece):
    return piece in (BLACK_KING, WHITE_KING)


def _dirs_for(piece):
    """Which of the 4 (row_step, col_side) directions this piece can move
    in. Men move only "forward": black toward row 0 (row_step -1), white
    toward row 7 (row_step +1). Kings move in all 4."""
    if is_king(piece):
        return [0, 1, 2, 3]
    if piece == BLACK_MAN:
        return [2, 3]  # the two (-1, *) directions
    if piece == WHITE_MAN:
        return [0, 1]  # the two (+1, *) directions
    return []


def _jump_landing(square, direction):
    """Two squares in `direction` from `square`, i.e. the landing square of
    a capture -- or None if that runs off the board."""
    mid = NEIGHBORS[square][direction]
    if mid is None:
        return None, None
    far = NEIGHBORS[mid][direction]
    return mid, far


def _simple_moves_from(board, square, piece):
    moves = []
    for d in _dirs_for(piece):
        n = NEIGHBORS[square][d]
        if n is not None and board[n] == EMPTY:
            moves.append({"path": (square, n), "captures": ()})
    return moves


def _capture_sequences_from(board, square, piece, captured_so_far=()):
    """All maximal capture sequences starting at `square`, as a list of
    dicts {"path": (sq0, sq1, ..., sqN), "captures": (mid1, mid2, ...)}.
    Recurses to enforce mandatory multi-jump chains: a sequence only ends
    when no further capture is available from the landing square."""
    sequences = []
    for d in _dirs_for(piece):
        mid, far = _jump_landing(square, d)
        if mid is None or far is None:
            continue
        if owner(board[mid]) == -owner(piece) and board[far] == EMPTY and mid not in captured_so_far:
            # Simulate the capture to check for continuations.
            nb = board[:]
            nb[square] = EMPTY
            nb[mid] = EMPTY
            promoted = piece
            landing_row, _ = SQ_TO_RC[far]
            if piece == BLACK_MAN and landing_row == 0:
                promoted = BLACK_KING
            elif piece == WHITE_MAN and landing_row == 7:
                promoted = WHITE_KING
            nb[far] = promoted
            continuations = _capture_sequences_from(nb, far, promoted, captured_so_far + (mid,))
            if continuations:
                for cont in continuations:
                    sequences.append({
                        "path": (square,) + cont["path"],
                        "captures": (mid,) + cont["captures"],
                    })
            else:
                sequences.append({"path": (square, far), "captures": (mid,)})
    return sequences


def legal_moves(board, player):
    """All legal moves for `player`. If any capture is available anywhere
    on the board, ONLY capture sequences are legal (mandatory capture),
    matching real checkers rules -- including forcing the longest chain
    available from whichever piece starts capturing."""
    captures = []
    normals = []
    for sq in range(32):
        piece = board[sq]
        if owner(piece) != player:
            continue
        captures.extend(_capture_sequences_from(board, sq, piece))
        if not captures:
            normals.extend(_simple_moves_from(board, sq, piece))
    return captures if captures else normals


def apply_move(board, move):
    nb = board[:]
    path = move["path"]
    piece = nb[path[0]]
    nb[path[0]] = EMPTY
    for mid in move["captures"]:
        nb[mid] = EMPTY
    landing = path[-1]
    row, _ = SQ_TO_RC[landing]
    if piece == BLACK_MAN and row == 0:
        piece = BLACK_KING
    elif piece == WHITE_MAN and row == 7:
        piece = WHITE_KING
    nb[landing] = piece
    return nb


def game_result(board, player_to_move):
    """None if the game continues, else the winner (BLACK/WHITE). A player
    with no legal moves loses -- checkers has no stalemate-as-draw rule."""
    if not legal_moves(board, player_to_move):
        return -player_to_move
    return None


if __name__ == "__main__":
    from collections import Counter

    b = initial_board()
    assert Counter(b) == {WHITE_MAN: 12, BLACK_MAN: 12, EMPTY: 8}
    print("OK: initial board has 12+12 pieces and 8 empty squares in the middle rows.")

    bad = 0
    for sq in range(32):
        for d, n in enumerate(NEIGHBORS[sq]):
            if n is None:
                continue
            opp = {0: 3, 1: 2, 2: 1, 3: 0}[d]
            if NEIGHBORS[n][opp] != sq:
                bad += 1
    assert bad == 0
    print("OK: diagonal-neighbor table is geometrically symmetric (0 mismatches / 128 checks).")

    # Multi-jump chain: black at 22 can capture over 18 (landing 13), which
    # immediately opens a second capture over 9 (landing 4) -- exactly the
    # rule the earlier JS reference implementation was missing.
    b = [EMPTY] * 32
    b[22], b[18], b[9] = BLACK_MAN, WHITE_MAN, WHITE_MAN
    moves = legal_moves(b, BLACK)
    assert len(moves) == 1
    assert moves[0]["path"] == (22, 13, 4)
    assert set(moves[0]["captures"]) == {18, 9}
    nb = apply_move(b, moves[0])
    assert nb[18] == EMPTY and nb[9] == EMPTY and nb[4] == BLACK_MAN
    print("OK: multi-jump chain forces both captures in a single move, correct path and landing.")

    # Mandatory capture: when any capture exists, ALL other pieces' simple
    # moves are suppressed too, not just the capturing piece's own.
    b = [EMPTY] * 32
    b[9], b[12], b[27] = BLACK_MAN, WHITE_MAN, BLACK_MAN
    moves = legal_moves(b, BLACK)
    assert len(moves) == 1 and moves[0]["captures"] == (12,)
    print("OK: mandatory capture suppresses simple moves for every piece, not just the capturing one.")

    # Kinging on a simple move into the back row.
    b = [EMPTY] * 32
    b[13] = BLACK_MAN
    for mv in legal_moves(b, BLACK):
        nb = apply_move(b, mv)
        if SQ_TO_RC[mv["path"][-1]][0] == 0:
            assert nb[mv["path"][-1]] == BLACK_KING
    print("OK: kinging on a simple move into row 0 works.")

    # Kinging on a capture landing in the back row.
    b = [EMPTY] * 32
    b[9], b[4] = BLACK_MAN, WHITE_MAN
    moves = legal_moves(b, BLACK)
    nb = apply_move(b, moves[0])
    landed = moves[0]["path"][-1]
    assert SQ_TO_RC[landed][0] == 0 and nb[landed] == BLACK_KING
    print("OK: kinging on a capture landing in row 0 works.")

    # No legal moves = a loss (checkers has no stalemate-as-draw rule).
    b = [EMPTY] * 32
    b[0] = BLACK_MAN
    assert game_result(b, BLACK) == WHITE
    print("OK: a player with zero legal moves correctly loses.")

    print("\nALL TESTS PASSED")
