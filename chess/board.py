import numpy as np
import numpy.typing as npt
import numpy.linalg as la
from enum import Enum
from typing import Optional
from itertools import product
from dataclasses import dataclass

# board size

NUM_FILES = 8
NUM_RANKS = 8

# define player colors

class Color(Enum):
    White = 1
    Black = 2

MAP_FEN_COLOR = {
    'w': Color.White,
    'b': Color.Black,
}
MAP_COLOR_FEN = {color: fen for fen, color in MAP_FEN_COLOR.items()}

# define pieces

class PieceType(Enum):
    King   = 1
    Queen  = 2
    Rook   = 3
    Bishop = 4
    Knight = 5
    Pawn   = 6

@dataclass(frozen=True)
class ColoredPiece:
    color: Color
    piece_type: PieceType

    def __str__(self) -> str:
        return f'{self.color.name} {self.piece_type.name}'

MAP_ASCII_PIECE = {
    'K': ColoredPiece(color=Color.White, piece_type=PieceType.King),
    'Q': ColoredPiece(color=Color.White, piece_type=PieceType.Queen),
    'R': ColoredPiece(color=Color.White, piece_type=PieceType.Rook),
    'B': ColoredPiece(color=Color.White, piece_type=PieceType.Bishop),
    'N': ColoredPiece(color=Color.White, piece_type=PieceType.Knight),
    'P': ColoredPiece(color=Color.White, piece_type=PieceType.Pawn),

    'k': ColoredPiece(color=Color.Black, piece_type=PieceType.King),
    'q': ColoredPiece(color=Color.Black, piece_type=PieceType.Queen),
    'r': ColoredPiece(color=Color.Black, piece_type=PieceType.Rook),
    'b': ColoredPiece(color=Color.Black, piece_type=PieceType.Bishop),
    'n': ColoredPiece(color=Color.Black, piece_type=PieceType.Knight),
    'p': ColoredPiece(color=Color.Black, piece_type=PieceType.Pawn),
}
MAP_PIECE_ASCII = {piece: fen for fen, piece in MAP_ASCII_PIECE.items()}

FEN_STARTING_POSITION = '''
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w
KQkq -
0 1
'''

# define movesets

VEC_PUSH   = np.asarray([1, 0], dtype=np.int_)
VEC_DIAG   = np.asarray([1, 1], dtype=np.int_)
VEC_LSHAPE = np.asarray([2, 1], dtype=np.int_)

MAT_ROT_CCW   = np.asarray([[0, -1], [1, 0]], dtype=np.int_) # rotate a quarter turn counterclockwise
MAT_FLIP_FILE = np.asarray([[1, 0], [0, -1]], dtype=np.int_) # reverse file
MAT_FLIP_RANK = np.asarray([[-1, 0], [0, 1]], dtype=np.int_) # reverse rank

VECS_ROOK   = np.hstack([la.matrix_power(MAT_ROT_CCW, i) @ VEC_PUSH[:, np.newaxis] for i in range(4)])
VECS_BISHOP = np.hstack([la.matrix_power(MAT_ROT_CCW, i) @ VEC_DIAG[:, np.newaxis] for i in range(4)])
VECS_KING   = np.hstack([VECS_ROOK, VECS_BISHOP])
VECS_KNIGHT = np.hstack([
    la.matrix_power(MAT_FLIP_FILE, flip_idx) @
    la.matrix_power(MAT_ROT_CCW, rot_idx) @
    VEC_LSHAPE[:, np.newaxis]
    for rot_idx, flip_idx in product(range(4), range(2))])
VECS_PAWN_CAPTURE = np.hstack([la.matrix_power(MAT_FLIP_FILE, i) @ VEC_DIAG[:, np.newaxis] for i in range(2)])

@dataclass(frozen=True)
class Moveset:
    directions: npt.NDArray
    max_steps: Optional[int]

MAP_PIECE_TYPE_MOVESET = {
    PieceType.Queen:  Moveset(VECS_KING,   None),
    PieceType.Rook:   Moveset(VECS_ROOK,   None),
    PieceType.Bishop: Moveset(VECS_BISHOP, None),
    PieceType.Knight: Moveset(VECS_KNIGHT, 1),
}

MOVEEST_KING_WALK = Moveset(VECS_KING,   1),

MOVESET_PAWN_PUSH    = Moveset(VEC_PUSH[:, np.newaxis], 1)
MOVESET_PAWN_DOUBLE  = Moveset(VEC_PUSH[:, np.newaxis], 2)
MOVESET_PAWN_CAPTURE = Moveset(VECS_PAWN_CAPTURE, 1)

MAP_COLOR_ORIENTATION = {
    Color.White: np.eye(2, dtype=np.int_),
    Color.Black: MAT_FLIP_RANK,
}

MAP_COLOR_PAWN_START_REGION = {
    Color.White: np.asarray([[1, i] for i in range(NUM_FILES)], dtype=np.int_).T,
    Color.Black: np.asarray([[NUM_RANKS - 2, i] for i in range(NUM_FILES)], dtype=np.int_).T,
}

# define board

Coordinate = tuple[int, int] | list[int] | npt.NDArray[np.int_] | str
def _coordinate(coordinate: Coordinate) -> npt.NDArray[np.int_]:
    if isinstance(coordinate, str):
        rank = ord(coordinate[1]) - ord('1')
        file = ord(coordinate[0].lower()) - ord('a')
        coordinate = [rank, file]
    return np.asarray(coordinate, dtype=np.int_)

@dataclass
class Board:
    array: list[list[Optional[ColoredPiece]]]
    active_color: Color

    @classmethod
    def _inbounds(cls, position: npt.NDArray[np.int_]) -> bool:
        rank, file = position
        return 0 <= rank and rank < NUM_RANKS and\
               0 <= file and file < NUM_FILES

    def __getitem__(self, position: Coordinate) -> Optional[ColoredPiece]:
        rank, file = _coordinate(position)
        return self.array[rank][file]

    def __setitem__(self, position: Coordinate, piece: Optional[ColoredPiece]) -> Optional[ColoredPiece]:
        rank, file = _coordinate(position)
        self.array[rank][file] = piece
        return piece

    @classmethod
    def fromfen(cls, fen: str) -> 'Board':
        piece_placement, active_color, castling_rights, en_passent_target, halfmove_clock, fullmove_clock = fen.split()

        board = Board(
            array=[[None for _ in range(NUM_FILES)] for _ in range(NUM_RANKS)],
            active_color=MAP_FEN_COLOR[active_color],
        )

        for rank, line in enumerate(piece_placement.split('/')[::-1]):
            file = 0
            for char in line:
                if char.isdigit():
                    file += int(char)
                    continue
                board[rank, file] = MAP_ASCII_PIECE[char]
                file += 1

        return board

    @classmethod
    def default(cls) -> 'Board':
        return Board.fromfen(FEN_STARTING_POSITION)

    def tofen(self) -> str:
        fen = ''
        for rank in reversed(range(NUM_RANKS)):
            skip = 0
            for file in range(NUM_FILES):
                if self[rank, file] is None:
                    skip += 1
                    continue

                if skip > 0:
                    fen += str(skip)
                    skip = 0

                fen += MAP_PIECE_ASCII[self[rank, file]]

            if skip > 0:
                fen += str(skip)
                skip = 0

            if rank > 0:
                fen += '/'

        fen += f' {MAP_COLOR_FEN[self.active_color]}'
        return fen

    def ascii(self) -> str:
        text = ''

        ranks = range(NUM_RANKS)[::-1]
        files = range(NUM_FILES)

        for rank in ranks:
            text += '   ' + ('+---' * NUM_FILES) + '+\n'
            text += f' {rank+1} '
            for file in files:
                if self[rank, file] is None:
                    text += '|   '
                    continue
                text += f'| {MAP_PIECE_ASCII[self[rank, file]]} '
            text += '|\n'
        text += '   ' + ('+---' * NUM_FILES) + '+\n'
        text += '     ' + '   '.join(map(lambda file: chr(ord('a') + file), files))
        return text

    def __repr__(self) -> str:
        text = self.ascii()
        text += f'\n\n   {self.active_color.name} to play'
        return text
    
    def _illegal_moves_from_moveset_at(self, orig: npt.NDArray, color: Color, moveset: Moveset, capture: Optional[bool]) -> list[npt.NDArray]:
        orientation = MAP_COLOR_ORIENTATION[color]
        directions = orientation @ moveset.directions
        moves = []
        for direction_idx in range(directions.shape[1]):
            direction = directions[:, direction_idx]
            steps = 1
            dest = orig + direction
            while (moveset.max_steps is None or steps <= moveset.max_steps):
                # Hit the edge
                if (not self._inbounds(dest)):
                    break

                # Hit own piece
                if self[dest] is not None and self[dest].color == color:
                    break

                move = np.hstack([orig[:, np.newaxis], dest[:, np.newaxis]])

                # Move to empty
                if (self[dest] is None and not capture):
                    moves.append(move)

                # Capture and stop
                if (self[dest] is not None and self[dest].color != color and (capture is None or capture is True)):
                    moves.append(move)                    
                    break

                dest += direction
                steps += 1
        return moves

    def _illegal_moves_at(self, orig: npt.NDArray) -> list[npt.NDArray]:
        # Illegal moves will propose moves that are not blocked by own pieces or the edge of the board
        if self[orig] is None:
            return []
        
        piece = self[orig]

        match piece.piece_type:
            case PieceType.Pawn:
                moves = []
                start_region = MAP_COLOR_PAWN_START_REGION[piece.color]
                in_start_region = (orig[:, np.newaxis] == start_region).all(axis=1).any()
                if in_start_region:
                    moves += self._illegal_moves_from_moveset_at(orig, piece.color, MOVESET_PAWN_DOUBLE, capture=None)
                else:
                    moves += self._illegal_moves_from_moveset_at(orig, piece.color, MOVESET_PAWN_PUSH, capture=None)
                moves += self._illegal_moves_from_moveset_at(orig, piece.color, MOVESET_PAWN_CAPTURE, capture=True)
                # TODO en-passant
                return moves
            case PieceType.King:
                moves = self._illegal_moves_from_moveset_at(orig, piece.color, MOVEEST_KING_WALK, capture=None)
                # TODO castling
                return moves
            case _:
                moveset = MAP_PIECE_TYPE_MOVESET[piece.piece_type]
                return self._illegal_moves_from_moveset_at(orig, piece.color, moveset, capture=None)
    
    def legal_moves_at(self, position: Coordinate):
        return self._illegal_moves_at(_coordinate(position))