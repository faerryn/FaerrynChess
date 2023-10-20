import numpy as np
import numpy.typing as npt
import numpy.linalg as la
from enum import IntEnum
from typing import Optional
import itertools
from dataclasses import dataclass
from typing import Iterable

# define player colors

class Color(IntEnum):
    White = 1
    Black = 2

MAP_FEN_COLOR = {
    'w': Color.White,
    'b': Color.Black,
}
MAP_COLOR_FEN = {color: fen for fen, color in MAP_FEN_COLOR.items()}

# define pieces

class PieceType(IntEnum):
    King   = 1
    Queen  = 2
    Rook   = 3
    Bishop = 4
    Knight = 5
    Pawn   = 6

# board size

NUM_FILES = 8
NUM_RANKS = 8

@dataclass(frozen=True)
class ColoredPiece:
    color: Color
    piece_type: PieceType

    @classmethod
    def encode(cls, piece: Optional['ColoredPiece']) -> int:
        return 0 if piece is None else 1 + (piece.color - 1) * len(PieceType) + (piece.piece_type - 1)

    @classmethod
    def decode(cls, encoded: int) -> Optional['ColoredPiece']:
        if encoded == 0:
            return None
        color = Color(1 + (encoded - 1) // len(PieceType))
        piece_type = PieceType(1 + (encoded - 1) % len(PieceType))
        return ColoredPiece(color, piece_type)

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

@dataclass(frozen=True)
class UnverifiedMove:
    orig: npt.NDArray
    dest: npt.NDArray
    capture: bool

# define board

@dataclass
class Board:
    npboard: npt.NDArray
    active_color: Color

    @classmethod
    def _inbounds(cls, position: npt.ArrayLike) -> bool:
        position = np.asarray(position)
        rank = position[0]
        file = position[1]
        return 0 <= rank and rank < NUM_RANKS and\
               0 <= file and file < NUM_FILES
    
    @classmethod
    def _position(cls, position: npt.ArrayLike | str) -> npt.ArrayLike:
        if isinstance(position, str):
            position = position.lower()
            file = ord(position[0]) - ord('a')
            rank = ord(position[1]) - ord('0')
            return [rank, file]
        assert Board._inbounds(position), f'{position} is out of bounds!'
        return position

    def __getitem__(self, position: npt.ArrayLike | str) -> Optional[ColoredPiece]:
        return ColoredPiece.decode(self.npboard[Board._position(position)])

    def __setitem__(self, position: npt.ArrayLike, piece: Optional[ColoredPiece]) -> Optional[ColoredPiece]:
        self.npboard[Board._position(position)] = ColoredPiece.encode(piece)
        return piece

    @classmethod
    def fromfen(cls, fen: str) -> 'Board':
        piece_placement, active_color, castling_rights, en_passent_target, halfmove_clock, fullmove_clock = fen.split()

        board = Board(
            npboard=np.zeros((NUM_RANKS, NUM_FILES)),
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
