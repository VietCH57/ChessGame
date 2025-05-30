import torch
import numpy as np
from chess_board import Position, PieceType, PieceColor

# Board representation constants
BOARD_SIZE = 8
# Smaller action space representation - each move is encoded as:
# from_square (64 options) x to_square (64 options)
# This gives 64*64 = 4096 possible actions (much more manageable than the 8x8x73=4672 we had before)
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE  # 4096 possible moves (64 squares to 64 squares)

def board_to_input(board, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Convert the chess board state to a format suitable for the neural network
    """
    state = np.zeros((19, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    # Piece planes (12 planes: 6 piece types x 2 colors)
    piece_idx = 0
    for color in [PieceColor.WHITE, PieceColor.BLACK]:
        for piece_type in [PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP, 
                          PieceType.ROOK, PieceType.QUEEN, PieceType.KING]:
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    piece = board.get_piece(Position(row, col))
                    if piece.type == piece_type and piece.color == color:
                        state[piece_idx, row, col] = 1.0
            piece_idx += 1
    
    # Current player (1 plane)
    state[12] = 1.0 if board.turn == PieceColor.WHITE else 0.0
    
    # Castling rights (4 planes)
    # White kingside
    state[13] = float(board.can_castle_kingside(PieceColor.WHITE))
    # White queenside
    state[14] = float(board.can_castle_queenside(PieceColor.WHITE))
    # Black kingside
    state[15] = float(board.can_castle_kingside(PieceColor.BLACK))
    # Black queenside
    state[16] = float(board.can_castle_queenside(PieceColor.BLACK))
    
    # En passant target (2 planes)
    if board.last_move and board.last_move.piece.type == PieceType.PAWN and \
       abs(board.last_move.start_pos.row - board.last_move.end_pos.row) == 2:
        # En passant is possible
        state[17] = 1.0
        # En passant target column
        col = board.last_move.end_pos.col
        state[18, :, col] = 1.0
    
    return torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension

def move_to_policy_idx(from_pos, to_pos):
    """
    Convert a move to its policy index using a simpler encoding:
    from_square (64 options) x to_square (64 options) = 4096 total indices
    """
    from_idx = from_pos.row * BOARD_SIZE + from_pos.col
    to_idx = to_pos.row * BOARD_SIZE + to_pos.col
    return from_idx * BOARD_SIZE * BOARD_SIZE + to_idx

def policy_idx_to_move(idx):
    """
    Convert a policy index back to a move using the simpler encoding
    """
    # In our simplified encoding:
    # idx = from_idx * 64 + to_idx
    to_idx = idx % (BOARD_SIZE * BOARD_SIZE)
    from_idx = idx // (BOARD_SIZE * BOARD_SIZE)
    
    # Convert indices to coordinates
    from_row, from_col = from_idx // BOARD_SIZE, from_idx % BOARD_SIZE
    to_row, to_col = to_idx // BOARD_SIZE, to_idx % BOARD_SIZE
    
    # Create Position objects
    from_pos = Position(from_row, from_col)
    to_pos = Position(to_row, to_col)
    
    return from_pos, to_pos