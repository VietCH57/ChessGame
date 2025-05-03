import random
from src.chess_board import PieceColor

class RandomChessAI:
    """A simple chess AI that makes random legal moves"""
    def __init__(self, name="RandomAI"):
        self.name = name
    
    def get_move(self, board_state, valid_moves):
        """Choose a random move from valid moves"""
        if not valid_moves:
            return None
        return random.choice(valid_moves)


class PrioritizeCapturingAI:
    """Chess AI that prioritizes capturing opponent pieces"""
    def __init__(self, name="CaptureAI"):
        self.name = name
    
    def get_move(self, board_state, valid_moves):
        """Choose a move that captures an opponent piece if possible"""
        if not valid_moves:
            return None
            
        # First look for capturing moves
        capturing_moves = [move for move in valid_moves 
                          if move.captured_piece and move.captured_piece.type.value != "empty"]
        
        if capturing_moves:
            return random.choice(capturing_moves)
        
        # If no capturing moves, make a random move
        return random.choice(valid_moves)