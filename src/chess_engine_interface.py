from chess_board import ChessBoard, Position, PieceType, PieceColor, Piece, Move

class ChessEngineInterface:
    """
    A direct interface for AI programs to interact with the chess game without needing a server.
    This class provides methods to access the game state, make moves, and query the board directly.
    """
    def __init__(self):
        """Initialize a new chess game interface"""
        self.board = ChessBoard()
        
    def reset_game(self):
        """Reset the game to its initial state"""
        self.board = ChessBoard()
        return self._get_game_state()
        
    def get_board_state(self):
        """Get the current board state as a 2D array"""
        board_state = []
        for row in range(8):
            row_data = []
            for col in range(8):
                piece = self.board.board[row][col]
                row_data.append({
                    "type": piece.type.value,
                    "color": piece.color.value,
                    "has_moved": piece.has_moved
                })
            board_state.append(row_data)
        return board_state
        
    def get_piece_at_position(self, row, col):
        """Get piece information at a specific position"""
        if 0 <= row < 8 and 0 <= col < 8:
            piece = self.board.get_piece(Position(row, col))
            return {
                "type": piece.type.value,
                "color": piece.color.value,
                "has_moved": piece.has_moved
            }
        return None
        
    def get_valid_moves(self, row, col):
        """Get all valid moves for a piece at the specified position"""
        if not (0 <= row < 8 and 0 <= col < 8):
            return []
            
        position = Position(row, col)
        valid_moves = self.board.get_valid_moves(position)
        
        return [self._format_move(move) for move in valid_moves]
        
    def make_move(self, start_row, start_col, end_row, end_col):
        """Make a move on the board"""
        start_pos = Position(start_row, start_col)
        end_pos = Position(end_row, end_col)
        
        move_result = self.board.move_piece(start_pos, end_pos)
        
        if not move_result:
            return {"success": False, "error": "Invalid move"}
            
        return {
            "success": True,
            "move_executed": self._format_move(move_result),
            "game_state": self._get_game_state()
        }
        
    def get_current_turn(self):
        """Get the current player's turn"""
        return self.board.turn.value
        
    def is_game_over(self):
        """Check if the game is over"""
        opponent_color = PieceColor.BLACK if self.board.turn == PieceColor.WHITE else PieceColor.WHITE
        
        if self.board.is_checkmate(self.board.turn):
            return {
                "game_over": True,
                "result": f"Checkmate! {opponent_color.value.capitalize()} wins!"
            }
        elif self.board.is_stalemate(self.board.turn):
            return {
                "game_over": True,
                "result": "Stalemate! Draw."
            }
        elif self.board.is_fifty_move_rule_draw():
            return {
                "game_over": True,
                "result": "Draw by fifty-move rule."
            }
        elif self.board.is_threefold_repetition():
            return {
                "game_over": True,
                "result": "Draw by threefold repetition."
            }
        else:
            return {
                "game_over": False
            }
            
    def get_fen(self):
        """Get the FEN notation of the current board state"""
        piece_symbols = {
            (PieceType.PAWN.value, PieceColor.WHITE.value): 'P',
            (PieceType.ROOK.value, PieceColor.WHITE.value): 'R',
            (PieceType.KNIGHT.value, PieceColor.WHITE.value): 'N',
            (PieceType.BISHOP.value, PieceColor.WHITE.value): 'B',
            (PieceType.QUEEN.value, PieceColor.WHITE.value): 'Q',
            (PieceType.KING.value, PieceColor.WHITE.value): 'K',
            (PieceType.PAWN.value, PieceColor.BLACK.value): 'p',
            (PieceType.ROOK.value, PieceColor.BLACK.value): 'r',
            (PieceType.KNIGHT.value, PieceColor.BLACK.value): 'n',
            (PieceType.BISHOP.value, PieceColor.BLACK.value): 'b',
            (PieceType.QUEEN.value, PieceColor.BLACK.value): 'q',
            (PieceType.KING.value, PieceColor.BLACK.value): 'k',
        }
        
        fen_parts = []
        
        # Board position
        for row in range(8):
            empty_count = 0
            row_str = ""
            
            for col in range(8):
                piece = self.board.board[row][col]
                if piece.type == PieceType.EMPTY:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += piece_symbols.get((piece.type.value, piece.color.value), '')
            
            if empty_count > 0:
                row_str += str(empty_count)
                
            fen_parts.append(row_str)
        
        position = '/'.join(fen_parts)
        
        # Active color
        active_color = 'w' if self.board.turn == PieceColor.WHITE else 'b'
        
        # Castling availability
        castling = ''
        if self.board.can_castle_kingside(PieceColor.WHITE):
            castling += 'K'
        if self.board.can_castle_queenside(PieceColor.WHITE):
            castling += 'Q'
        if self.board.can_castle_kingside(PieceColor.BLACK):
            castling += 'k'
        if self.board.can_castle_queenside(PieceColor.BLACK):
            castling += 'q'
        
        castling = castling or '-'
        
        # En passant target square
        en_passant = '-'
        if self.board.last_move and self.board.last_move.piece.type == PieceType.PAWN:
            if abs(self.board.last_move.start_pos.row - self.board.last_move.end_pos.row) == 2:
                # A pawn moved two squares
                file_letter = chr(ord('a') + self.board.last_move.end_pos.col)
                rank = 8 - ((self.board.last_move.start_pos.row + self.board.last_move.end_pos.row) // 2)
                en_passant = f"{file_letter}{rank}"
        
        # Halfmove clock and fullmove number
        halfmove_clock = str(self.board.half_move_clock)
        fullmove_number = str(len(self.board.move_history) // 2 + 1)
        
        return f"{position} {active_color} {castling} {en_passant} {halfmove_clock} {fullmove_number}"
        
    def _format_move(self, move):
        """Format a move object into a dictionary"""
        return {
            "start_pos": {"row": move.start_pos.row, "col": move.start_pos.col},
            "end_pos": {"row": move.end_pos.row, "col": move.end_pos.col},
            "piece": {
                "type": move.piece.type.value,
                "color": move.piece.color.value,
                "has_moved": move.piece.has_moved
            },
            "captured_piece": {
                "type": move.captured_piece.type.value,
                "color": move.captured_piece.color.value,
                "has_moved": move.captured_piece.has_moved
            } if move.captured_piece else None,
            "is_castling": move.is_castling,
            "is_en_passant": move.is_en_passant,
            "promotion_piece": {
                "type": move.promotion_piece.type.value,
                "color": move.promotion_piece.color.value,
                "has_moved": move.promotion_piece.has_moved
            } if move.promotion_piece else None
        }
        
    def _get_game_state(self):
        """Get the complete current game state"""
        game_state = {
            "board": self.get_board_state(),
            "turn": self.board.turn.value,
            "check": self.board.is_check(self.board.turn),
            "checkmate": self.board.is_checkmate(self.board.turn),
            "stalemate": self.board.is_stalemate(self.board.turn),
            "half_move_clock": self.board.half_move_clock
        }
        
        # Add game over info
        game_over_info = self.is_game_over()
        game_state.update(game_over_info)
        
        return game_state