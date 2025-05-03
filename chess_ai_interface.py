import sys
import os
import time

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import game components
from src.chess_board import ChessBoard, Position, PieceType, PieceColor, Piece, Move
from src.chess_game import ChessGame
import pygame
import threading

class ChessAIInterface:
    """
    Chess AI Interface for managing the chess game and AI interactions.
    """
    def __init__(self):
        """Initialize the interface"""
        self.game = None
        self.board = None
        self.game_thread = None
        self.white_ai = None  # AI for white pieces
        self.black_ai = None  # AI for black pieces
        self.game_running = False
        self.game_over = False
        
    def register_ai(self, color, ai_controller):
        """
        Register an AI to control a specific color
        
        Parameters:
        - color: PieceColor.WHITE or PieceColor.BLACK
        - ai_controller: An object with a get_move(board_state, valid_moves) method
        """
        if color == PieceColor.WHITE:
            self.white_ai = ai_controller
        elif color == PieceColor.BLACK:
            self.black_ai = ai_controller
        else:
            raise ValueError("Color must be WHITE or BLACK")
            
    def start_game(self):
        """Start the chess game with the graphical UI in a separate thread"""
        if self.game_thread and self.game_thread.is_alive():
            print("Game is already running")
            return False
            
        # Initialize pygame for the UI thread
        pygame.init()
        
        # Create a new game
        self.game = AIEnabledChessGame(self)
        self.board = self.game.board
        
        # Start the game in a separate thread
        self.game_running = True
        self.game_over = False
        self.game_thread = threading.Thread(target=self._run_game)
        self.game_thread.daemon = True
        self.game_thread.start()
        
        print("Game started.")
        ai_white = "AI" if self.white_ai else "Human"
        ai_black = "AI" if self.black_ai else "Human"
        print(f"White: {ai_white}, Black: {ai_black}")
        return True
        
    def _run_game(self):
        """Run the game in a separate thread"""
        try:
            self.game.run()
        finally:
            self.game_running = False
            pygame.quit()
            
    def get_board_state(self):
        """Get the current board state as a 2D array"""
        if not self.board:
            return None
            
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
        
    def get_valid_moves_for_position(self, row, col):
        """Get all valid moves for a piece at the specified position"""
        if not self.board:
            return []
            
        if not (0 <= row < 8 and 0 <= col < 8):
            return []
            
        position = Position(row, col)
        return self.board.get_valid_moves(position)
        
    def get_valid_moves_for_color(self, color):
        """Get all valid moves for a specific color"""
        if not self.board:
            return []
            
        all_moves = []
        for row in range(8):
            for col in range(8):
                position = Position(row, col)
                piece = self.board.get_piece(position)
                if piece.color == color:
                    moves = self.board.get_valid_moves(position)
                    for move in moves:
                        all_moves.append(move)
        return all_moves
        
    def get_current_turn(self):
        """Get the current player's turn"""
        if not self.board:
            return None
        return self.board.turn
        
    def is_game_over(self):
        """Check if the game is over"""
        if not self.board:
            return {"game_over": False}
            
        opponent_color = PieceColor.BLACK if self.board.turn == PieceColor.WHITE else PieceColor.WHITE
        
        if self.board.is_checkmate(self.board.turn):
            self.game_over = True
            return {
                "game_over": True,
                "result": f"Checkmate! {opponent_color.value.capitalize()} wins!"
            }
        elif self.board.is_stalemate(self.board.turn):
            self.game_over = True
            return {
                "game_over": True,
                "result": "Stalemate! Draw."
            }
        elif self.board.is_fifty_move_rule_draw():
            self.game_over = True
            return {
                "game_over": True,
                "result": "Draw by fifty-move rule."
            }
        elif self.board.is_threefold_repetition():
            self.game_over = True
            return {
                "game_over": True,
                "result": "Draw by threefold repetition."
            }
        else:
            return {
                "game_over": False
            }
    
    def close(self):
        """Close the game"""
        self.game_running = False
        if self.game_thread and self.game_thread.is_alive():
            self.game_thread.join(timeout=1.0)


class AIEnabledChessGame(ChessGame):
    """
    Extension of the ChessGame class that allows AIs to control players
    """
    def __init__(self, ai_interface):
        """Initialize the game with an AI interface"""
        super().__init__()
        self.ai_interface = ai_interface
        
    def run(self):
        """Main game loop with AI integration"""
        running = True
        while running and self.ai_interface.game_running:
            # Process AI moves if it's an AI's turn
            current_turn = self.board.turn
            ai_player = None
            
            if current_turn == PieceColor.WHITE and self.ai_interface.white_ai:
                ai_player = self.ai_interface.white_ai
            elif current_turn == PieceColor.BLACK and self.ai_interface.black_ai:
                ai_player = self.ai_interface.black_ai
                
            # If it's an AI's turn, get and execute its move
            if not self.game_over and ai_player:
                # Get board state for the AI
                board_state = self.ai_interface.get_board_state()
                
                # Get valid moves for the current color
                valid_moves = self.ai_interface.get_valid_moves_for_color(current_turn)
                
                if valid_moves:
                    # Ask the AI for a move
                    ai_move = ai_player.get_move(board_state, valid_moves)
                    
                    if ai_move:
                        start_pos = ai_move.start_pos
                        end_pos = ai_move.end_pos
                        
                        # Execute the move
                        move_result = self.board.move_piece(start_pos, end_pos)
                        if move_result:
                            # Check game state after move
                            self.check_game_state()
                            
                # Brief delay to make the game visually followable
                time.sleep(0.5)
                    
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Only handle clicks if it's a human player's turn
                        current_turn = self.board.turn
                        is_ai_turn = (current_turn == PieceColor.WHITE and self.ai_interface.white_ai) or \
                                    (current_turn == PieceColor.BLACK and self.ai_interface.black_ai)
                        if not is_ai_turn:
                            self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:  # Restart game with R key
                        self.reset_game()
                        
            # Clear the screen
            self.screen.fill((0, 0, 0))
            
            # Draw the board with all visual indicators
            self.draw_board()
            
            pygame.display.flip()
            self.clock.tick(60)
            
        pygame.quit()